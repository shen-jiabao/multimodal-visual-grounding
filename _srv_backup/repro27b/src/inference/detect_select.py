"""detect-then-select：把「序数/方位推理」拆成「枚举 + 确定性排序」两半（P2-1 原型）。

动机
----
复赛 19.4% 的 query 含序数词。人工核对 47 场景 / 116 条 query 的结论是：
8B VLM 能盖到**某个同类物体**，但盖不到 query **指定的那一个**——让它做
「从右数第三个」这种计数/排序推理，是让它做最不擅长的事。

本模块把它拆开：

    第 1 步（模型擅长的）："List EVERY <NP>" → 逐个给框（枚举，不判断序数）
    第 2 步（确定性计算）：按 x/y 中心排序取第 k 个；远近类用深度图 ROI 均值排序

**零样本约束内的做法**：不训练、不伪标注，只改提示词与后处理。

设计要点：约束「抹除」而非「切分」
--------------------------------
序数词经常嵌在名词短语**中间**（``The second white patio umbrella ... from
left to right``）。若把「序数→方位」当作一个连续 span 切掉，会把物体本身一起切掉
（实测产出 ``np='The'``）。故 :func:`extract_target` 改为**多段抹除**：只删掉
序数词、方位短语、远近短语各自的 span，剩下的即名词短语。

边界与风险
----------
- 本模块只做**纯函数**（NP 抽取 / 约束解析 / 索引选择），不碰 torch、不碰模型。
- 它**不保证**提分：与 ``chain_enforce`` 的教训一致，任何改动都要先过
  ``scripts/复赛/provable_errors.py`` 守门器。
- 与 :func:`src.inference.matcher.parse_spatial_constraint` **刻意分开**：后者被
  09-24 基线共用，就地扩展会改变基线行为、污染 A/B 归因。:func:`parse_constraint_ex`
  是其超集，只在 ``inference.detect_select != off`` 时被消费。
- NP 是**近似**的（只求让模型知道该枚举哪类物体），不追求语言学正确。
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence, Tuple

from src.utils.text import clean_query

# ---------------------------------------------------------------- 词表
_ORD = {
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
    "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10,
    "eleventh": 11, "twelfth": 12,
}
_ORD_PAT = "|".join(_ORD)
_ORD_RE = re.compile(r"\b(" + _ORD_PAT + r")\b", re.IGNORECASE)

# 这些词前的 first/second 不是序数（first aid / first floor），不参与约束
_ORD_BLACKLIST = {"aid", "floor", "class", "place", "time", "base", "hand"}

# 方位词 -> (轴, 方向)。方向 +1 表示 rank 1 落在**坐标值最小**的一端
# （left / top），-1 表示落在坐标值最大的一端（right / bottom）。
_DIR2AXIS: Dict[str, Tuple[str, int]] = {
    "left": ("x", 1), "right": ("x", -1),
    "top": ("y", 1), "bottom": ("y", -1),
}
_DEPTH_NEAR = ("closest", "nearest", "closer", "frontmost")
_DEPTH_FAR = ("farthest", "furthest", "farther", "backmost")
_DEPTH_PAT = "|".join(_DEPTH_NEAR + _DEPTH_FAR)
_DIR_PAT = "left|right|top|bottom"
_NEARFAR_PAT = "near|far|front|back"
_IC = re.IGNORECASE

# ---------------------------------------------------------------- 约束正则
# 语序②："from left to right, the first ..."（序数在方位短语**之后**）
_RE_A2B = re.compile(
    r"from\s+(" + _DIR_PAT + r")\s+to\s+(?:" + _DIR_PAT + r")", _IC)
# "second lowest floor"（楼层序数：lowest = 图下方 = y 增大 = direction -1）
_RE_FLOOR = re.compile(r"\b(" + _ORD_PAT + r")\b\s+(lowest|highest)\s+floor\b", _IC)
# "second crane farthest from the camera"
# 修饰词允许连字符（wall-mounted / ring-tailed），否则 
# "second rectangular wall-mounted box from the left" 会整条解析失败。
_MOD = r"[\w-]+"
_RE_DEPTH = re.compile(
    r"\b(" + _ORD_PAT + r")\b\s+(?:" + _MOD + r"\s+){0,4}?(" + _DEPTH_PAT + r")\b", _IC)
# "counted from near to far"（无方位词的深度序）
_RE_NEARFAR = re.compile(
    r"(?:counted\s+)?from\s+(" + _NEARFAR_PAT + r")\s+to\s+(" + _NEARFAR_PAT + r")", _IC)
# -est 方位词
_RE_MOST = re.compile(r"\b(leftmost|rightmost|topmost|bottommost)\b", _IC)
# 语序①："the first black swan from bottom to top"
# 长度不设上限：名词短语里可插任意多修饰词（实测最长 9 词才到 from），
# 设 {0,6} 会让 11 条 query 掉出解析（相对老实现成为**子集**，不可接受）。
_RE_ORD_FROM = re.compile(
    r"\b(" + _ORD_PAT + r")\b\s+(?:" + _MOD + r"\s+)*?from\s+(?:the\s+)?(" + _DIR_PAT + r")", _IC)
# 单独的 "from <dir>"（抹除方位短语用）
_RE_FROM_DIR = re.compile(r"\bfrom\s+(?:the\s+)?(" + _DIR_PAT + r")\b", _IC)
# 裸远近词："the bear closest to the camera"
_RE_DEPTH_BARE = re.compile(
    r"\b(" + _DEPTH_PAT + r")\b(?:\s+(?:to|from)\s+(?:the\s+)?camera)?", _IC)
# 中间 / 居中
_RE_MIDDLE = re.compile(r"\b(?:in|at)\s+the\s+(middle|center|centre)\b", _IC)
# 深度短语的完整形态（抹除时连 "closest to the camera" 一起去掉）
_RE_DEPTH_PHRASE = re.compile(
    r"\b(" + _DEPTH_PAT + r")\b(?:\s+(?:to|from)\s+(?:the\s+)?camera)?"
    r"(?:\s+from\s+(?:" + _NEARFAR_PAT + r")\s+to\s+(?:" + _NEARFAR_PAT + r"))?", _IC)

# ---------------------------------------------------------------- NP 清洗
_LEAD_ARTICLE = re.compile(
    r"^(?:it\s+is\s+|there\s+is\s+|this\s+is\s+|that\s+is\s+|count\s+the\s+|"
    r"another\s+|the\s+|a\s+|an\s+)", _IC)
_NUM_WORDS_PAT = "one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve"
_LEAD_OF = re.compile(
    r"^of\s+(?:" + _ORD_PAT + r"|" + _NUM_WORDS_PAT + r"|\d+)\s+", _IC)
# 尾部悬空修饰（介词 / 系词 / 方位词结尾）。**必须锚定在末尾**（``$``）：
# 不锚定会把 "umbrella in the outdoor area" 中间的 "in the" 也删掉，
# 粘出 "umbrellaoutdoor" 这类伪词，并吃掉 "buildings of similar height" 的 "of"。
_TRAIL_JUNK = re.compile(
    r"(?:[\s.,;:!?]+\b(?:on|in|at|of|from|to|for|with|and|or|the|a|an|"
    r"that|which|who|whose|is|are|was|were|be|been|being|"
    r"below|above|beside|near|behind|under|over|left|right|top|bottom)\b)+"
    r"[\s.,;:!?]*$", _IC)
_TRAIL_PUNCT = re.compile(r"[\s.,;:!?]+$")

# NP 有效性判定的停用词（至少要有 1 个实词）
_STOP = {
    "the", "a", "an", "of", "on", "in", "to", "from", "and", "or", "is", "it",
    "that", "this", "with", "at", "by", "for", "as", "its", "their", "there",
    "when", "which", "who", "whom", "are", "was", "were", "be", "has", "have",
    "had", "his", "her", "then", "than", "also", "up", "down", "out", "over",
}

# 名词短语最大词数（超过即视为描述型 query，不适合枚举）
_NP_MAX_WORDS = 14


def _mk(axis: str, direction: int, rank: Optional[int], kind: str) -> dict:
    return {"axis": axis, "direction": int(direction),
            "rank": None if rank is None else int(rank), "kind": kind}


def _ws(text: str) -> str:
    """去零宽字符 + 折叠空白（保留原始大小写，便于提示词可读）。"""
    return " ".join(clean_query(text).split())


def _ordinal_spans(s: str) -> List[Tuple[int, int]]:
    """所有「真序数词」的 span；``first aid`` 这类黑名单组合不算。"""
    out = []
    for m in _ORD_RE.finditer(s):
        nxt = s[m.end():].lstrip().split(" ", 1)[0].lower().strip(".,")
        if nxt in _ORD_BLACKLIST:
            continue
        out.append(m.span())
    return out


# ---------------------------------------------------------------- 约束解析
def parse_constraint_ex(query: str) -> Tuple[Optional[dict], List[Tuple[int, int]]]:
    """扩展版空间/序数约束解析。

    返回 ``(constraint, spans)``：

    - ``constraint``：``{axis, direction, rank, kind}``；``kind`` ∈
      ``{"pos", "depth", "middle"}``；``rank=None`` 仅用于 ``middle``。
    - ``spans``：**需要在文本上抹除**的下标区间列表（在 :func:`_ws` 归一后的
      字符串上），供 :func:`extract_target` 得到名词短语。两者同时为空/None
      表示未识别出任何约束。

    相对 :func:`src.inference.matcher.parse_spatial_constraint` 新增：
    ``<ord> ... from the <dir>`` 之外的 ``from A to B, the <ord>`` 四种组合、
    ``<ord> lowest/highest floor``、``<ord> closest/farthest``、
    ``from near to far``、``in the middle``。
    """
    if not query or not query.strip():
        return None, []
    s = _ws(query)

    # 1) <ord> lowest/highest floor —— 序数 + 楼层
    m = _RE_FLOOR.search(s)
    if m:
        return (_mk("y", 1 if m.group(2).lower() == "highest" else -1,
                    _ORD[m.group(1).lower()], "pos"),
                [m.span(1), m.span(2)])

    # 2) <ord> ... closest/farthest —— 序数 + 远近
    m = _RE_DEPTH.search(s)
    if m:
        near = m.group(2).lower() in _DEPTH_NEAR
        spans = [m.span(1)]
        ph = _RE_DEPTH_PHRASE.match(s, m.start(2))
        if ph:
            spans.append(ph.span())
        return _mk("y", 1 if near else -1, _ORD[m.group(1).lower()], "depth"), spans

    # 3) -est 方位词（rank 固定 1）
    m = _RE_MOST.search(s)
    if m:
        axis, sign = _DIR2AXIS[m.group(1).lower()[: -len("most")]]
        return _mk(axis, sign, 1, "pos"), [m.span()]

    # 4) <ord> ... from <dir> [to <dir>]（语序①；覆盖 95% 的序数 query）
    #    抹除的必须是**整个** "from the left" / "from left to right" 短语，
    #    只抹方位词会留下 "from the"（实测产出 'box from the on the back wall'）。
    m = _RE_ORD_FROM.search(s)
    if m:
        axis, sign = _DIR2AXIS[m.group(2).lower()]
        spans = [m.span(1)]
        a2b = _RE_A2B.search(s)
        if a2b:
            spans.append(a2b.span())
        else:
            fr = None
            for cand in _RE_FROM_DIR.finditer(s, 0, m.end(2)):
                fr = cand          # 取 group(2) 之前**最后**一个 "from <dir>"
            spans.append(fr.span() if fr else m.span(2))
        return _mk(axis, sign, _ORD[m.group(1).lower()], "pos"), spans

    # 5) from <A> to <B>[,] the <ord>（语序②，含四种组合）
    #    序数可能出现在方位短语**之前**（"From left to right, the first ..."）
    #    或**之后**（"... the first ... from left to right"）；两向都要找，
    #    否则 rank 会静默退化成 1（实测 573 条语序②里绝大多数属前者）。
    m = _RE_A2B.search(s)
    if m:
        axis, sign = _DIR2AXIS[m.group(1).lower()]
        rank, spans = 1, [m.span()]
        mo = _ORD_RE.search(s, m.end())
        if mo and mo.start() - m.end() < 30:
            rank, spans = _ORD[mo.group(1).lower()], spans + [mo.span()]
        else:
            mo = None
            for cand in _ORD_RE.finditer(s, 0, m.start()):
                mo = cand
            if mo is not None:
                rank, spans = _ORD[mo.group(1).lower()], spans + [mo.span()]
        return _mk(axis, sign, rank, "pos"), spans

    # 6) from near to far（深度序；序数可能在前面）
    m = _RE_NEARFAR.search(s)
    if m:
        near = m.group(1).lower() in ("near", "front")
        rank, spans = 1, [m.span()]
        mo = _ORD_RE.search(s)
        if mo and mo.start() < m.start():
            rank = _ORD[mo.group(1).lower()]
            spans.append(mo.span())
        return _mk("y", 1 if near else -1, rank, "depth"), spans

    # 7) 裸远近词（无「from A to B」时的兜底）
    m = _RE_DEPTH_BARE.search(s)
    if m:
        near = m.group(1).lower() in _DEPTH_NEAR
        return _mk("y", 1 if near else -1, 1, "depth"), [m.span()]

    # 8) 中间 / 居中
    m = _RE_MIDDLE.search(s)
    if m:
        return _mk("x", 1, None, "middle"), [m.span()]

    return None, []


# ---------------------------------------------------------------- NP 抽取
def _cleanup(text: str) -> str:
    """去掉残留的引导词 / 悬空介词 / 标点，得到名词短语。"""
    s = " ".join(text.split())
    for _ in range(8):
        before = s
        s = _LEAD_ARTICLE.sub("", s)
        s = _LEAD_OF.sub("", s)
        s = _TRAIL_PUNCT.sub("", s)
        s = _TRAIL_JUNK.sub("", s)
        s = " ".join(s.split())
        if s == before:
            break
    return _TRAIL_PUNCT.sub("", s).strip()


def is_valid_np(text: str) -> bool:
    """名词短语是否可用作枚举目标（非空、不过长、含实词）。"""
    ws = text.split()
    if not ws or len(ws) > _NP_MAX_WORDS:
        return False
    return any(len(w) > 1 and w.lower() not in _STOP for w in ws)


def _mask(s: str, spans: Sequence[Tuple[int, int]]) -> str:
    """把若干 span 从字符串里抹掉（用空格占位，保持其余下标不变）。

    抹掉的位置**两端各补一个空格**，避免出现 ``umbrellaoutdoor`` 这类
    被粘连的伪词（抹除区间恰好从空格处开始时会发生）。
    """
    if not spans:
        return s
    chars = list(s)
    for a, b in spans:
        for i in range(max(0, a), min(len(chars), b)):
            chars[i] = " "
        if 0 <= a < len(chars):
            chars[a] = " "
        if 0 <= b < len(chars):
            chars[b] = " "
    return "".join(chars)


def extract_target(query: str) -> Optional[dict]:
    """query → ``{"query", "np", "constraint"}``；抽不出可用 NP 时返回 None。

    ``np`` 是**抹掉约束后剩余的名词短语**（保留 ``on the left fence`` 这类容器
    短语）——同一张图里同类的多条 query 会抽到同一个 NP，可直接作为枚举分组的键。
    """
    if not query or not query.strip():
        return None
    cons, spans = parse_constraint_ex(query)
    np_text = _cleanup(_mask(_ws(query), spans))
    if not is_valid_np(np_text):
        return None
    return {"query": query, "np": np_text, "constraint": cons}


# ---------------------------------------------------------------- 确定性选择
def box_centers(boxes: Sequence[Sequence[float]], axis: str) -> List[float]:
    """一组 xyxy 框在给定轴上的中心坐标（``axis`` ∈ {"x", "y"}）。"""
    idx = 0 if axis == "x" else 1
    return [(float(b[idx]) + float(b[idx + 2])) / 2.0 for b in boxes]


def select_index(values: Sequence[float], constraint: Optional[dict]) -> Optional[int]:
    """按约束在 ``values`` 里选出目标下标（第 2 步）。

    ``values`` 对 ``kind="pos"`` 是框中心坐标，对 ``kind="depth"`` 是 ROI 平均深度
    （越小越近）。方向语义：``direction=+1`` 时 rank 1 取最小值，``-1`` 取最大值。
    越界 rank 夹到端点；``middle`` 取排序后的中间项。
    """
    n = len(values)
    if n == 0 or not constraint:
        return None
    if constraint.get("kind") == "middle":
        order = sorted(range(n), key=lambda i: values[i])
        return order[len(order) // 2]
    rank = constraint.get("rank")
    if rank is None:
        return None
    order = sorted(range(n), key=lambda i: values[i],
                   reverse=(int(constraint.get("direction", 1)) < 0))
    return order[min(max(int(rank), 1), n) - 1]


def select_box(boxes: Sequence[Sequence[float]], constraint: Optional[dict],
               depth_means: Sequence[float] | None = None
               ) -> Tuple[Optional[Sequence[float]], Optional[int]]:
    """在候选框里按约束选出目标框，返回 ``(box, index)``。

    ``kind="depth"`` 必须提供 ``depth_means``（每个框的 ROI 平均深度）；
    缺失时返回 ``(None, None)`` —— **宁可回退，也不猜**。
    """
    if not boxes or not constraint:
        return None, None
    if constraint.get("kind") == "depth":
        if depth_means is None:
            return None, None
        idx = select_index(list(depth_means), constraint)
    else:
        idx = select_index(box_centers(boxes, constraint.get("axis", "x")), constraint)
    if idx is None:
        return None, None
    return boxes[idx], idx

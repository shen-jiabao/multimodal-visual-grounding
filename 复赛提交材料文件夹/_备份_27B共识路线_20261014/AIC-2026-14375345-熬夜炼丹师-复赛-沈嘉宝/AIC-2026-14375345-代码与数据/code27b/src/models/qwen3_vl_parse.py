"""Qwen3-VL 输出解析与坐标换算（纯函数，不依赖 torch / transformers，便于单测）。

输出解析要点：
    - Qwen3-VL 输出坐标体系为 **[0,1000] 归一化**（区别于 Qwen2.5-VL 的像素坐标）；
    - 输出模式①单框 JSON 与模式③同图多框共用本模块的容错解析；
    - 坐标换算与坐标非均匀缩放修正在此实现，供封装层与单测复用。

设计约定：本模块只做"文本→[0,1000] 框→像素框/[0,1] 框"的纯计算，不做合法性
判定（由封装层复用 ``postprocess.bbox.is_valid_bbox``）。
"""
from __future__ import annotations

import json
import re
from typing import List, Optional, Sequence, Tuple

from src.utils.text import clean_prompt_text, clean_query

# Qwen3-VL 归一化坐标上限
COORD_MAX = 1000.0

# 可能的框字段名（不同版本/提示词下模型可能用不同键）
_BOX_KEYS = ("bbox_2d", "bbox", "box", "bbox_2d_1000", "coordinates")
# 可能的标签字段名
_LABEL_KEYS = ("label", "target", "name", "object", "description", "query")
# 可能的数字 ID 字段名（ID 化输出：用 target 序号代替整句回显）
_ID_KEYS = ("id", "index", "idx", "target_id", "query_id", "target_index")

_RE_JSON_OBJ = re.compile(r"\{[^{}]*\}", re.DOTALL)
_RE_NUM = re.compile(r"-?\d+(?:\.\d+)?")
_RE_ARRAY = re.compile(r"\[[^\[\]]*\]")


# ---------------- 提示词构造 ----------------

def _modality_preamble(modality_labels: Sequence[str] | None) -> str:
    """多图输入的图像角色说明（图序 + bbox 落在图 1）。无则返回空串。"""
    if not modality_labels:
        return ""
    lines = "\n".join(f"Image {i + 1}: {lab}" for i, lab in enumerate(modality_labels))
    return (
        f"You are given {len(modality_labels)} images of the SAME scene, in this order:\n"
        f"{lines}\n"
        "Report the target's bounding box on Image 1 (visible RGB).\n"
    )


def build_query_prompt(query: str, modality_labels: Sequence[str] | None = None) -> str:
    """模式①：逐查询单框定位指令（要求 [0,1000] 的 JSON）。

    ``modality_labels`` 非空时前置「多图角色说明」（多模态多图输入）。
    """
    return (
        _modality_preamble(modality_labels)
        + "Locate the single target described below in the image. "
        'Return ONLY a JSON object of the form {"bbox_2d": [x1, y1, x2, y2]} '
        "where each coordinate is an integer in the normalized range [0, 1000]. "
        "Rules: "
        "(1) When a query says 'the first/second/third/leftmost/rightmost' object, START SCANNING from the specified direction edge (left edge for left-to-right), count each matching object one by one in order, and box ONLY the nth one. Do NOT pick the largest or most prominent object; pick the one at the exact ordinal position. Do not include adjacent objects. "
        "(2) Determine left/right reference from the query: if it says 'X's left/right' or 'left/right of X', use X's own left/right (if X faces you, X's right is on your left); if it says 'on the left/right of the image/scene' or 'leftmost/rightmost in the frame', use the image's left/right. "
        "(3) When a query asks for a part of an object (headlight, logo, label, hand, mirror, wheel), box ONLY that small part, not the whole object. "
        "(4) Box tightly around the target with minimal background. "
        "(5) IF the query mentions a color word (white/red/black/green/blue/yellow/orange/etc.), that color is a HARD constraint: do NOT box an object of the wrong color even if its position seems right. Keep scanning until you find an object matching BOTH the position AND the color. If no color is mentioned in the query, ignore this rule. "
        "Do not output any other text.\n"
        f"Target description: {clean_prompt_text(query)}"
    )


def build_multi_prompt(queries: Sequence[str],
                       modality_labels: Sequence[str] | None = None,
                       use_id: bool = False) -> str:
    """模式③：同图多框定位指令（返回 JSON 列表）。

    ``use_id=False``（缺省，保持既有契约）：要求模型**回显整句** label，
    形如 ``{"label": <target text>, "bbox_2d": [...]}``。

    ``use_id=True``：Targets 带序号，要求模型只回**序号**，
    形如 ``{"id": <target number>, "bbox_2d": [...]}``。三重收益：

      · 输出长度大幅缩短（一条整句 label ≈ 20 token → 一个数字 ≈ 1~2 token），
        直接缓解 multi_box 的回复截断（占位框的唯一成因）；
      · 归属不再有歧义 —— 数字 ID 不会匹配错，`match_label_to_query` 的
        模糊归属（子串/词重叠）完全绕过；
      · 释放的 token 预算可换更多框。
    """
    if use_id:
        lines = "\n".join(f"{i + 1}. {clean_prompt_text(q)}"
                          for i, q in enumerate(queries))
        return (
            _modality_preamble(modality_labels)
            + "Locate EACH of the following targets in the image. Return ONLY a JSON list "
            'of objects, one per target: {"id": <target number>, '
            '"bbox_2d": [x1, y1, x2, y2]} where <target number> is the number '
            "printed before that target below, and each coordinate is an integer "
            "in the normalized range [0, 1000]. "
            "Rules: "
            "(1) When a query says 'the first/second/third/leftmost/rightmost' object, START SCANNING from the specified direction edge, count each matching object one by one in order, and box ONLY the nth one. Do NOT pick the largest or most prominent object. Do not include adjacent objects. "
            "(2) Determine left/right reference from the query: if it says 'X's left/right' or 'left/right of X', use X's own left/right (if X faces you, X's right is on your left); if it says 'on the left/right of the image/scene' or 'leftmost/rightmost in the frame', use the image's left/right. "
            "(3) When a query asks for a part of an object (headlight, logo, label, hand, mirror, wheel), box ONLY that small part, not the whole object. "
            "(4) Box tightly around the target with minimal background. "
            "(5) IF the query mentions a color word (white/red/black/green/blue/yellow/orange/etc.), that color is a HARD constraint: do NOT box an object of the wrong color even if its position seems right. If no color is mentioned in the query, ignore this rule. "
            "Do not output any other text.\nTargets:\n" + lines
        )

    lines = "\n".join(f"- {clean_prompt_text(q)}" for q in queries)
    return (
        _modality_preamble(modality_labels)
        + "Locate EACH of the following targets in the image. Return ONLY a JSON list "
        'of objects, one per target: {"label": <target text>, "bbox_2d": [x1, y1, x2, y2]} '
        "where each coordinate is an integer in the normalized range [0, 1000]. "
        "Rules: "
        "(1) When a query says 'the first/second/third/leftmost/rightmost' object, START SCANNING from the specified direction edge, count each matching object one by one in order, and box ONLY the nth one. Do NOT pick the largest or most prominent object. Do not include adjacent objects. "
        "(2) Determine left/right reference from the query: if it says 'X's left/right' or 'left/right of X', use X's own left/right (if X faces you, X's right is on your left); if it says 'on the left/right of the image/scene' or 'leftmost/rightmost in the frame', use the image's left/right. "
        "(3) When a query asks for a part of an object (headlight, logo, label, hand, mirror, wheel), box ONLY that small part, not the whole object. "
        "(4) Box tightly around the target with minimal background. "
        "(5) IF the query mentions a color word (white/red/black/green/blue/yellow/orange/etc.), that color is a HARD constraint: do NOT box an object of the wrong color even if its position seems right. If no color is mentioned in the query, ignore this rule. "
        "Do not output any other text.\nTargets:\n" + lines
    )


# ---------------- 解析 ----------------

def _coerce_box(val) -> Optional[List[float]]:
    """把候选值转成 4 个 float 的框；失败返回 None。"""
    if isinstance(val, dict):
        for k in _BOX_KEYS:
            if k in val:
                box = _coerce_box(val[k])
                if box is not None:
                    return box
        return None
    if isinstance(val, (list, tuple)) and len(val) == 4:
        try:
            return [float(v) for v in val]
        except (TypeError, ValueError):
            return None
    return None


def _box_from_obj(obj) -> Optional[List[float]]:
    if isinstance(obj, (list, tuple)):
        return _coerce_box(obj)
    if isinstance(obj, dict):
        for k in _BOX_KEYS:
            if k in obj:
                box = _coerce_box(obj[k])
                if box is not None:
                    return box
    return None


def _extract_label(obj) -> str:
    if isinstance(obj, dict):
        for k in _LABEL_KEYS:
            v = obj.get(k)
            if isinstance(v, str):
                return v
    return ""


def _extract_id(obj) -> Optional[int]:
    """从对象里取数字 target ID（模式③ ID 化输出）；取不到返回 None。

    容忍 int / float / 纯数字字符串；显式跳过 bool（``True`` 不是 ID）。
    """
    if not isinstance(obj, dict):
        return None
    for k in _ID_KEYS:
        v = obj.get(k)
        if isinstance(v, bool):
            continue
        if isinstance(v, (int, float)):
            return int(v)
        if isinstance(v, str):
            s = v.strip().lstrip("+-")
            if s.isdigit():
                return int(v.strip())
    return None


def parse_single_box(text: str) -> Optional[List[float]]:
    """从模型回复解析单个 [0,1000] 框；失败返回 None。

    容错：优先取 JSON 对象内的 bbox 键；退化到"最后一个长度 4 的数字数组"。
    取**最后**一个（模型最终答案通常在末尾；可跳过思考/草稿片段）。
    """
    if not text:
        return None
    objs = list(_RE_JSON_OBJ.finditer(text))
    for m in reversed(objs):
        try:
            obj = json.loads(m.group(0))
        except (json.JSONDecodeError, ValueError):
            continue
        box = _box_from_obj(obj)
        if box is not None:
            return box
    for arr in reversed(_RE_ARRAY.findall(text)):
        nums = _RE_NUM.findall(arr)
        if len(nums) == 4:
            return [float(n) for n in nums]
    return None


def parse_multi_boxes_ex(text: str) -> List[Tuple[Optional[int], str, List[float]]]:
    """解析同图多框回复 → ``[(target_id, label, box1000), ...]``（无序，取全部合法对象）。

    同时抽取数字 ID 与文本 label，两者都可能缺失（分别为 None / ""）。
    调用方优先用 ID 归属（模式③ ID 化输出），无 ID 时退回 label 匹配。
    """
    out: List[Tuple[Optional[int], str, List[float]]] = []
    if not text:
        return out
    for m in _RE_JSON_OBJ.finditer(text):
        try:
            obj = json.loads(m.group(0))
        except (json.JSONDecodeError, ValueError):
            continue
        box = _box_from_obj(obj)
        if box is None:
            continue
        out.append((_extract_id(obj), _extract_label(obj), box))
    return out


def parse_multi_boxes(text: str) -> List[Tuple[str, List[float]]]:
    """解析同图多框回复 → ``[(label, box1000), ...]``（无序，取全部合法对象）。

    向后兼容的薄封装（既有单测与调用方契约不变）；ID 化输出请用
    :func:`parse_multi_boxes_ex`。
    """
    return [(label, box) for _tid, label, box in parse_multi_boxes_ex(text)]


# ---------------- 坐标换算 ----------------

def coords_1000_to_unit(box: Sequence[float],
                        scale_x: float = 1.0,
                        scale_y: float = 1.0) -> List[float]:
    """[0,1000] 归一化 → [0,1] 域 xyxy。

    ``scale_x/scale_y`` 为可选的非均匀缩放修正系数（默认 1.0，如需修正须有实测依据）。
    """
    x1, y1, x2, y2 = (float(v) / COORD_MAX for v in box)
    x1, x2 = x1 * scale_x, x2 * scale_x
    y1, y2 = y1 * scale_y, y2 * scale_y
    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1
    return [x1, y1, x2, y2]


def coords_1000_to_pixel(box: Sequence[float], width: int, height: int,
                         scale_x: float = 1.0,
                         scale_y: float = 1.0) -> List[float]:
    """[0,1000] 归一化 → 像素 xyxy（保持模型层"像素坐标"契约）。"""
    ux1, uy1, ux2, uy2 = coords_1000_to_unit(box, scale_x, scale_y)
    return [ux1 * width, uy1 * height, ux2 * width, uy2 * height]


# ---------------- 标签 → 查询匹配（模式③前置） ----------------

def _norm(s: str) -> str:
    """匹配用的归一：与提示词侧同一套清洗（超集），再小写化。"""
    return clean_prompt_text(s).lower()


def infer_id_base(ids: Sequence[Optional[int]], n_queries: int) -> int:
    """从**一次回复**里全部 target ID 反推模型用的是 1-based 还是 0-based 编号。

    :func:`resolve_multi_target` 逐条无法判断编号基准（把 ``id=0`` 和 ``id=1``
    都兼容成 ``queries[0]`` 会碰撞），但**一次回复的 ID 全集**可以判断。保守规则
    （三条同时成立才判 0-based，否则一律 1-based）：

    1. 出现了 ``0``——1-based 模型永远不会吐 0；
    2. **没有**出现 ``n_queries``——排除「0 是杂讯、真正的编号仍是 1..n」；
    3. 全部 ID 都落在 ``[0, n_queries-1]``。

    判 0-based 时把越界的 ``n_queries`` 交给调用方回退 label 匹配，不猜。
    """
    vals = {int(i) for i in ids if i is not None}
    if not vals or 0 not in vals:
        return 1
    if n_queries in vals:
        return 1
    if any(v < 0 or v >= n_queries for v in vals):
        return 1
    return 0


def propagate_identical_queries(queries: Sequence[str],
                                got: Sequence[Tuple[str, List[float]]]
                                ) -> List[Tuple[str, List[float]]]:
    """同图**文本完全相同**的 query 之间补齐缺失框（弱占优）。

    ID 化提示词把 Targets 编号后，两条逐字相同的 query（实测 9 组 / 20 条）
    在模型眼里是同一个目标，可能只回一个 ``id``，另一条落空 → 占位框
    ``[0.49,0.49,0.51,0.51]``，与任何真值的 IoU ≈ 0。

    因此「把已得到的那个框复制给同文本的兄弟」是**弱占优**：只在某条**完全
    没拿到框**、且同图存在**逐字相同**且已拿到框的 query 时才动手；它不会
    改动任何已存在的框，最坏情况也只是把 IoU 0 换成另一个 IoU 0。

    Args:
        queries: 本次调用的全部 query（顺序即提示词编号顺序）。
        got: 已归属的 ``(query, box1000)``，可能不含部分 query。

    Returns:
        与 ``got`` 同构、但补齐了缺失兄弟的列表（原顺序 + 追加顺序）。
    """
    # 用**计数**而不是集合：同图两条逐字相同的 query 无法靠文本区分，
    # `q in have` 会把第二条兄弟也当成「已经有了」而漏补。
    remaining: dict = {}
    donor: dict = {}
    for q, b in got:
        k = clean_prompt_text(q)
        remaining[k] = remaining.get(k, 0) + 1
        donor.setdefault(k, b)          # 首次出现即作供体，保证确定性
    out = list(got)
    for q in queries:
        k = clean_prompt_text(q)
        n = remaining.get(k, 0)
        if n > 0:
            remaining[k] = n - 1        # 这次调用已有归属，不重复补
            continue
        src = donor.get(k)
        if src is not None:
            out.append((q, list(src)))  # 必须复制：不能与供体共享列表
    return out


def resolve_multi_target(target_id: Optional[int], label: str,
                         queries: Sequence[str],
                         id_base: int = 1) -> Optional[str]:
    """把模式③的一条输出归属到 ``queries`` 里的某条查询。

    提示词把 Targets 编号为 ``1..n``（见 :func:`build_multi_prompt`），故规则是
    **严格 1-based**：``id == id_base + k`` ⇒ ``queries[k]``；越界视为无效 ID。

    无 ID（模型忽略 ID 化提示词、仍回显 label）或 ID 越界时，退回
    :func:`match_label_to_query` 的文本匹配，保证行为不退化。

    为什么不在这里「兼容 0-based 漂移」：单条记录无法判断模型到底用了哪套编号——
    若模型系统性 0-based，逐条兼容会把 ``id=0`` 和 ``id=1`` **都**映射到
    ``queries[0]``（碰撞），比严格 1-based 更糟。正确做法是**让它可观测**：
    diag 落盘的 ``pairs[].target_id`` 会记录模型实际吐出的 ID，一次云端运行
    即可判定是否存在漂移，届时用 ``id_base=0`` 整批重解即可（无需重跑模型）。
    """
    if target_id is not None:
        k = int(target_id) - int(id_base)
        if 0 <= k < len(queries):
            return queries[k]
    return match_label_to_query(label, queries)


def match_label_to_query(label: str, queries: Sequence[str]) -> Optional[str]:
    """把模型自由文本 label 归属到最匹配的查询文本。

    依次尝试：精确 → 双向子串 → 词重叠最大（重叠为 0 视为不匹配，返回 None）。
    """
    lab = _norm(label)
    if not lab:
        return None
    for q in queries:
        if _norm(q) == lab:
            return q
    for q in queries:
        nq = _norm(q)
        if nq and (lab in nq or nq in lab):
            return q
    best, best_n = None, 0
    lab_words = set(lab.split())
    for q in queries:
        n = len(lab_words & set(_norm(q).split()))
        if n > best_n:
            best, best_n = q, n
    return best if best_n > 0 else None

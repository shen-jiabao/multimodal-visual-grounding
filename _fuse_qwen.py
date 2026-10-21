# -*- coding: utf-8 -*-
"""三方择优融合：Qwen3-VL 直接框 + SoM 序数框 + 当前锁定提交。

输入
- result_locate_merge_som.json  当前锁定提交（融合基线）
- qwen3vl_all.json              Qwen3-VL 逐条直接推理（扁平 qid -> {bbox,n,raw}）
- qwen3vl_som_all.json          序数问句 SoM 选中结果（扁平 qid -> {core,n_cands,num,bbox}）
- _ordinal_cores_v3.json        序数问句集合（2014 条）

规则
- FROZEN 永不改动：护栏 4 样本 + 人工已确认的 SoM 修复 2 样本
- SOM_UNTRUSTED 永不采用 SoM：核心词退化（只剩序数/空间/泛义词，如
  "first closest tree" 抽成 "first"）或核心词实为参照物（009396_006/007）。
  这类条目即使枚举"成功"也可能选中无关类别，一律保留基线
- SOM_UNDERCOVER 永不采用 SoM：问句要第 k 个实例、而候选表只有 n_cands < k
  个 —— 第 k 个实例不在候选表里，模型任何合法编号都对应不到正确答案，必错。
  与方向无关（"从右数第 k 个"同样需要至少 k 个候选）。这是逻辑必然的防错，
  不是经验猜测
- 序数（所有策略一致）：SoM 成功即替换；与基线 IoU>=IOU_KEEP 视为一致、
  保留基线以减少无谓改动；SoM 失败则保留基线
- 非序数按策略：
    qwen   Qwen 为准——只要 Qwen 框合法就采纳，不问与基线是否一致
           （基线无特权地位：它同样是模型输出+合并链的产物，非 ground truth）
    safe   仅当基线明显错误（缺框 / 非法 / 面积>MAX_AREA）时用 Qwen 框替换
    shrink 基线为可疑宽合并框（面积>BIG_AREA）且 Qwen 框更小但仍占基线
           一定比例（>=SHRINK_KEEP）时替换——沿用「缩小=修正方向」的既定判断
    full   只要与 Qwen 不一致（IoU<IOU_KEEP）且 Qwen 框合理即替换
- 确定性序数选择策略（不采用模型给出的编号）：
    det     按 `expected_ordinal_num` 推算期望编号，直接取该候选；推算不可靠
            （候选未按 x 排序 / 方向不明 / k>候选数 / 序数是参照物）时回落常规
    det_rb  只把「模型答案与期望不符」的可疑条目回退基线；模型答案与期望一致
            的条目保留 SoM 结果（回退靶心是可疑条目，不是整层序数）
    det_full 序数分支同 det；非序数分支改用 full（两模型一致时保留基线）。
            依据：线上实测 full=0.7606 略优于 qwen=0.7590，说明「Qwen 与基线在
            同一物体上仅框紧度不同（IoU>=0.5）」时，保留基线略优。两个改进落在
            **不相交**的条目集（序数 2014 / 非序数 7541），故可叠加。
    det_full_qfb / det_full_qfb_wide
            在 det_full 之上，把「最终保留基线的序数」改用 Qwen 直接框：
            窄口径（_qfb）只接管回退点（枚举失败 / k>候选数 / 核心词不可信 /
            期望候选非法，实测 429 条）；宽口径（_qfb_wide）再并入 SOM_AGREE
            （SoM 与基线一致故按规则保留基线，实测 83 条）。两者都保留基线
            作为兜底：Qwen 无框或框非法时不替换
  动机：小样实测显示模型对该步主要是「对 k 做算术」而非读图上的编号——
  只澄清「1=最左」它重复原答案，给出算式才改口。既然方向与 k 都能从问句
  文本程序化提取，就不必让模型重推。
- 任何策略下，Qwen/SoM 均不可用时（无框、候选非法），保留基线兜底——
  提交必须 9555 条且无 error 条目，那时基线是唯一可用来源

输出
- result_qwen_{policy}.json / .zip
- _audit_qwen_{policy}.json  逐条改动明细（供人工复核）
- --viz N 另存 N 条等距抽样的新旧框对比图到 viz_out_qwen/

用法：
    python _fuse_qwen.py                          # 默认 qwen 策略
    python _fuse_qwen.py --policies det det_rb    # 确定性选择 / 回退基线
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import zipfile
from pathlib import Path

from PIL import Image, ImageDraw

R = Path(r"d:\Desktop\2026AiC")
DATA = R / "初赛数据集-基于大模型的多模态视觉理解与推理"
BASE_JSON = R / "result_locate_merge_som.json"
QD_JSON = R / "qwen3vl_all.json"
QS_JSON = R / "qwen3vl_som_all.json"
CORES_JSON = R / "_ordinal_cores_v3.json"
TEST_JSON = DATA / "test.json"
VIZ_DIR = R / "viz_out_qwen"

N_ENTRIES = 9555
# 护栏 4 样本（硬约束）+ 人工确认的 SoM 修复 2 样本（防回归）
FROZEN = {"000282_004", "000282_008", "002004_001", "002004_002",
          "000023_001", "000023_005"}
MAX_AREA = 0.35          # 超过即视为「合并多个目标」的宽框
BIG_AREA = 0.20          # 可疑宽框阈值（shrink 策略用）
MIN_AREA = 1e-5          # 退化框
SHRINK_KEEP = 0.25       # shrink 策略：新框面积不得小于基线的该比例
IOU_KEEP = 0.5           # IoU 达到该值视为两模型一致，保留基线
FIELDS = {"visible", "infrared", "depth", "query", "bbox"}
# 策略全集。渲染器 `_viz_qwen_vs_base` 直接引用本常量，避免两处各写一份而漂移。
POLICIES = ("qwen", "safe", "shrink", "full")
# 确定性序数选择策略（不采用模型给出的编号，改为按方向+序号程序化取候选）。
# 与 POLICIES 分开：渲染器的 `policies_for()` 只表达「模型选框被哪个策略采纳」，
# 而这两个策略根本不看模型的编号，语义不同，混进去会让渲染器的判据失真。
DET_POLICIES = ("det", "det_rb", "det_full", "det_full_qfb",
                "det_full_qfb_wide", "det_full_qfb_nosom",
                "det_full_qfb_nosom_qonly")
# 各确定性序数策略在【非序数】上采用的非序数策略。det/det_rb 沿用 qwen（Qwen 为准）；
# det_full 改用 full（IoU>=0.5 时保留基线）——线上实测 full 略优于 qwen。
# `qonly` 把非序数拉回 qwen：**非序数一律采纳 Qwen 框**（不再保留基线的紧框）。
# 用途：衡量"只依赖 Qwen 一个模型"的代价——当前最高分有 60% 的框来自基线链
# （非序数保留基线 5757 条），qonly 把这部分交还 Qwen，是"单模型化"的定量试验。
# 声明：仍非 100% 单模型——Qwen 无合法框的 227 条必须兜底基线，否则不满足
# "9555 条且无 error 条目"的硬约束。
DET_NONORD = {"det": "qwen", "det_rb": "qwen", "det_full": "full",
              "det_full_qfb": "full", "det_full_qfb_wide": "full",
              "det_full_qfb_nosom": "full",
              "det_full_qfb_nosom_qonly": "qwen"}
# 序数回退来源开关（Qwen FallBack，仅用于对照实验，不改变既有策略行为）：
# 默认情况下，当 SoM 与位置推算都不可用时（枚举失败 / k>候选数 / 核心词不可信 /
# 期望候选非法），序数条目一律退回基线。开启 qfb 后，改用 Qwen 的**直接框**作回退
# ——Qwen 直接框本就随全量推理落在本地，故这是零 GPU 成本的规则改动。
# 依据与风险见交付文档 §8：基线在序数类上有自己的空间排序逻辑，Qwen 直接框未必更优，
# 故这是一个**待线上验证的假设**，不是既成结论。
# 窄口径 qfb：只接管上述「回退点」（实测 429 条）。
# 宽口径 qfb_wide：在窄口径之上，连「SoM 与基线一致（IoU>=IOU_KEEP）故按规则保留
# 基线」的 SOM_AGREE 也改采纳 Qwen 直接框——语义为「凡最终保留基线的序数条目，一律
# 交给 Qwen 直接框」。实测该类仅 83 条，故两口径的候选差集只有 84 条。
QFB_POLICIES = ("det_full_qfb", "det_full_qfb_wide", "det_full_qfb_nosom",
                "det_full_qfb_nosom_qonly")
QFB_WIDE_POLICIES = ("det_full_qfb_wide", "det_full_qfb_nosom",
                     "det_full_qfb_nosom_qonly")
# 最激进的一档：连 `SOM`（模型自报编号）分支也改用 Qwen 直接框。开启后，序数条目的
# 框只可能来自两个来源——「位置推算」或「Qwen 直接框」，**模型自报编号不再参与**。
# 动机：模型自报编号存在系统性方向偏差（RTL 错误率 27.6%，为 LTR 的 2.6 倍；
# 且位置推算相对"回退基线"又净增 75 条），而这 222 条是规则层唯一从未做过
# 取舍实验的分支。它检验的是一个明确命题：**SoM 的模型编号分支是否还有价值**。
# 注意这与 `_wide` 不同：`_wide` 只覆盖 SoM 与基线**一致**的条目（83 条，实测无结论），
# 本档覆盖 SoM 与基线**不一致**、即真正由模型编号决定的条目（222 条）。
QFB_SOM_POLICIES = ("det_full_qfb_nosom", "det_full_qfb_nosom_qonly")

# 序数问句「核心词退化」判据：核心词若只由序数/空间/泛义/尺寸颜色等修饰词构成，
# 说明抽取失败（例：query "The first closest tree" 被抽成 "first"）。此时枚举
# 提示词变成 `Detect ALL instances of "first"`——危害不止于枚举失败（那会自动
# 回退基线），而是它可能勉强返回多个框并被选中一个完全无关的类别。
_ORD = ("first second third fourth fifth sixth seventh eighth ninth tenth "
        "1st 2nd 3rd 4th last next one ones").split()
_SPA = ("left right leftmost rightmost closest nearest farthest furthest "
        "front back behind top bottom upper lower middle center centre "
        "side end frontmost backmost").split()
_DET = "the a an and or of in on to from by with at as that which is are".split()
_GEN = ("object objects item items thing things element elements part parts "
        "entity entities stuff area areas region regions portion portions "
        "piece pieces").split()
_ADJ = ("large small big little hard soft thick thin long short tall wide "
        "narrow high low new old green blue red white black brown yellow grey "
        "gray orange pink purple gold golden silver dark bright").split()
# 注意："light" 被刻意排除在 _ADJ 之外——它在本题数据里是**目标名词**
# （"The first light from left to right at the third floor"），单字核心词
# "light" / "white light" 枚举"灯"再选第 N 个是正确用法，误判会白丢 9 条增益。
NON_HEAD_TOKENS = set(_ORD + _SPA + _DET + _GEN + _ADJ)

# 核心词抽成了「参照物」而非目标：如 009396_006/007 的目标是天花板上的白色
# 圆盘/烟感，而 "first and second rectangular air vents" 只是定位参照。
# 枚举参照物再选中一个 = 必错，故一律不采用 SoM。
SOM_REF_UNTRUSTED = {"009396_006", "009396_007"}

_UNTRUSTED_CACHE: set | None = None


def core_degenerate(core: str) -> bool:
    """核心词是否已退化为纯修饰词堆（不含任何实际目标名词）。"""
    toks = [t for t in re.split(r"[\s\-]+", core.lower()) if t]
    return not toks or all(t in NON_HEAD_TOKENS for t in toks)


def untrusted_ordinals() -> set:
    """SoM 不可信的序数问句集合（核心词退化 ∪ 参照物），结果缓存。"""
    global _UNTRUSTED_CACHE
    if _UNTRUSTED_CACHE is None:
        cores = json.loads(CORES_JSON.read_text(encoding="utf-8"))
        _UNTRUSTED_CACHE = {q for q, c in cores.items()
                            if core_degenerate(c)} | SOM_REF_UNTRUSTED
    return _UNTRUSTED_CACHE


def sha256(p: Path) -> str:
    """计算文件 SHA256（大写十六进制）。"""
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def iou(a, b) -> float:
    """两个归一化框的 IoU。"""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def area(b) -> float:
    """归一化框面积。"""
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def ordered(b) -> bool:
    """框是否有序且落在 [0,1]（不约束面积）。存量宽框也要满足这一条。"""
    if not b or len(b) != 4:
        return False
    x1, y1, x2, y2 = (float(v) for v in b)
    return 0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0


def valid_cand(b) -> bool:
    """候选框是否可用：有序 + 面积落在 [MIN_AREA, MAX_AREA]。

    注意与 `ordered` 的区别：锁定基线中本就存在 169 条面积>0.35 的宽合并框，
    它们是既成事实（也是本轮要修的目标），不能用新框的标准去否决存量。
    """
    return ordered(b) and MIN_AREA <= area(b) <= MAX_AREA


def norm(b) -> list:
    """量化到 6 位小数，避免浮点噪声进入提交。"""
    return [round(float(v), 6) for v in b]


# 序数词 -> 序号。仅取「目标序号」语义的词（last/next/one 不在此列，
# 它们不给出可比较的位置数）。
ORD_MAP = {
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
    "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10,
    "1st": 1, "2nd": 2, "3rd": 3, "4th": 4, "5th": 5,
    "6th": 6, "7th": 7, "8th": 8, "9th": 9, "10th": 10,
}
ORD_PAT = re.compile(r"\b(" + "|".join(ORD_MAP) + r")\b", re.I)


def ordinal_index(query: str) -> int | None:
    """抽取问句里的目标序号（first->1, second->2 …）；无则 None。

    局限：只取最先出现的序数词。"the light on the third floor" 这类把楼层
    当序数的问句会误抽成 3。误抽的后果仅是「多回退一次基线」（保守方向），
    且需同时满足 k > n_cands 才触发，故不额外特判。
    """
    if not query:
        return None
    m = ORD_PAT.search(query)
    return ORD_MAP[m.group(1).lower()] if m else None


# ---------------------------------------------------------------- 序数方向判定
# 以下四个判据是「确定性序数选择」的唯一实现，分析/渲染脚本一律从此处引用，
# 避免多处各写一份而漂移（本项目已因双份判据出过一次错）。
# 注意 RTL 先于 LTR 判定："from right to left" 不会被 LTR 的 "to left" 误吞。
DIR_RTL = re.compile(
    r"(right\s*to\s*left|rightmost|right-most|from\s+the\s+right|"
    r"from\s+right)", re.I)
DIR_LTR = re.compile(
    r"(left\s*to\s*right|leftmost|left-most|from\s+the\s+left|"
    r"from\s+left)", re.I)
# 垂直序数轴（复赛变体启用，见 `ordinal_axis` 的 axes 参数）。**默认不启用**：
# 初赛的归档提交件由本模块产出，无条件启用会让它们不再逐字节可复现。
# 两组无重叠（"from bottom to top" 不含 "from the top"），仍按 BTT 先判。
DIR_BTT = re.compile(
    r"(bottom\s*to\s*top|from\s+the\s+bottom|from\s+bottom|"
    r"bottommost|bottom-most|lowermost)", re.I)
DIR_TTB = re.compile(
    r"(top\s*to\s*bottom|from\s+the\s+top|from\s+top|"
    r"topmost|top-most|uppermost)", re.I)


# 深度序数轴（复赛变体启用，须在 axes 里含 'z'）。两组无重叠，先判 NFR 再判 FRN。
# 与 `DIR_BTT/TTB` 同样**默认不启用**：初赛归档件由本模块产出，无条件启用会让它们
# 不再逐字节可复现。
DIR_NFR = re.compile(
    r"(closest|nearest)\s+to\s+(the\s+)?camera|front\s*to\s*back|"
    r"from\s+(the\s+)?(camera|front)\s+to\s+the\s+(back|rear)|"
    r"\b(closest|nearest)\b", re.I)
DIR_FRN = re.compile(
    r"(farthest|furthest)\s+from\s+(the\s+)?camera|back\s*to\s*front|"
    r"\b(farthest|furthest|deepest)\b", re.I)
# 深度轴的护栏：两类问句里 "closest/nearest" 并非指远近，必须排除——
#   ① "to the left/right of … closest to the camera"：closest 属于**参照物**，
#      序数指的是"相对该参照物左侧的第一个"（实测 015100_003）
#   ② "nearest to the right side of the screen"：指**靠屏幕右缘**，是水平位置
#      （实测 015877_003）
#   ③ 同时出现 bottom/top 的（如 "at the bottom closest to the camera"）垂直与深度
#      两种读法并存，不猜（实测 000624_003）
DEPTH_GUARD = re.compile(r"to\s+the\s+(left|right)\b|\b(bottom|top)\b", re.I)


def direction(query: str) -> str:
    """问句指定的**水平**排序方向：'LTR' / 'RTL' / ''（未指定）。"""
    if not query:
        return ""
    if DIR_RTL.search(query):
        return "RTL"
    if DIR_LTR.search(query):
        return "LTR"
    return ""


def ordinal_axis(query: str, axes: tuple = ("x",)) -> tuple[str, str]:
    """问句指定的排序（轴, 朝向）；判不出返回 ('', '')。

    axes 限定允许的轴集合。默认只允许 'x'，故对既有调用方的行为与 `direction`
    完全一致；复赛变体传 ('x','y') 才启用垂直轴、传 ('x','y','z') 才再启用深度轴。
    优先级：水平 > 垂直 > 深度。同时出现（如 'the leftmost from the top'）时按优先级
    取靠前者——沿用旧版既有优先级，不新增歧义分支。
    深度轴还需过 `DEPTH_GUARD`（closest/nearest 不指远近的三类句式）。
    """
    d = direction(query)
    if d:
        return ("x", d)
    if "y" in axes:
        if DIR_BTT.search(query or ""):
            return ("y", "BTT")
        if DIR_TTB.search(query or ""):
            return ("y", "TTB")
    if "z" in axes and query and not DEPTH_GUARD.search(query):
        if DIR_NFR.search(query):
            return ("z", "NFR")
        if DIR_FRN.search(query):
            return ("z", "FRN")
    return ("", "")


def ordered_candidates(query: str, cands: list,
                       axes: tuple = ("x",), zkeys: list | None = None) -> list | None:
    """按问句指定的轴给出「1 号在最前」的候选序列；轴不可判定返回 None。

    水平轴沿用既有约定：`ENUM_PROMPT` 要求候选按 x 从左到右排列，故 LTR = 原序、
    RTL = 逆序；逆序取第 k 个与旧版 `cands[n-k]` 逐元素等价。
    垂直轴按 y 中心排序——这是**修正输入**而非放宽判据：序号由文本推算，不依赖
    模型给出的编号，只需一份沿「问句所指的轴」正确的候选序列。
    深度轴按 `zkeys`（每个候选框的深度代表值，与 cands 一一对应）排序，近→远即
    NFR 的前序。深度值需从 16 位深度图按框取中位数，属数据层计算，故由调用方传入；
    **zkeys 缺失时返回 None**（宁可不动，不猜顺序）。
    """
    ax, ori = ordinal_axis(query, axes)
    if not ax:
        return None
    if ax == "x":
        return list(cands) if ori == "LTR" else list(reversed(cands))
    if ax == "y":
        sy = sorted(cands, key=lambda b: (b[1] + b[3]) / 2)
        return sy if ori == "TTB" else list(reversed(sy))
    if zkeys is None or len(zkeys) != len(cands):
        return None
    order = sorted(range(len(cands)), key=lambda i: zkeys[i])
    seq = [cands[i] for i in order]
    return seq if ori == "NFR" else list(reversed(seq))


# 只识别「方位介词引导的参照物」，不含 "of the" 属格——后者多为
# "the trunk of the second tree" 这类「序数仍是目标标识」的结构，
# 剔除会误伤真错判（已在 000870_003 / 003614_003 / 009669_002 上验证）。
REF_SPATIAL = re.compile(
    r"(to\s+the\s+(right|left)\s+of|"
    r"(right|left)[-\s](rear|front)?\s*of\s+the|"
    r"behind\s+the|in\s+front\s+of\s+the|next\s+to\s+the|beside\s+the|"
    r"near\s+the|between\s+the|above\s+the|below\s+the|under\s+the|"
    r"on\s+top\s+of\s+the|adjacent\s+to\s+the)\s*$", re.I)
REF_WIN = 32


def ordinal_is_reference(query: str) -> bool:
    """序数词是否被方位介词引导而成为参照物（此时目标不是该序数实例）。"""
    if not query:
        return False
    for m in ORD_PAT.finditer(query):
        pre = query[max(0, m.start() - REF_WIN):m.start()]
        if REF_SPATIAL.search(pre):
            return True
    return False


CAND_X_TOL = 0.02     # 允许的 x 中心逆序容差（重复检出会导致微小逆序）
CAND_MIN_N = 2


def cands_left_to_right(cands: list) -> bool:
    """候选是否确实按 x 从左到右排列。

    这是「编号 -> 位置」映射可靠的前提：`ENUM_PROMPT` 要求候选 ordered from
    left to right，`SEL_PROMPT` 声明 1 = leftmost。实测 1641 条多候选记录中
    213 条严重逆序（最大回退 >0.02），对这些记录编号无法映射到位置。
    """
    if not cands or len(cands) < CAND_MIN_N:
        return False
    cx = [(b[0] + b[2]) / 2 for b in cands]
    return all(cx[i] <= cx[i + 1] + CAND_X_TOL for i in range(len(cx) - 1))


def expected_ordinal_num(query: str, cands: list,
                         n_cands: int | None = None,
                         axes: tuple = ("x",),
                         zkeys: list | None = None) -> int | None:
    """推算问句要求的候选序号（1 基，沿 `ordinal_axis` 判出的轴计）；不可靠返回 None。

    四个前提缺一不可，任一不满足即拒绝推算（宁可回退基线，不给错框）：
      1. 候选表存在；**水平轴**还要求它确实按 x 从左到右排列——垂直/深度轴无此要求，
         因为该轴的序号由我们自己按 y 或深度重排后取，与模型给出的编号无关
      2. 问句方向明确（`ordinal_axis` 能同时判出轴与朝向）
      3. 序号 k 可抽取，且 k <= 候选数
      4. 序数词不是方位介词引导的参照物

    返回值是 `ordered_candidates(query, cands, axes, zkeys)` 里的 1 基下标——**调用方
    必须按同一序列取值**（见 `det_ordinal`），否则各轴会取错元素。
    """
    if not query or not cands:
        return None
    n = len(cands)
    if n_cands is not None and n_cands != n:
        return None                      # 记录与候选表条目数不符，不可信
    ax, _ori = ordinal_axis(query, axes)
    if not ax:
        return None
    if ax == "x" and not cands_left_to_right(cands):
        return None
    if ax == "z" and (zkeys is None or len(zkeys) != n):
        return None                      # 无深度值就排不了序，宁可不动
    k = ordinal_index(query)
    if k is None or k > n:
        return None
    if ordinal_is_reference(query):
        return None
    return k


def det_ordinal(policy: str, qrec: dict, query: str, axes: tuple = ("x",),
                zkeys: list | None = None):
    """确定性序数选择：返回 (新框, 规则名)。

    `expected_ordinal_num` 判定可靠时：
        det     取「沿问句所指的轴、1 号在最前」那个序列的第 k 个（若其非法
                则回退基线）——水平轴下该序列就是原候选表或其逆序，与旧版
                `cands[k-1]` / `cands[n-k]` 逐元素等价
        det_rb  仅当模型答案与期望不符（= 可疑条目）时保留基线；模型答案与期望
                一致时返回 ORD_RB_AGREE 让调用方回落常规分支，即保留 SoM 结果
                ——回退的靶心是可疑条目，不是整层序数（把一致的 1027 条也退回
                基线等于丢掉整个序数层）
    判定不可靠时返回 ORD_UNDET，调用方回落到常规（模型编号）分支——即只在
    「有把握」的条目上覆盖模型，其余不动。

    axes 默认 ("x",)，与旧版行为完全一致；传 ("x","y") 启用垂直序数轴，
    传 ("x","y","z") 再启用深度序数轴（此时必须同时给 `zkeys`，否则判为不可靠）。

    动机：小样实测发现模型对该步主要是「对 k 做算术」而非读图上的编号——只澄清
    「1=leftmost」它重复原答案，给出算式才改口。既然如此，方向与序号都能从问句
    文本程序化提取，就不必再让模型重推。
    """
    cands = qrec.get("bbox_all")
    exp = expected_ordinal_num(query, cands, qrec.get("n_cands"), axes, zkeys)
    if exp is None:
        return None, "ORD_UNDET"
    if policy == "det_rb":
        if qrec.get("num") == exp:
            return None, "ORD_RB_AGREE"
        return None, "ORD_ROLLBACK"
    seq = ordered_candidates(query, cands, axes, zkeys)
    if seq is None or exp > len(seq):
        return None, "ORD_UNDET"
    box = seq[exp - 1]
    if not valid_cand(box):
        return None, "ORD_DET_INVALID"
    return norm(box), "ORD_DET"


# 这些规则表示「不做确定性主张，回落常规（模型编号）分支」
FALLTHROUGH_RULES = ("ORD_UNDET", "ORD_RB_AGREE")


def decide(policy: str, base, qd, qs, is_ordinal: bool,
           query: str | None = None, n_cands: int | None = None,
           qfb: bool = False, qfb_wide: bool = False, qfb_som: bool = False):
    """返回 (新框, 规则名)；不替换时新框为 None。

    query / n_cands 仅用于序数分支的「候选不足」判定：候选表按从左到右共
    n_cands 个，若问句要第 k 个实例而 k > n_cands，第 k 个实例**不在候选表
    里**，模型任何合法编号都对应不到正确答案 → 必错，故不采纳 SoM。该推理
    与方向无关（"从右数第 k 个" 同样需要至少 k 个候选）。

    qfb=True 时，序数回退改取 Qwen 直接框而非基线（见 QFB_POLICIES 注释）。
    qfb_wide=True 时，连同 SOM_AGREE（SoM 与基线一致故保留基线）也改取
    Qwen 直接框；Qwen 框非法时仍保留基线。
    qfb_som=True 时，连 SOM（模型自报编号）分支也改取 Qwen 直接框，等价于
    序数只承认「位置推算」与「Qwen 直接框」两个来源。
    """
    if is_ordinal and qs is not None and valid_cand(qs):
        k = ordinal_index(query) if query else None
        if k is not None and n_cands is not None and k > n_cands:
            if qfb and valid_cand(qd):
                return norm(qd), "ORD_QFB_UNDER"
            return None, "SOM_UNDERCOVER"
        if base is None or iou(base, qs) < IOU_KEEP:
            if qfb_som and valid_cand(qd):
                return norm(qd), "ORD_QFB_SOM"
            return norm(qs), "SOM"
        if qfb_wide and valid_cand(qd):
            return norm(qd), "ORD_QFB_AGREE"
        return None, "SOM_AGREE"
    if is_ordinal:
        # 序数回退：默认退回基线；qfb 开关下改用 Qwen 直接框（若合法）
        if qfb and valid_cand(qd):
            return norm(qd), "ORD_QFB"
        return None, "SOM_FAIL_KEEP"
    # 非序数：Qwen 直接框
    if not valid_cand(qd):
        return None, "QD_INVALID_KEEP"
    if base is None:
        return norm(qd), "QD_FILL"
    if policy == "qwen":
        # Qwen 为准：不因与基线一致与否改变决定（基线无特权地位）
        return norm(qd), "QWEN"
    if policy == "safe":
        if area(base) > MAX_AREA:
            return norm(qd), "QD_BIG"
        return None, "KEEP"
    if policy == "shrink":
        if area(base) > MAX_AREA:
            return norm(qd), "QD_BIG"
        if area(base) > BIG_AREA and area(qd) < area(base) \
                and area(qd) >= SHRINK_KEEP * area(base):
            return norm(qd), "QD_SHRINK"
        return None, "KEEP"
    if policy == "full":
        if iou(base, qd) < IOU_KEEP:
            return norm(qd), "QD_DIFF"
        return None, "QD_AGREE"
    raise ValueError(f"未知策略 {policy}")


def resolve_img(rel: str) -> Path | None:
    """把提交里的 visible 相对路径解析为本地图片路径。

    提交中的路径形如 `Images/visible/000002.png`，相对数据集根目录；
    个别条目可能只给文件名，故用 basename 兜底。
    """
    if not rel:
        return None
    p = DATA / rel
    if p.exists():
        return p
    p = DATA / "Images" / "visible" / Path(rel).name
    return p if p.exists() else None


def render(qid: str, entry: dict, old, new, out_dir: Path) -> bool:
    """在原图上画 绿=基线 / 红=新框，保存对比图。"""
    p = resolve_img(entry.get("visible"))
    if p is None:
        return False
    try:
        im = Image.open(p).convert("RGB")
    except Exception as ex:
        print(f"  [viz] 读图失败 {qid}: {ex!r}")
        return False
    d = ImageDraw.Draw(im)
    w, h = im.size
    lw = max(3, int(min(w, h) * 0.006))
    if old:
        d.rectangle([old[0] * w, old[1] * h, old[2] * w, old[3] * h],
                    outline=(0, 255, 0), width=lw)
    if new:
        d.rectangle([new[0] * w, new[1] * h, new[2] * w, new[3] * h],
                    outline=(255, 0, 0), width=lw)
    out_dir.mkdir(parents=True, exist_ok=True)
    im.save(out_dir / f"{qid}.png")
    return True


def fuse(policy: str, base_all: dict, qd_all: dict, qs_all: dict,
         ordinal: set) -> tuple[dict, list, dict]:
    """按策略生成新提交、改动明细与规则计数。"""
    data = json.loads(json.dumps(base_all))  # 深拷贝
    changes = []
    stats: dict[str, int] = {}
    untrusted = untrusted_ordinals()
    qfb = policy in QFB_POLICIES
    qfb_wide = policy in QFB_WIDE_POLICIES
    qfb_som = policy in QFB_SOM_POLICIES
    for qid, entry in base_all.items():
        base = entry.get("bbox")
        qd = qd_all.get(qid, {}).get("bbox")
        qrec = qs_all.get(qid, {})
        qs = qrec.get("bbox")
        is_ord = qid in ordinal
        new, rule, ft = None, None, False
        if qid in FROZEN:
            stats["FROZEN"] = stats.get("FROZEN", 0) + 1
            continue
        if qid in untrusted:
            # 核心词退化/参照物 → 枚举基座无意义，故不做 SoM 主张。默认保留基线；
            # qfb 下改用 Qwen 直接框——这类条目的缺陷出在「枚举基座」，
            # 与直接定位无关，故直接框是合理替代。
            if qfb and valid_cand(qd):
                new, rule = norm(qd), "ORD_QFB_UNTRUSTED"
            else:
                stats["SOM_UNTRUSTED"] = stats.get("SOM_UNTRUSTED", 0) + 1
                continue
        elif is_ord and policy in DET_POLICIES:
            # 先尝试确定性选择；不可靠（或 det_rb 遇模型答案本就一致）时回落到
            # 常规分支，即只在「有把握且模型可疑」的条目上做主张
            new, rule = det_ordinal(policy, qrec, entry.get("query"))
            # qfb 下「期望候选非法」也纳入回落，从而改用 Qwen 直接框作回退
            ft = rule in FALLTHROUGH_RULES or (qfb and rule == "ORD_DET_INVALID")
            if ft:
                stats[rule] = stats.get(rule, 0) + 1
        if rule is None or ft:
            # 非序数分支按策略映射：det/det_rb → qwen；det_full 及其 qfb 变体 → full
            pol = DET_NONORD.get(policy, policy)
            new, rule = decide(pol, base, qd, qs, is_ord,
                               query=entry.get("query"),
                               n_cands=qrec.get("n_cands"), qfb=qfb,
                               qfb_wide=qfb_wide, qfb_som=qfb_som)
        stats[rule] = stats.get(rule, 0) + 1
        if new is None:
            continue
        if base is not None and norm(base) == new:
            # 与基线逐位相同 → 不是真实改动。不记入明细，否则 qwen 策略会把
            # 全部 7541 条非序数都列成"改动"，审计表失去核对价值。
            stats["SAME_AS_BASE"] = stats.get("SAME_AS_BASE", 0) + 1
            continue
        data[qid]["bbox"] = new
        changes.append({
            "qid": qid, "rule": rule, "ordinal": is_ord,
            "core": qs_all.get(qid, {}).get("core"),
            "old": base, "new": new,
            "iou": round(iou(base, new), 4) if base else None,
            "old_area": round(area(base), 5) if base else None,
            "new_area": round(area(new), 5),
        })
    return data, changes, stats


def validate(data: dict, test: dict, base_all: dict) -> tuple[int, int]:
    """硬校验并返回 (基线的存量宽框数, 融合后仍超限的宽框数)。

    硬失败：键集不一致 / 条数不符 / 字段集不符 / 框无序或越界 / 护栏被改。
    面积>0.35 只统计不判失败——锁定基线中本就有 169 条，属待修项而非
    新引入的错误。
    """
    assert set(data) == set(test), "键集与 test.json 不一致"
    assert len(data) == N_ENTRIES, f"条数 {len(data)} != {N_ENTRIES}"
    bad = [k for k, v in data.items()
           if set(v) != FIELDS or not ordered(v["bbox"])]
    assert not bad, f"非法条目 {bad[:5]}"
    for g in FROZEN:
        assert data[g]["bbox"] == base_all[g]["bbox"], f"护栏被改 {g}"
    nb = sum(1 for v in base_all.values() if area(v["bbox"]) > MAX_AREA)
    nn = sum(1 for v in data.values() if area(v["bbox"]) > MAX_AREA)
    return nb, nn


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policies", nargs="+", default=["qwen"],
                    help="默认 qwen（Qwen 为准 + SoM 序数）；"
                         "det/det_rb/det_full 为确定性序数选择各变体；"
                         "det_full_qfb（窄口径）/ det_full_qfb_wide（宽口径）/ "
                         "det_full_qfb_nosom（连模型自报编号也不用）"
                         "为序数改用 Qwen 直接框的对照实验；"
                         "safe/shrink/full 为对照策略，按需显式指定")
    ap.add_argument("--viz", type=int, default=0,
                    help="每个策略输出多少张等距抽样对比图")
    args = ap.parse_args()

    base_all = json.loads(BASE_JSON.read_text(encoding="utf-8"))
    qd_all = json.loads(QD_JSON.read_text(encoding="utf-8"))
    qs_all = json.loads(QS_JSON.read_text(encoding="utf-8"))
    ordinal = set(json.loads(CORES_JSON.read_text(encoding="utf-8")))
    test = json.loads(TEST_JSON.read_text(encoding="utf-8"))
    print(f"基线 {len(base_all)} | Qwen直接 {len(qd_all)} | SoM {len(qs_all)} "
          f"| 序数 {len(ordinal)}")
    assert len(qd_all) == N_ENTRIES, f"Qwen 直接框不完整: {len(qd_all)}"
    assert len(qs_all) == len(ordinal), \
        f"SoM 不完整: {len(qs_all)} != {len(ordinal)}"

    for policy in args.policies:
        data, changes, stats = fuse(policy, base_all, qd_all, qs_all, ordinal)
        n_big_base, n_big_new = validate(data, test, base_all)

        out_json = R / f"result_qwen_{policy}.json"
        out_zip = R / f"result_qwen_{policy}.zip"
        payload = json.dumps(data, ensure_ascii=False)
        out_json.write_text(payload, encoding="utf-8")
        with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
            # 固定条目时间戳：zipfile 默认写入「当前时间」，会让 ZIP 逐字节不可复现
            # （同样输入两次运行得到不同 ZIP SHA，容易被误判为内容不一致）。
            # 固定为 ZIP 纪元起点后，ZIP 与 JSON 均可逐字节复现。
            info = zipfile.ZipInfo(out_json.name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            zf.writestr(info, payload.encode("utf-8"))
        (R / f"_audit_qwen_{policy}.json").write_text(
            json.dumps(changes, ensure_ascii=False, indent=1),
            encoding="utf-8")

        print(f"\n== 策略 {policy} ==")
        for k in sorted(stats):
            print(f"  {k:16s} {stats[k]}")
        print(f"  改动合计          {len(changes)}")
        shr = [c for c in changes if c["old"] and c["new_area"] < c["old_area"]]
        print(f"  其中缩小 {len(shr)} / 放大 {len(changes) - len(shr)}")
        print(f"  宽合并框(>0.35) {n_big_base} → {n_big_new}")
        print(f"  JSON SHA256 {sha256(out_json)}")
        print(f"  ZIP  SHA256 {sha256(out_zip)}")

        if args.viz:
            step = max(1, len(changes) // args.viz)
            picked = changes[::step][:args.viz]
            n_ok = 0
            for c in picked:
                if render(c["qid"], base_all[c["qid"]], c["old"], c["new"],
                          VIZ_DIR / policy):
                    n_ok += 1
            print(f"  对比图 {n_ok}/{len(picked)} → {VIZ_DIR / policy}")
            (VIZ_DIR / policy / "_list.json").write_text(
                json.dumps(picked, ensure_ascii=False, indent=1),
                encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

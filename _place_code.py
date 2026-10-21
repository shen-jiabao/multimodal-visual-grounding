# -*- coding: utf-8 -*-
"""把核心代码分别插到对应方法的描述之后（§3.3.2 五个方法 + §3.5.2 共识投票）。

- §3.3.2：拆为 5 个「方法名 + 说明 + 对应代码」块，代码紧贴其说明
- §3.5：把共识投票的实现从末尾 §3.5.6 移到方法描述 §3.5.2 之后
代码按行号从 code27b/ 源文件摘取，保持与随包源码逐字一致。
"""
import re
from pathlib import Path

MD = Path(r"d:\Desktop\2026AiC\技术文档.md")
CODE = Path(r"d:\Desktop\2026AiC\复赛提交材料文件夹"
            r"\AIC-2026-14375345-熬夜炼丹师-复赛-沈嘉宝"
            r"\AIC-2026-14375345-代码与数据\code27b")


def grab(rel: str, first: int, last: int) -> str:
    lines = (CODE / rel).read_text(encoding="utf-8").splitlines()
    return "\n".join(lines[first - 1:last])


# 方法名 -> (代码注释头, 代码正文)
CODES = {
    "候选匹配链": ("# src/inference/matcher.py —— 三级匹配链 + 按约束排序取第 k 个",
              grab("src/inference/matcher.py", 147, 188)),
    "提示词层": ("# src/models/qwen3_vl_parse.py —— 定位提示词（规则 1：序数扫描；规则 2：参照系）",
             grab("src/models/qwen3_vl_parse.py", 48, 66)),
    "几何层": ("# src/inference/matcher.py —— 方位与序数约束解析（与"
             "\"排序取第 k 个\"同口径）",
            grab("src/inference/matcher.py", 75, 144)),
    "可选增强": ("# src/inference/detect_select.py —— 确定性选择：按轴排序取第 k 个（可选增强）",
             grab("src/inference/detect_select.py", 313, 337)),
    "跨查询一致性": ("# src/inference/chains.py —— 序数链的构成与「排序式重指派」\n"
               "# 另有 pava_non_decreasing()（PAVA 保序回归，供 mode=\"project\" 使用）与\n"
               "# classify_chain()（链状态审计）此处从略。",
               grab("src/inference/chains.py", 61, 79) + "\n\n"
               + grab("src/inference/chains.py", 158, 234)),
}

text = MD.read_text(encoding="utf-8")

# ---------- 1) 重建 §3.3.2 ----------
head, rest = text.split("#### 3.3.2 候选匹配与序数层", 1)
body, tail = rest.split("#### 3.3.3 相对位置与参照物语义", 1)

intro, bullets_part = body.split("- **候选匹配链**", 1)
bullets_part = "- **候选匹配链**" + bullets_part
closing = "以上均为" + bullets_part.split("以上均为", 1)[1]
closing = closing.split("\n\n", 1)[0].strip()

blocks = [intro.strip()]
for line in bullets_part.splitlines():
    m = re.match(r"- \*\*(.+?)\*\*(.*)", line)
    if not m:
        continue
    name, rem = m.group(1), m.group(2)
    suffix, desc = rem.split("：", 1)
    note, code = CODES[name]
    blocks.append(f"**{name}{suffix}**\n\n{desc.strip()}\n\n"
                  f"```python\n{note}\n{code}\n```")
blocks.append(closing)

new_332 = "\n\n".join(blocks) + "\n\n"
text = head + "#### 3.3.2 候选匹配与序数层\n\n" + new_332 \
    + "#### 3.3.3 相对位置与参照物语义" + tail

# ---------- 2) 移除 §3.5.6，把代码移到 §3.5.2 之后 ----------
head, rest = text.split("#### 3.5.6 核心代码实现", 1)
_, tail = rest.split("### 3.6 提交件校验与结果", 1)
text = head + "### 3.6 提交件校验与结果" + tail

iou = grab("scripts/consensus_vote.py", 54, 64)
core = grab("scripts/consensus_vote.py", 78, 120)
consensus = (
    "**对应实现（`scripts/consensus_vote.py` 核心片段）**\n\n"
    "```python\n"
    "# scripts/consensus_vote.py —— 核心片段：IoU、投票判决、确定性打包\n"
    f"{iou}\n\n"
    "# …（略）_bboxes()：读预测 JSON，取 qid 到 bbox 的映射\n"
    "# …（略）路径常量与命令行参数解析，见文件头\n\n"
    f"{core}\n```\n\n")
anchor = "#### 3.5.3 判据设计说明"
assert text.count(anchor) == 1
text = text.replace(anchor, consensus + anchor, 1)

MD.write_text(text, encoding="utf-8")
print("已重构 §3.3.2（5 个方法各自配码）并把共识投票代码移到 §3.5.2 之后")

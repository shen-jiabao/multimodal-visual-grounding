# -*- coding: utf-8 -*-
"""按「只保留关键代码」调整技术文档：

1. §3.3.2 追加 chains.py 的关键函数（链构成 + 排序式重指派）
2. §3.5.6 由 consensus_vote.py 全文改为核心片段（IoU、投票判决、确定性打包）
代码仍按行号直接从 code27b/ 源文件摘取。
"""
from pathlib import Path

MD = Path(r"d:\Desktop\2026AiC\技术文档.md")
CODE = Path(r"d:\Desktop\2026AiC\复赛提交材料文件夹"
            r"\AIC-2026-14375345-熬夜炼丹师-复赛-沈嘉宝"
            r"\AIC-2026-14375345-代码与数据\code27b")


def grab(rel: str, first: int, last: int) -> str:
    lines = (CODE / rel).read_text(encoding="utf-8").splitlines()
    return "\n".join(lines[first - 1:last])


# ---------- 1) chains.py 关键函数 ----------
chain_of = grab("src/inference/chains.py", 61, 79)
enforce = grab("src/inference/chains.py", 158, 234)
CHAINS = f"""**跨查询一致性的实现（关键函数）**

```python
# src/inference/chains.py —— 序数链的构成与「排序式重指派」
# 另有 pava_non_decreasing()（PAVA 保序回归，供 mode="project" 使用）与
# classify_chain()（链状态审计）此处从略。
{chain_of}

{enforce}
```

"""

# ---------- 2) consensus_vote.py 精简为核心片段 ----------
iou = grab("scripts/consensus_vote.py", 54, 64)
core = grab("scripts/consensus_vote.py", 78, 120)
CONSENSUS = f"""#### 3.5.6 核心代码实现

以下摘自 `scripts/consensus_vote.py`，即 §3.5.1–§3.5.3 所述规则的实现核心
（完整文件见 `code27b/scripts/`，路径常量、`_bboxes()` 取框辅助与哈希打印从略）。

```python
# scripts/consensus_vote.py —— 核心片段：IoU、投票判决、确定性打包
{iou}

# …（略）_bboxes()：读预测 JSON，取 qid 到 bbox 的映射
# …（略）路径常量与命令行参数解析，见文件头

{core}
```

"""

text = MD.read_text(encoding="utf-8")

assert text.count("#### 3.3.3 相对位置与参照物语义") == 1
text = text.replace("#### 3.3.3 相对位置与参照物语义",
                    CHAINS + "#### 3.3.3 相对位置与参照物语义", 1)

head, rest = text.split("#### 3.5.6 核心代码实现", 1)
_, tail = rest.split("### 3.6 提交件校验与结果", 1)
text = head + CONSENSUS + "### 3.6 提交件校验与结果" + tail

MD.write_text(text, encoding="utf-8")
print("chains.py 关键函数", len(chain_of.splitlines()) + len(enforce.splitlines()),
      "行；consensus 片段", len(iou.splitlines()) + len(core.splitlines()), "行")

# -*- coding: utf-8 -*-
"""把核心方法代码嵌入 技术文档.md 的对应小节。

- §3.3.2 之后：定位提示词（序数/参照系规则）、方位与序数约束解析、确定性选择
- §3.5 之后：consensus_vote.py 全文（新增 §3.5.6 核心代码实现）
代码直接从 code27b/ 源文件按行号摘取，保证与技术文档所述一致。
"""
from pathlib import Path

MD = Path(r"d:\Desktop\2026AiC\技术文档.md")
CODE = Path(r"d:\Desktop\2026AiC\复赛提交材料文件夹"
            r"\AIC-2026-14375345-熬夜炼丹师-复赛-沈嘉宝"
            r"\AIC-2026-14375345-代码与数据\code27b")


def grab(rel: str, first: int, last: int) -> str:
    """按 1-based 行号区间取源码（含首尾）。"""
    lines = (CODE / rel).read_text(encoding="utf-8").splitlines()
    return "\n".join(lines[first - 1:last])


S1 = grab("src/models/qwen3_vl_parse.py", 48, 66)
S2 = grab("src/inference/matcher.py", 75, 144)
S3 = grab("src/inference/detect_select.py", 313, 337)
S4 = grab("scripts/consensus_vote.py", 3, 130)

BLOCK_A = f"""**核心实现代码**

```python
# src/models/qwen3_vl_parse.py —— 定位提示词（规则 1：序数扫描；规则 2：参照系）
{S1}
```

```python
# src/inference/matcher.py —— 方位与序数约束解析（与"排序取第 k 个"同口径）
{S2}
```

```python
# src/inference/detect_select.py —— 确定性选择：按轴排序取第 k 个（可选增强）
{S3}
```

"""

BLOCK_B = f"""#### 3.5.6 核心代码实现

以下为 `scripts/consensus_vote.py` 全文，即 §3.5.1–§3.5.3 所述规则的完整实现：
输入为 `data_outputs/` 下的主件、参照件与投票池，输出 `result.zip`；无训练、无随机性。

```python
{S4}
```

"""

text = MD.read_text(encoding="utf-8")
for anchor, block in (("#### 3.3.3 相对位置与参照物语义", BLOCK_A),
                      ("### 3.6 提交件校验与结果", BLOCK_B)):
    assert text.count(anchor) == 1, f"锚点不唯一: {anchor}"
    text = text.replace(anchor, block + anchor)

MD.write_text(text, encoding="utf-8")
print(f"已插入：§3.3.2 后 {len(S1.splitlines()) + len(S2.splitlines()) + len(S3.splitlines())} 行代码，"
      f"§3.5 后 {len(S4.splitlines())} 行代码")

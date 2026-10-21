# -*- coding: utf-8 -*-
"""扫描 test.json 中的 query：统计含 CJK（中文）字符的条目并抽样展示。"""
import json
import re
from pathlib import Path

TEST = Path(r"d:\Desktop\2026AiC\初赛数据集-基于大模型的多模态视觉理解与推理\test.json")
data = json.loads(TEST.read_text(encoding="utf-8"))
CJK = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf]")

hits = [(k, v["query"]) for k, v in data.items() if CJK.search(v["query"])]
pure = [(k, q) for k, q in hits if len(CJK.findall(q)) >= 4 and
        len(re.findall(r"[A-Za-z]", q)) < 4]
print(f"总 query: {len(data)}")
print(f"含中文(CJK): {len(hits)} ({100*len(hits)/len(data):.2f}%)")
print(f"其中基本为纯中文: {len(pure)}")
print("\n=== 纯中文示例(前 12) ===")
for k, q in pure[:12]:
    print(f"  {k}: {q[:70]}")
print("\n=== 中英混合示例(前 8) ===")
for k, q in hits:
    if (k, q) not in pure:
        print(f"  {k}: {q[:80]}")
        if len([1 for kk, qq in hits if (kk, qq) not in pure][:8]) == 8:
            break
# 按场景分散统计
scenes = {k.split("_")[0] for k, _ in hits}
print(f"\n涉及场景数: {len(scenes)}")

# -*- coding: utf-8 -*-
"""打印 6 条抽检样本的候选框/CLIP 分数/选中结果。"""
import json
from pathlib import Path

R = Path(r"d:\Desktop\2026AiC")
pilot = json.loads((R / "fusion_pilot_235.json").read_text(encoding="utf-8"))
det = json.loads((R / "fusion_detail_local.json").read_text(encoding="utf-8"))
test = json.loads((R / "初赛数据集-基于大模型的多模态视觉理解与推理"
                   / "test.json").read_text(encoding="utf-8"))


def area(b):
    return (b[2] - b[0]) * (b[3] - b[1])


QIDS = ["016263_003", "003908_003", "008248_001",
        "015962_002", "008304_004", "008276_004"]
for q in QIDS:
    r = pilot[q]
    print("=" * 92)
    print(f"{q} | {test[q]['query'][:92]}")
    print(f"  chosen={r['source']}  bbox={[round(v,3) for v in r['bbox']]}  "
          f"area={area(r['bbox']):.3f}")
    if "sim" in r:
        print(f"  CLIP sims: {r['sim']}")
    for k, b in det[q]["boxes"].items():
        print(f"    {k:22s} area={area(b):.3f}  {[round(v,3) for v in b]}")

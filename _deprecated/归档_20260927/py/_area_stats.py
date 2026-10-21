# -*- coding: utf-8 -*-
"""统计新旧结果中「近整图大框」数量，评估退化风险。"""
import json
from pathlib import Path

R = Path(r"d:\Desktop\2026AiC")
new = json.loads((R / "result_gd_3modal_v2.json").read_text(encoding="utf-8"))
old = json.loads((R / "result_gd_3modal.json").read_text(encoding="utf-8"))


def area(b):
    return (b[2] - b[0]) * (b[3] - b[1])


for name, d in (("云端新", new), ("本地旧", old)):
    big = [k for k, v in d.items() if area(v["bbox"]) > 0.35]
    huge = [k for k, v in d.items() if area(v["bbox"]) > 0.60]
    tiny = [k for k, v in d.items() if area(v["bbox"]) < 0.002]
    print(f"{name}: 面积>0.35 的 {len(big)} 条; >0.60 的 {len(huge)} 条; <0.002 的 {len(tiny)} 条")

big_new = {k for k, v in new.items() if area(v["bbox"]) > 0.35}
big_old = {k for k, v in old.items() if area(v["bbox"]) > 0.35}
print(f"\n新增的大框(旧不超新超): {len(big_new - big_old)} 条")
print("示例（新的大框）:")
for k in sorted(big_new - big_old)[:8]:
    print(f"  {k}: {[round(float(x),3) for x in new[k]['bbox']]}  "
          f"q={new[k]['query'][:70]}")
# 旧的兜底框计数
fb = [k for k, v in old.items() if v["bbox"] == [0.25, 0.25, 0.75, 0.75]]
print(f"\n旧结果中的兜底框(0.25,0.25,0.75,0.75): {len(fb)} 条")

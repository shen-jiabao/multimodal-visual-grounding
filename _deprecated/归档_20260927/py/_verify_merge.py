# -*- coding: utf-8 -*-
"""最终融合输出整体校验。"""
import hashlib
import json

R = r"d:\Desktop\2026AiC"
final = json.load(open(R + r"\result_locate_merge.json", encoding="utf-8"))
base = json.load(open(R + r"\result_gd_3modal_fullseq.json", encoding="utf-8"))
test = json.load(open(R + r"\初赛数据集-基于大模型的多模态视觉理解与推理\test.json",
                      encoding="utf-8"))

print("条目:", len(final), "== test:", len(test))
print("键集一致:", set(final) == set(test))
FIELDS = {"visible", "infrared", "depth", "query", "bbox"}
bad_f = [k for k, v in final.items() if set(v) != FIELDS]
bad_b = [k for k, v in final.items()
         if not (0 <= v["bbox"][0] < v["bbox"][2] <= 1
                 and 0 <= v["bbox"][1] < v["bbox"][3] <= 1)]
print("字段异常:", len(bad_f), " 非法bbox:", len(bad_b))


def area(b):
    return (b[2] - b[0]) * (b[3] - b[1])


big_base = sum(1 for v in base.values() if area(v["bbox"]) > 0.35)
big_final = sum(1 for v in final.values() if area(v["bbox"]) > 0.35)
print(f"大框(>0.35): 基线 {big_base} -> 融合 {big_final}")
diff = sum(1 for k in final if final[k]["bbox"] != base[k]["bbox"])
print(f"与基线不同: {diff} ({100*diff/len(final):.1f}%)")
print("SHA256:",
      hashlib.sha256(open(R + r"\result_locate_merge.json", "rb").read()
                     ).hexdigest().upper())
for k in ["000023_002", "001038_003", "016774_001"]:
    print(f"  {k}: 融合={final[k]['bbox']} (基线={base[k]['bbox']})")
# -*- coding: utf-8 -*-
"""统计 REGION_RE 命中样本在融合中的采用情况。"""
import json
import re

R = r"d:\Desktop\2026AiC"
final = json.load(open(R + r"\result_locate_merge.json", encoding="utf-8"))
base = json.load(open(R + r"\result_gd_3modal_fullseq.json", encoding="utf-8"))
q = json.load(open(R + r"\初赛数据集-基于大模型的多模态视觉理解与推理\queries\queries.json",
                   encoding="utf-8"))

REGION_RE = re.compile(
    r"\b(lake|water|pond|river|sea|ocean|sky|cloud|grass|lawn|ground|floor|"
    r"road|street|sidewalk|path|field|sand|snow|wall|fence|barrier|railing|"
    r"bamboo|tree|door|area|region|scenery|background|foreground|landscape)\b",
    re.I)
reg = [k for k in q if REGION_RE.search(q[k]["query"])]
chg = [k for k in reg if final[k]["bbox"] != base[k]["bbox"]]
print("REGION_RE 命中数:", len(reg))
print("区域类中使用Locate(changed):", len(chg))
# 其中 Locate 框面积 < 基线 0.5x（缩小→错框嫌疑）的样本
def area(b):
    return (b[2]-b[0])*(b[3]-b[1])
shrink = [k for k in chg
          if k in json.load(open(R + r"\locate_bbox_all.json", encoding="utf-8"))
          and json.load(open(R + r"\locate_bbox_all.json", encoding="utf-8"))[k].get("bbox")
          and area(json.load(open(R + r"\locate_bbox_all.json",
                                  encoding="utf-8"))[k]["bbox"])
          < 0.5 * area(base[k]["bbox"])]
print("其中 Locate 显著缩小(<0.5x 基线, 错框嫌疑):", len(shrink))
for k in shrink[:15]:
    lb = json.load(open(R + r"\locate_bbox_all.json", encoding="utf-8"))[k]["bbox"]
    print(f"  {k} | {q[k]['query'][:55]} | loc_area={area(lb):.3f} base_area={area(base[k]['bbox']):.3f}")
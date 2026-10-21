# -*- coding: utf-8 -*-
"""复核 4 护栏样本零改动。"""
import json

R = r"d:\Desktop\2026AiC"
f = json.load(open(R + r"\result_locate_merge.json", encoding="utf-8"))
b = json.load(open(R + r"\result_gd_3modal_fullseq.json", encoding="utf-8"))
for k in ["000282_004", "000282_008", "002004_001", "002004_002"]:
    same = f[k]["bbox"] == b[k]["bbox"]
    print(f"  {k}: 相同={same}")
print("ALL GUARD OK" if all(f[k]["bbox"] == b[k]["bbox"]
                            for k in ["000282_004", "000282_008",
                                      "002004_001", "002004_002"])
      else "GUARD VIOLATED")
# -*- coding: utf-8 -*-
"""校验全量结果：条目数、error、bbox 合法性、与本地部分结果一致性。"""
import json
from pathlib import Path

NEW = Path(r"d:\Desktop\2026AiC\result_gd_3modal_v2.json")
OLD = Path(r"d:\Desktop\2026AiC\result_gd_3modal_local_partial.json")
TEST = Path(r"d:\Desktop\2026AiC\初赛数据集-基于大模型的多模态视觉理解与推理\test.json")

new = json.loads(NEW.read_text(encoding="utf-8"))
print(f"结果条目数: {len(new)}")
print(f"文件大小: {NEW.stat().st_size/1024:.1f} KB")

err = [k for k, v in new.items() if "error" in v]
print(f"含 error 字段: {len(err)}  {err[:5]}")

bad_order = bad_range = bad_nan = 0
for k, v in new.items():
    b = v.get("bbox")
    if not b or len(b) != 4:
        bad_range += 1
        continue
    x1, y1, x2, y2 = b
    if any(x != x for x in b):
        bad_nan += 1
    if not (0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0):
        bad_order += 1
print(f"bbox 非法(越界/顺序): {bad_order}, NaN: {bad_nan}, 缺字段: {bad_range}")

# 与本地跑出的部分结果交集一致性
old = json.loads(OLD.read_text(encoding="utf-8"))
common = set(new) & set(old)
print(f"与本地部分结果交集: {len(common)}")
if common:
    diffs = []
    same = 0
    for k in common:
        a = new[k]["bbox"]
        b = old[k]["bbox"]
        d = max(abs(x - y) for x, y in zip(a, b))
        diffs.append(d)
        if d <= 1e-6:
            same += 1
    print(f"完全一致(<=1e-6): {same}/{len(common)}")
    print(f"最大坐标差: {max(diffs):.6f}, 平均最大差: {sum(diffs)/len(diffs):.6f}")

# 与测试集 key 对齐
test = json.loads(TEST.read_text(encoding="utf-8"))
print(f"测试集条目: {len(test)}; 缺失: {len(set(test)-set(new))}; 多余: {len(set(new)-set(test))}")

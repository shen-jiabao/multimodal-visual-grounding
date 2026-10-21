#!/bin/bash
# 第1步：抽 20 条对比 enriched vs 原句（只读）
set -u
QJSON="/root/dataset/初赛数据集-基于大模型的多模态视觉理解与推理/queries/queries.json"
ENRICH="/root/output/enriched_queries_qwen25.json"

python3 - "$QJSON" "$ENRICH" <<'PY'
import json, sys, random
orig = json.load(open(sys.argv[1]))
enr  = json.load(open(sys.argv[2]))
print(f"原句 queries.json 条目: {len(orig)}  类型: {type(orig).__name__}")
print(f"enriched 条目: {len(enr)}  类型: {type(enr).__name__}")

# 统一成 {qid: query} 视图
def norm(d):
    out = {}
    if isinstance(d, dict):
        for k, v in d.items():
            if isinstance(v, dict) and 'query' in v:
                out[k] = v['query']
            elif isinstance(v, str):
                out[k] = v
    return out

vo, ve = norm(orig), norm(enr)
common = [k for k in vo if k in ve]
print(f"共有的 qid 数: {len(common)}")

# 结构示例
print("\n--- enriched 结构试探 ---")
sample_key = next(iter(enr))
print("enr 顶层 key 示例:", repr(sample_key))
print("enr 顶层 value 示例:", repr(enr[sample_key])[:200])

random.seed(42)
picks = random.sample(common, 20)
print("\n===== 20 条对比 =====")
same = 0
for k in picks:
    o, e = vo[k], ve[k]
    is_same = (o.strip() == e.strip())
    same += is_same
    print(f"\n[{k}] {'同' if is_same else '异'}")
    print(f"  O: {o}")
    print(f"  E: {e}")
print(f"\n完全相同的条数: {same}/20")
PY
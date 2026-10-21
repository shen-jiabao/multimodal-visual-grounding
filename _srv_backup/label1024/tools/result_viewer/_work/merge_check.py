import json, glob, os, sys

base = r"E:\AIC人工智能算法精英大赛-基于大模型的多模态视觉理解与推理赛道\tools\result_viewer\_work"
src = os.path.join(base, "unique_queries.txt")
with open(src, encoding="utf-8") as f:
    lines = f.read().splitlines()
lines = [ln for ln in lines if ln.strip() != ""]
print("unique_queries 行数:", len(lines))

merged = {}
for p in sorted(glob.glob(os.path.join(base, "zh_part_*.json"))):
    with open(p, encoding="utf-8") as f:
        d = json.load(f)
    print(os.path.basename(p), "keys:", len(d))
    merged.update(d)
print("合并后 keys:", len(merged))

missing = []
for i, ln in enumerate(lines, start=1):
    if ln not in merged:
        missing.append((i, ln))
print("缺失条数:", len(missing))
for i, ln in missing[:60]:
    print("MISS", i, ln[:120])

extra = [k for k in merged if k not in set(lines)]
print("多余 keys:", len(extra))
for k in extra[:30]:
    print("EXTRA", k[:120])

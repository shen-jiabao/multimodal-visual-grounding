#!/bin/bash
# 打印 config.json 全字段（含嵌套），确认层数/维度/dtype 声明与视觉塔
set -u
D=/root/rivermind-data/models/Qwen3.8-27B
/root/rivermind-data/venv_q38/bin/python - "$D" <<'PY'
import json, sys, os
d = sys.argv[1]
cfg = json.load(open(os.path.join(d, "config.json"), encoding="utf-8"))

def show(prefix, o, depth=0):
    for k, v in o.items():
        if isinstance(v, dict) and depth < 2:
            print(f"  {prefix}{k}:")
            show(prefix + "  ", v, depth + 1)
        else:
            s = str(v)
            if len(s) > 90: s = s[:90] + "…"
            print(f"  {prefix}{k} = {s}")

print("=== config.json 全字段 ===")
show("", cfg)

print()
print("=== 目录内非权重文件 ===")
for f in sorted(os.listdir(d)):
    if not f.endswith(".safetensors"):
        p = os.path.join(d, f)
        print(f"  {f:34s} {os.path.getsize(p):>10,} B")

print()
print("=== generation_config.json ===")
gp = os.path.join(d, "generation_config.json")
if os.path.exists(gp):
    g = json.load(open(gp, encoding="utf-8"))
    for k, v in g.items():
        print(f"  {k} = {v}")
else:
    print("  （无）")
PY
echo "=== DONE ==="

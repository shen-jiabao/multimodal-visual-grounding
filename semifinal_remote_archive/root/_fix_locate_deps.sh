#!/bin/bash
# 一次性扫出 remote code 的全部第三方依赖并在隔离 venv 里补齐，
# 避免"装一个、跑一次、再报一个"的逐轮试错。
set -u
V=/srv/venv-la2
PY=/root/miniconda3/envs/myconda/bin/python
exec > >(tee -a /srv/output/fix_deps.log) 2>&1
echo "=== [$(date)] 扫描 remote code 依赖 ==="

echo "== 1) remote code 里出现的第三方模块 =="
$PY - <<'PY'
import ast, os, sys
root = "/srv/models/LocateAnything-3B"
mods = set()
for dirpath, _, files in os.walk(root):
    for f in files:
        if not f.endswith(".py"):
            continue
        p = os.path.join(dirpath, f)
        try:
            tree = ast.parse(open(p, encoding="utf-8").read())
        except Exception:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    mods.add(a.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module and node.level == 0:
                    mods.add(node.module.split(".")[0])
local = {os.path.splitext(f)[0] for _, _, fs in os.walk(root)
         for f in fs if f.endswith(".py")}
std = set(sys.stdlib_module_names)
third = sorted(m for m in mods if m not in std and m not in local)
print("  第三方候选:", third)
open("/tmp/need_mods.txt", "w").write("\n".join(third))
PY

echo
echo "== 2) 在 venv 里逐个试 import，列出缺失 =="
$V/bin/python - <<'PY'
missing = []
for m in open("/tmp/need_mods.txt").read().split():
    try:
        __import__(m)
        print(f"  OK      {m}")
    except Exception as e:
        print(f"  MISSING {m}  ({type(e).__name__})")
        missing.append(m)
open("/tmp/miss_mods.txt", "w").write("\n".join(missing))
PY

echo
echo "== 3) 安装缺失项（官方 PyPI）=="
MISS=$(cat /tmp/miss_mods.txt | tr '\n' ' ')
if [ -z "$(echo $MISS | tr -d ' ')" ]; then
  echo "  无缺失"
else
  echo "  将安装: $MISS"
  $V/bin/pip install -q --index-url https://pypi.org/simple/ $MISS 2>/tmp/pip_fix.log
  echo "  PIP_RC=$?"
  tail -5 /tmp/pip_fix.log
fi

echo
echo "== 4) 复检 =="
$V/bin/python - <<'PY'
bad = []
for m in open("/tmp/need_mods.txt").read().split():
    try:
        __import__(m)
    except Exception as e:
        bad.append((m, type(e).__name__))
print("  仍缺失:", bad if bad else "无")
PY

echo
echo "== 5) 重跑复赛小样探针 =="
PROBE_PY=$V/bin/python bash /root/_probe_locate_semi.sh
echo "=== [$(date)] 结束 ==="

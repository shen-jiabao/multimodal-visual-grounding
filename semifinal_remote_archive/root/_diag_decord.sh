#!/bin/bash
# 诊断 decord / lmdb 在 py3.13 上的可用性——这决定 LocateAnything 能否跑起来。
# 背景：venv 里两者都"已安装"，但 pip check 报 decord 平台不支持；
#       myconda 里两者都缺。必须先搞清 decord 到底能不能 import。
PY=/root/miniconda3/envs/myconda/bin/python
VPY=/srv/venv-locate/bin/python

echo "===== A) venv 侧 import 实测 ====="
$VPY - <<'PY' 2>&1 | tail -12
for m in ("lmdb", "decord"):
    try:
        mod = __import__(m)
        print(f"  venv {m:8s} OK   ver={getattr(mod,'__version__','?')} file={getattr(mod,'__file__','?')}")
    except Exception as e:
        print(f"  venv {m:8s} FAIL {type(e).__name__}: {e}")
PY

echo
echo "===== B) myconda 侧 import 实测 ====="
$PY - <<'PY' 2>&1 | tail -12
for m in ("lmdb", "decord"):
    try:
        mod = __import__(m)
        print(f"  myconda {m:8s} OK   ver={getattr(mod,'__version__','?')}")
    except Exception as e:
        print(f"  myconda {m:8s} FAIL {type(e).__name__}: {e}")
PY

echo
echo "===== C) py3.13 上 decord 有无可用 wheel ====="
$PY -m pip index versions decord 2>&1 | head -5
echo "--- lmdb ---"
$PY -m pip index versions lmdb 2>&1 | head -5

echo
echo "===== D) remote code 到底在哪用到 decord/lmdb ====="
grep -rn "decord\|lmdb" /srv/models/LocateAnything-3B/*.py /srv/models/LocateAnything-3B/*/*.py 2>/dev/null | head -20

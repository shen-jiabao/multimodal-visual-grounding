#!/bin/bash
# LocateAnything 第二方案：新建 venv 挂 myconda（继承 transformers 4.57.6，
# 其 AutoProcessor 实测可导入），只往**新建 venv** 里补 decord+lmdb。
#
# 为什么不直接往 myconda 装：myconda 正是当前 8B 全量推理的解释器，
# 装 decord 可能连带改 numpy 等共享包，风险不可接受——隔离到 venv 里，
# myconda 的 site-packages 一个字节都不动。
set -u
PY=/root/miniconda3/envs/myconda/bin/python
V=/srv/venv-la2
LOG=/srv/output/setup_la2.log
exec > >(tee -a $LOG) 2>&1
echo "=== [$(date)] 建隔离 venv 并补依赖 ==="

echo "== 1) 建 venv（--system-site-packages，继承 myconda 的 torch/transformers）=="
rm -rf $V
$PY -m venv --system-site-packages $V
$V/bin/python - <<'PY'
import transformers
print("  继承 transformers", transformers.__version__)
from transformers import AutoProcessor, AutoModel, AutoTokenizer, AutoConfig
print("  AutoProcessor / AutoModel / AutoTokenizer / AutoConfig 全部可导入")
PY

echo
echo "== 2) 只往本 venv 装 decord + lmdb（官方 PyPI）=="
$V/bin/pip install -q --index-url https://pypi.org/simple/ decord lmdb 2>/tmp/pip_la2.log
echo "  PIP_RC=$?"
tail -4 /tmp/pip_la2.log

echo
echo "== 3) 自检（含 remote code 静态扫描依赖的包）=="
$V/bin/python - <<'PY'
for m in ("torch", "transformers", "decord", "lmdb", "PIL", "numpy"):
    try:
        mod = __import__(m)
        print(f"  {m:14s}", getattr(mod, "__version__", "?"))
    except Exception as e:
        print(f"  {m:14s} FAIL {type(e).__name__}: {e}")
PY

echo
echo "== 4) 用该 venv 跑复赛小样探针 =="
PROBE_PY=$V/bin/python bash /root/_probe_locate_semi.sh
echo "=== [$(date)] 结束 ==="

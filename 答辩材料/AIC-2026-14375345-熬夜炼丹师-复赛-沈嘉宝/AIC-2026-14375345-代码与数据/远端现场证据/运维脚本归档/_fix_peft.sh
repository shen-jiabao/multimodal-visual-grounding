#!/bin/bash
# 只补 transformers 的 check_imports 实际点名的包（peft），不批量装
# 本地子包(batch_utils/kernel_utils)或可选加速库(flash_attn/magi_attention)——
# 上一版脚本把它们当 PyPI 包，导致 pip 整体失败、peft 漏装。
set -u
V=/srv/venv-la2
exec > >(tee -a /srv/output/fix_peft.log) 2>&1
echo "=== [$(date)] 补 peft ==="

$V/bin/pip install -q --index-url https://pypi.org/simple/ peft 2>/tmp/pip_peft.log
echo "PIP_RC=$?"
tail -6 /tmp/pip_peft.log

echo
echo "== 复检 import =="
$V/bin/python - <<'PY'
for m in ("torch", "transformers", "peft", "decord", "lmdb", "PIL", "numpy"):
    try:
        mod = __import__(m)
        print(f"  {m:14s}", getattr(mod, "__version__", "?"))
    except Exception as e:
        print(f"  {m:14s} FAIL {type(e).__name__}: {e}")
PY

echo
echo "== 重跑复赛小样探针 =="
PROBE_PY=$V/bin/python bash /root/_probe_locate_semi.sh
echo "=== [$(date)] 结束 ==="

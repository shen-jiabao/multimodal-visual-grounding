#!/bin/bash
# 找 vLLM 0.23.0 里禁用 custom all-reduce 的真实开关（CLI / config / env）
set -u
V=/root/rivermind-data/venv_q38/lib/python3.10/site-packages/vllm

echo "=== ① 源码中的 disable_custom_all_reduce ==="
grep -rn "disable_custom_all_reduce" "$V" --include=*.py | head -20 | sed 's/^/  /'

echo; echo "=== ② 环境变量开关 ==="
grep -rn "VLLM_DISABLE_CUSTOM_ALL_REDUCE\|DISABLE_CUSTOM_ALL_REDUCE" "$V" --include=*.py | head -10 | sed 's/^/  /'

echo; echo "=== ③ 相关 envs（config/envs.py 里 all_reduce 段）==="
grep -rn "ALL_REDUCE" "$V/config/envs.py" 2>/dev/null | head -20 | sed 's/^/  /'

echo; echo "=== ④ CLI 里带 all-reduce 的选项（含别名）==="
grep -rn "custom-all-reduce\|custom_all_reduce" "$V/engine/arg_utils.py" "$V/entrypoints" 2>/dev/null | head -20 | sed 's/^/  /'

echo; echo "=== ⑤ 当前 vllm serve 是否支持该参数（直接问 argparse）==="
/root/rivermind-data/venv_q38/bin/python - <<'PY' 2>&1 | sed 's/^/  /'
import inspect
from vllm.engine.arg_utils import EngineArgs
import dataclasses
names = [f.name for f in dataclasses.fields(EngineArgs)]
hit = [n for n in names if "all_reduce" in n or "custom" in n]
print("EngineArgs 里含 all_reduce/custom 的字段：", hit)
PY
echo "=== 结束 ==="

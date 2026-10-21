#!/bin/bash
# 无卡环境准备与核查（目标：GPU 一开就能直接起服务）
set -u
R=/root/rivermind-data
V=$R/venv/bin/python
D=$R/dataset

echo "=== 1) 数据集：C 路线所需的 1150 张 visible 是否已就位 ==="
"$V" - <<'PYEOF'
import json, os
R = "/root/rivermind-data"
q = json.load(open(f"{R}/dataset/queries/queries.json", encoding="utf-8"))
cores = set(json.load(open(f"{R}/_ordinal_cores_r2.json", encoding="utf-8"))) \
    if os.path.exists(f"{R}/_ordinal_cores_r2.json") else set()
have = set(os.listdir(f"{R}/dataset/Images/visible"))
if cores:
    need = {os.path.basename(v["visible"]) for k, v in q.items() if k not in cores}
    miss = sorted(need - have)
    print(f"  序数表已就位 ⇒ C 作用范围需 {len(need)} 张，缺 {len(miss)} 张")
    if miss:
        print("  ★缺样例:", miss[:5])
else:
    print(f"  ★序数表不在远端（需上传 _ordinal_cores_r2.json），现有 visible {len(have)} 张")
print(f"  visible 目录共 {len(have)} 张")
PYEOF

echo
echo "=== 2) 依赖：modelscope（下载用）是否已装 ==="
if "$V" -c "import modelscope; print('  已装 modelscope', modelscope.__version__)" 2>/dev/null; then
  echo "  无需安装"
else
  echo "  未装 ⇒ 安装（下载步需要）"
  "$V" -m pip install -q modelscope 2>&1 | tail -3
  "$V" -c "import modelscope; print('  安装后版本', modelscope.__version__)" 2>/dev/null \
    || echo "  ★安装失败（GPU 前需解决）"
fi

echo
echo "=== 3) vLLM 的 AWQ 支持（★GPU 前必须确认，否则开了卡也起不来）==="
"$V" - <<'PYEOF' 2>/dev/null
import os, vllm
sp = os.path.dirname(vllm.__file__)
qd = os.path.join(sp, "model_executor", "layers", "quantization")
print("  vllm", vllm.__version__)
if os.path.isdir(qd):
    files = sorted(f for f in os.listdir(qd) if "awq" in f.lower())
    print("  quantization 目录下与 awq 相关:", files or "★无")
try:
    from vllm.model_executor.layers.quantization import get_quantization_config
    for m in ("awq", "awq_marlin", "gptq_marlin", "fp8"):
        try:
            get_quantization_config(m); print(f"  ✓ 量化方法已注册: {m}")
        except Exception as ex:
            print(f"  ✗ {m}: {type(ex).__name__}")
except Exception as ex:
    print("  查询失败:", type(ex).__name__, str(ex)[:100])
PYEOF

echo
echo "=== 4) 磁盘总账（决定模型能否预下）==="
df -h 2>/dev/null | grep -Ev 'tmpfs|udev' | sed 's/^/  /'
echo "  --- 各挂载可用（GB）---"
df -BG --output=target,avail 2>/dev/null | tail -n +2 | sed 's/^/  /'

echo
echo "=== 5) GPU 现状（预期：无）==="
ls /dev/nvidia[0-9]* 2>/dev/null | head -3 || echo "  ★无设备节点（符合预期，等你开卡）"

echo
echo "=== 6) 关键脚本是否已在远端 ==="
for f in _qwen_full_pipeline.py _ordinal_cores_r2.json _c_sample300.json; do
  [ -e "$R/$f" ] && echo "  [有] $f  ($(stat -c%s "$R/$f") B)" || echo "  [缺] $f"
done
echo "=== 结束 ==="
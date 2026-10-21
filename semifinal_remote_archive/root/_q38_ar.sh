#!/bin/bash
# 诊断 custom_all_reduce.cuh:455 'invalid argument'（TP=2 起服时 TP1 死）
set -u
R=/root/rivermind-data
VENV=$R/venv_q38

echo "=== ① 驱动 / CUDA ==="
head -3 /proc/driver/nvidia/version 2>/dev/null | sed 's/^/  /'
nvidia-smi --query-gpu=index,name,memory.total,driver_version,compute_cap --format=csv,noheader | sed 's/^/  /'

echo; echo "=== ② 拓扑 ==="
nvidia-smi topo -m 2>&1 | sed 's/^/  /'

echo; echo "=== ③ P2P 能力（torch 视角）==="
$VENV/bin/python - <<'PY' 2>&1 | sed 's/^/  /'
try:
    import torch
    n = torch.cuda.device_count()
    print("device_count", n, "| torch", torch.__version__)
    for i in range(n):
        p = torch.cuda.get_device_properties(i)
        print(f"  [{i}] {p.name} cc={p.major}.{p.minor} pci={p.pci_bus_id if hasattr(p,'pci_bus_id') else '?'}")
    if n >= 2:
        print("  can_device_access_peer(0,1) =", torch.cuda.can_device_access_peer(0, 1))
except Exception as e:
    print("  ★", type(e).__name__, e)
PY

echo; echo "=== ④ vLLM 是否有禁用 custom all-reduce 的开关 ==="
$VENV/bin/vllm serve --help 2>&1 | grep -inE "custom.all.reduce|all.reduce" | head -10 | sed 's/^/  /'

echo; echo "=== ⑤ 环境变量开关（vLLM 侧）==="
$VENV/bin/python - <<'PY' 2>&1 | sed 's/^/  /'
import inspect
try:
    from vllm.distributed.device_communicators import custom_all_reduce as C
    src = inspect.getsource(C.init_custom_all_reduce)
    print(src[:1200])
except Exception as e:
    print("★", type(e).__name__, e)
PY
echo "=== 结束 ==="

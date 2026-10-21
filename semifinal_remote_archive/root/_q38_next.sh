#!/bin/bash
# 下一杠杆的可行性核查：显存/磁盘/模型支持/序数层可用性
set -u
R=/root/rivermind-data
V38=$R/venv_q38

echo "=== ① GPU ==="
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader | sed 's/^/  /'
echo "  服务：$(curl -s --max-time 5 http://127.0.0.1:8000/v1/models 2>/dev/null | grep -o 'qwen38-27b' | head -1 || echo '未响应')"

echo; echo "=== ② 磁盘 ==="
df -h $R | tail -1 | sed 's/^/  /'
du -sh $R/models/* 2>/dev/null | sed 's/^/  /'
du -sh $R/venv* 2>/dev/null | sed 's/^/  /'
echo "  output：$(du -sh $R/output 2>/dev/null | cut -f1)"

echo; echo "=== ③ vLLM 是否支持 InternVL / 其他大地基模型 ==="
$V38/bin/python - <<'PY' 2>&1 | tail -20 | sed 's/^/  /'
try:
    from vllm.model_executor.models.registry import ModelRegistry
    archs = ModelRegistry.get_supported_archs()
    print("架构总数", len(archs))
    for pat in ("InternVL", "Qwen3VL", "Qwen2_5_VL", "Llama", "Gemma3", "Mllama"):
        hit = sorted(a for a in archs if pat.lower() in a.lower())
        print(f"  {pat}: {hit[:6]}")
except Exception as e:
    print("★", type(e).__name__, e)
PY

echo; echo "=== ④ 序数层现状（1103 条） ==="
$V38/bin/python - <<'PY' 2>&1 | sed 's/^/  /'
import json, os
R = "/root/rivermind-data"
cores = set(json.load(open(f"{R}/_ordinal_cores_r2.json", encoding="utf-8")))
print("序数核心词表条数", len(cores))
for n in ("result_semi_n32bf16_nonord.json", "result_semi_n32_orddirect.json"):
    print(f"  基线件 {n}:", "在" if os.path.exists(f"{R}/{n}") else "不在")
print("  output 目录:", sorted(os.listdir(f"{R}/output"))[:12] if os.path.isdir(f"{R}/output") else "无")
PY

echo; echo "=== ⑤ 序数层相关脚本/产物是否在场 ==="
ls -1 $R | grep -iE "ord|som|q32bf16|probe" | head -20 | sed 's/^/  /'
echo "=== 结束 ==="

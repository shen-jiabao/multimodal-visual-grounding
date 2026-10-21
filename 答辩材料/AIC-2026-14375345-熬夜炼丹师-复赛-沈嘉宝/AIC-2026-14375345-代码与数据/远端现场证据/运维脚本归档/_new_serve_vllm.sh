#!/bin/bash
# 核对 32B-AWQ 权重完整性，然后启动 vLLM 服务（配置对齐归档实例）。
#
# 归档实例的 vLLM 参数（证据：vllm_restart.log）
#   model=/root/models/Qwen3-VL-32B-Instruct-AWQ  served_model_name=qwen3-vl-32b
#   max_model_len=32768  gpu_memory_utilization=0.92  max_num_seqs=8
# 本机显存 40 GB（归档那台 A40 是 49 GB），故 util 降到 0.90；
# util 只影响 KV cache 大小与批处理，不改模型数学，故 temperature=0 下输出不变。
set -u
VENV=/root/rivermind-data/venv
M=/root/rivermind-data/models/Qwen3-VL-32B-Instruct-AWQ
LOG=/root/rivermind-data/vllm_serve.log

echo "=== 1) 权重完整性 ==="
n_sf=$(ls "$M"/*.safetensors 2>/dev/null | wc -l)
echo "  safetensors 分片: $n_sf"
for f in config.json model.safetensors.index.json preprocessor_config.json; do
  [ -f "$M/$f" ] && echo "  有 $f ($(stat -c%s "$M/$f") 字节)" || echo "  ★缺 $f"
done
python3 - "$M" <<'PYEOF'
import json, os, sys
M = sys.argv[1]
idx = json.load(open(os.path.join(M, "model.safetensors.index.json")))
shards = sorted(set(idx["weight_map"].values()))
bad = [(s, os.path.getsize(os.path.join(M, s))) for s in shards
       if not os.path.exists(os.path.join(M, s)) or os.path.getsize(os.path.join(M, s)) < 1024 * 1024]
print("  索引要求分片 " + str(len(shards)) + " 个；缺失或过小 " + str(len(bad)))
if bad:
    print("  ★问题分片: " + str(bad[:5]))
    sys.exit(9)
tot = sum(os.path.getsize(os.path.join(M, s)) for s in shards)
print("  分片合计 " + format(tot / 1024**3, ".2f") + " GB")
PYEOF
[ $? -eq 9 ] && { echo "★ 权重不完整，拒绝启动"; exit 9; }

echo
echo "=== 2) 启动 vLLM（后台）==="
if pgrep -f "vllm serve" > /dev/null; then
  echo "  已在运行 PID=$(pgrep -f 'vllm serve' | head -1)，跳过"
else
  cd /root
  setsid nohup "$VENV/bin/vllm" serve "$M" \
    --served-model-name qwen3-vl-32b \
    --host 0.0.0.0 --port 8000 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.90 \
    --max-num-seqs 8 \
    > "$LOG" 2>&1 < /dev/null &
  echo "  已发起启动，日志 $LOG"
fi

echo
echo "=== 3) 等待服务就绪（最多 8 分钟）==="
for i in $(seq 1 48); do
  sleep 10
  if curl -s --max-time 5 http://127.0.0.1:8000/v1/models 2>/dev/null | grep -q "qwen3-vl-32b"; then
    echo "  ✓ 服务就绪（等待 $((i*10)) 秒）"
    curl -s http://127.0.0.1:8000/v1/models | head -c 300; echo
    break
  fi
  if [ "$i" = "48" ]; then
    echo "  ★ 8 分钟内未就绪，日志尾部："
    tail -25 "$LOG"
    exit 1
  fi
  [ $((i % 6)) -eq 0 ] && echo "    …已等 $((i*10)) 秒，日志尾: $(tail -1 "$LOG" | cut -c1-110)"
done

echo
echo "=== 4) 显存占用 ==="
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
#!/bin/bash
# 模型可用性与网络速率实测：决定 20 GB 权重与 3.93 GB 数据要等多久。
RIVER=/root/rivermind-data
mkdir -p "$RIVER/models" "$RIVER/dataset" "$RIVER/ms-cache"

echo "=== 1) ModelScope 上是否存在 Qwen3-VL-32B-Instruct-AWQ ==="
for mid in "Qwen/Qwen3-VL-32B-Instruct-AWQ" "Qwen/Qwen3-VL-32B-Instruct" "Qwen/Qwen3-VL-30B-A3B-Instruct-AWQ"; do
  code=$(curl -s -o /tmp/ms.json -w '%{http_code}' --max-time 20 \
    "https://www.modelscope.cn/api/v1/models/$mid")
  size=$(python3 -c "import json;d=json.load(open('/tmp/ms.json'));print(d.get('Data',{}).get('Name','?'), d.get('Data',{}).get('Id','?'))" 2>/dev/null || echo "解析失败")
  echo "  $mid -> HTTP $code  $size"
done

echo
echo "=== 2) 实测从 ModelScope 的下载速率（取一个真实权重分片，限时 20 秒）==="
curl -s --max-time 20 -o /dev/null -w '  速率 %{speed_download} B/s  已下 %{size_download} 字节\n' \
  -r 0-80000000 \
  "https://www.modelscope.cn/models/Qwen/Qwen3-VL-32B-Instruct-AWQ/resolve/master/model-00001-of-00005.safetensors" 2>&1 | tail -2

echo
echo "=== 3) 实测 hf-mirror 速率（备选源）==="
curl -s --max-time 20 -o /dev/null -w '  速率 %{speed_download} B/s  已下 %{size_download} 字节\n' \
  -r 0-80000000 \
  "https://hf-mirror.com/Qwen/Qwen3-VL-32B-Instruct-AWQ/resolve/main/model-00001-of-00005.safetensors" 2>&1 | tail -2

echo
echo "=== 4) 磁盘现状 ==="
df -h /root/rivermind-data | tail -1
df -h / | tail -1

echo
echo "=== 5) pip 装 vllm 的可见版本（不装，只看能否解析到 0.11.0）==="
/root/rivermind-data/venv/bin/pip index versions vllm 2>&1 | head -3 || \
  /root/rivermind-data/venv/bin/pip install --dry-run vllm==0.11.0 2>&1 | tail -5
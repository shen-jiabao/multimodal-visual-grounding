#!/bin/bash
# 重试起服【保守配置】+ 先取上次 OOM 的完整上下文（避免凭猜）
#
# 上次失败：TP=2 / util=0.90 / maxlen=32768，加载到 57% 时 GPU0 已用 35.22 GiB
#   （PyTorch 占 33.12 GiB）再要 5.86 GiB ⇒ OOM。
#   权重按 TP=2 应只占 31 GiB/卡 ⇒ 33.12 GiB **对不上**，需看日志定位多出来的是什么。
set -u
R=/root/rivermind-data
LOG=$R/vllm32bf16.log

echo "=== 0) 清场 ==="
pkill -f "_get_serve_q32bf16.sh" 2>/dev/null && echo "  已停起服脚本" || echo "  （无起服脚本）"
pkill -f "vllm serve" 2>/dev/null && echo "  已停 vllm serve" || echo "  （无 vllm）"
sleep 5
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | sed 's/^/  /'

echo; echo "=== 1) 上次 OOM 的现场（关键上下文，不猜） ==="
echo "  -- 内存/加载相关行 --"
grep -nE "memory|GiB|Loading safetensors|load_format|vision|encoder|mm_|duplicat" "$LOG" 2>/dev/null \
  | tail -25 | cut -c1-160 | sed 's/^/  /'
echo "  -- 配置行 --"
grep -nE "non-default args|max_model_len|tensor_parallel|gpu_memory_util|enforce_eager|Architecture|Resolved" "$LOG" 2>/dev/null \
  | head -25 | cut -c1-160 | sed 's/^/  /'

echo; echo "=== 2) 模型体量核对（解释 33 GiB 的来源） ==="
/root/rivermind-data/venv/bin/python - /root/rivermind-data/models/Qwen3-VL-32B-Instruct <<'PY'
import json, os, sys
d = sys.argv[1]
c = json.load(open(os.path.join(d, "config.json"), encoding="utf-8"))
print(f"  architectures={c.get('architectures')}")
print(f"  model_type={c.get('model_type')}  tie_word_embeddings={c.get('tie_word_embeddings')}")
txt = c.get("text_config", {})
vis = c.get("vision_config", {})
print(f"  text: layers={txt.get('num_hidden_layers')} hidden={txt.get('hidden_size')} "
      f"vocab={txt.get('vocab_size')}")
print(f"  vision: depth={vis.get('depth')} hidden={vis.get('hidden_size')}")
m = json.load(open(os.path.join(d, "model.safetensors.index.json"), encoding="utf-8"))["weight_map"]
# 按前缀归类参数量（字节数近似：bf16 2 B/参数）
import collections
size_of = lambda n: os.path.getsize(os.path.join(d, n)) if os.path.exists(os.path.join(d, n)) else 0
# 无法逐权重取大小，改按分片统计
shards = sorted(set(m.values()))
tot = sum(size_of(s) for s in shards)
print(f"  分片合计 {tot/1024**3:.2f} GiB（15 位精度：{tot/1024**3:.4f}）")
print(f"  ⇒ TP=2 每卡权重应约 {tot/1024**3/2:.2f} GiB")
# 视觉塔占比：按权重名里含 visual 的条目数近似
vis_keys = [k for k in m if "visual" in k or "vision" in k]
print(f"  视觉塔权重条目 {len(vis_keys)} / {len(m)}（{len(vis_keys)/len(m):.1%}）")
PY

echo; echo "=== 3) 保守配置重试（eager + 短上下文 + 高 util） ==="
echo "  配置：TP=2 util=0.95 maxlen=8192 extra='--enforce-eager' alloc=expandable_segments:True"
cd $R
: > serve32bf16b.log
Q32_UTIL=0.95 Q32_MAXLEN=8192 Q32_EXTRA="--enforce-eager" Q32_ATTN=default \
  setsid nohup bash _get_serve_q32bf16.sh > serve32bf16b.log 2>&1 < /dev/null &
echo "  已启动（后台），日志 $R/serve32bf16b.log"

echo; echo "=== 4) 观察 150 秒 ==="
sleep 150
tail -18 serve32bf16b.log | sed 's/^/  /'
echo
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | sed 's/^/  /'
echo "=== 结束 ==="
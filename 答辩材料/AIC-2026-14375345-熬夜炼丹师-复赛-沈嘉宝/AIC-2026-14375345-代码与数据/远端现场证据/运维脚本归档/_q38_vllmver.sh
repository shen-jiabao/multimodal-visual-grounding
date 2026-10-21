#!/bin/bash
# 找「最早登记 Qwen3_5ForConditionalGeneration 的 vLLM 版本」（秒级，不安装）
set -u
echo "=== 各版本 vLLM registry 是否登记 Qwen3_5* ==="
for tag in v0.30.0 v0.29.0 v0.28.0 v0.27.1 v0.26.0 v0.25.1 v0.24.0 v0.23.0 v0.22.1 v0.21.0 v0.20.2; do
  url="https://raw.githubusercontent.com/vllm-project/vllm/$tag/vllm/model_executor/models/registry.py"
  body=$(curl -sL --max-time 25 "$url" 2>/dev/null)
  if [ -z "$body" ]; then echo "  $tag  取不到"; continue; fi
  hit=$(echo "$body" | grep -oE '"Qwen3_5[A-Za-z_]*"' | sort -u | tr '\n' ' ')
  if [ -n "$hit" ]; then echo "  ✓ $tag  → $hit"; else echo "  ✗ $tag  无 Qwen3_5"; fi
done

echo; echo "=== 是否登记**多模态**变体 Qwen3_5ForConditionalGeneration ==="
for tag in v0.30.0 v0.29.0 v0.28.0 v0.27.1 v0.26.0; do
  url="https://raw.githubusercontent.com/vllm-project/vllm/$tag/vllm/model_executor/models/registry.py"
  curl -sL --max-time 25 "$url" 2>/dev/null \
    | grep -c "Qwen3_5ForConditionalGeneration" \
    | xargs -I{} echo "  $tag  Qwen3_5ForConditionalGeneration 出现 {} 次"
done

echo; echo "=== 该版本的 python/torch 要求（看是否与 A100+cu128 相容） ==="
for tag in v0.30.0; do
  echo "  --- $tag setup/requirements 里的 torch 约束 ---"
  curl -sL --max-time 25 "https://raw.githubusercontent.com/vllm-project/vllm/$tag/requirements/common.txt" 2>/dev/null | grep -iE "^torch|^transformers" | head -5 | sed 's/^/    /'
done
echo "=== 结束 ==="
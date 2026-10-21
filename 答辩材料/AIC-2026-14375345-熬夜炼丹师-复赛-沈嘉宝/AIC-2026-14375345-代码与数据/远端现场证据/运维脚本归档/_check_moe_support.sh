#!/bin/bash
# 关键前置：vLLM 0.11.0 是否支持 Qwen3VLMoeForConditionalGeneration（235B-A22B 的 MoE 架构）
# 不支持则整条路线作废 —— 必须在下载 116 GB 之前查清。不需要 GPU。
set -u
V=/root/rivermind-data/venv/bin/python

echo "=== 1) registry 里全部 Qwen3 相关架构 ==="
"$V" - <<'PYEOF' 2>/dev/null
from vllm.model_executor.models.registry import ModelRegistry
reg = ModelRegistry.get_supported_archs()
qs = {k: v for k, v in reg.items() if "Qwen3" in k or "qwen3" in k}
if not qs:
    print("  ★一个都没有")
for k in sorted(qs):
    print(f"  {k:44s} -> {qs[k]}")
print()
hit = [k for k in reg if "Qwen3VLMoe" in k]
print("  ★Qwen3VLMoeForConditionalGeneration 是否注册:", 
      "是 → " + str(reg[hit[0]]) if hit else "否")
PYEOF

echo
echo "=== 2) vLLM 源码里是否有 MoE 的 Qwen3-VL 实现 ==="
SP=$("$V" -c "import vllm,os;print(os.path.dirname(vllm.__file__))" 2>/dev/null)
echo "  包路径: $SP"
ls -1 "$SP/model_executor/models/" 2>/dev/null | grep -i "qwen3" | sed 's/^/    /'
echo "  --- 在 vLLM 源码树中搜 Qwen3VLMoe 字样 ---"
grep -rl "Qwen3VLMoe" "$SP" 2>/dev/null | head -8 | sed 's|.*site-packages/||; s/^/    /'
echo "  --- 搜 qwen3_vl_moe 字样 ---"
grep -rl "qwen3_vl_moe" "$SP" 2>/dev/null | head -8 | sed 's|.*site-packages/||; s/^/    /'

echo
echo "=== 3) transformers 是否认这个 config（vLLM 会先解析它）==="
"$V" - <<'PYEOF' 2>&1 | tail -12
import transformers as T
print("  transformers:", T.__version__)
try:
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(
        "QuantTrio/Qwen3-VL-235B-A22B-Instruct-AWQ",
        trust_remote_code=True)
    print("  AutoConfig 成功:", type(cfg).__name__)
    print("  model_type:", getattr(cfg, "model_type", None))
except Exception as ex:
    print("  AutoConfig 失败:", type(ex).__name__, str(ex)[:160])
PYEOF
echo "=== 结束 ==="
#!/bin/bash
# 在 venv 里装 vLLM。版本取 **0.11.0**——与归档实例实测成功的那次一致
# （日志证据：vLLM 0.11.0 + quantization=awq_marlin + Resolved architecture:
#  Qwen3VLForConditionalGeneration），是最有把握的组合。
#
# 全部重物落大盘：venv 与 pip 缓存都在 /root/rivermind-data，系统盘只 30 GB。
set -u
RIVER=/root/rivermind-data
VENV=$RIVER/venv
LOG=$RIVER/install_vllm.log
export PIP_CACHE_DIR=$RIVER/pip-cache
export TMPDIR=$RIVER/tmp
mkdir -p "$PIP_CACHE_DIR" "$TMPDIR"

echo "[$(date +%H:%M:%S)] 开始安装 vllm==0.11.0（含 torch 2.8 + CUDA 运行时，约 5-8 GB）"
"$VENV/bin/pip" install -q -U pip 2>&1 | tail -2
"$VENV/bin/pip" install vllm==0.11.0 2>&1 | tail -25
echo "[$(date +%H:%M:%S)] pip 返回码 $?"

echo "=== 校验 ==="
"$VENV/bin/python" - <<'PYEOF'
import importlib.util as u
for n in ("vllm", "torch", "transformers", "openai", "PIL"):
    if not u.find_spec(n):
        print("  " + n.ljust(13) + "MISSING"); continue
    try:
        m = __import__(n)
        v = getattr(m, "__version__", "?")
        extra = ""
        if n == "torch":
            extra = "  cuda_available=" + str(m.cuda.is_available())
            if m.cuda.is_available():
                extra += "  device=" + m.cuda.get_device_name(0)
        print("  " + n.ljust(13) + str(v) + extra)
    except Exception as e:
        print("  " + n.ljust(13) + "装了但导入失败: " + type(e).__name__ + ": " + str(e)[:80])
PYEOF
echo "=== vllm CLI 是否存在 ==="
ls -l "$VENV/bin/vllm" 2>/dev/null || echo "  无 vllm 可执行文件"
echo "=== 体量与磁盘 ==="
du -sh "$VENV" 2>/dev/null
df -h "$RIVER" | tail -1
df -h / | tail -1
echo "[$(date +%H:%M:%S)] DONE"
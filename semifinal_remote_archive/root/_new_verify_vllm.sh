#!/bin/bash
# 验证 vLLM 安装可用性（含与 transformers 大版本是否兼容），并回收磁盘。
set -u
VENV=/root/rivermind-data/venv
echo "=== 1) 可执行文件与版本 ==="
ls -l "$VENV/bin/vllm" 2>/dev/null || echo "  无 bin/vllm（可能需 python -m vllm 调用）"
"$VENV/bin/python" -m vllm --version 2>&1 | tail -3
echo
echo "=== 2) 关键包版本与 CUDA 可用性 ==="
"$VENV/bin/python" - <<'PYEOF'
import importlib.util as u
for n in ("vllm", "torch", "transformers", "openai", "PIL", "ray"):
    if not u.find_spec(n):
        print("  " + n.ljust(13) + "MISSING"); continue
    try:
        m = __import__(n)
        extra = ""
        if n == "torch":
            extra = "  cuda=" + str(m.cuda.is_available())
            if m.cuda.is_available():
                extra += "  dev=" + m.cuda.get_device_name(0)
        print("  " + n.ljust(13) + str(getattr(m, "__version__", "?")) + extra)
    except Exception as e:
        print("  " + n.ljust(13) + "★导入失败: " + type(e).__name__ + ": " + str(e)[:120])
PYEOF
echo
echo "=== 3) 能否解析出 Qwen3VL 架构（不加载权重，只查注册表）==="
"$VENV/bin/python" - <<'PYEOF'
try:
    from vllm.model_executor.models.registry import ModelRegistry
    names = [k for k in ModelRegistry.get_supported_archs() if "Qwen3" in k or "Qwen2" in k]
    print("  支持的 Qwen 架构: " + str(sorted(names)[:12]))
    print("  含 Qwen3VLForConditionalGeneration: "
          + str("Qwen3VLForConditionalGeneration" in ModelRegistry.get_supported_archs()))
except Exception as e:
    print("  ★查询失败: " + type(e).__name__ + ": " + str(e)[:160])
PYEOF
echo
echo "=== 4) 回收磁盘：删 pip 缓存 ==="
before=$(df -h /root/rivermind-data | tail -1 | awk '{print $4}')
rm -rf /root/rivermind-data/pip-cache /root/rivermind-data/tmp
after=$(df -h /root/rivermind-data | tail -1 | awk '{print $4}')
echo "  可用空间 $before -> $after"
echo
echo "=== 5) 现状 ==="
du -sh "$VENV" /root/rivermind-data/models /root/rivermind-data/dataset 2>/dev/null
df -h /root/rivermind-data | tail -1
ls /root/rivermind-data/dataset/Images/visible 2>/dev/null | wc -l
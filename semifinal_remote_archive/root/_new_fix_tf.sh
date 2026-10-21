#!/bin/bash
# 修 transformers 大版本不兼容：vLLM 0.11.0 需要 4.x，pip 却解析到 5.17.0，
# 启动时报 `Qwen2Tokenizer has no attribute all_special_tokens_extended`
# （该属性在 transformers 5.x 被移除）。归档实例的可用环境是 4.57.x。
set -u
VENV=/root/rivermind-data/venv
export PIP_CACHE_DIR=/root/rivermind-data/pip-cache

echo "=== 修前 ==="
"$VENV/bin/python" -c "import transformers;print('  transformers',transformers.__version__)" 2>&1 | tail -1
"$VENV/bin/pip" show vllm 2>/dev/null | grep -E "^Requires" | head -2

echo
echo "=== 降到 4.57.x ==="
"$VENV/bin/pip" install "transformers>=4.56,<5" 2>&1 | tail -8

echo
echo "=== 修后 ==="
"$VENV/bin/python" - <<'PYEOF'
import importlib.metadata as md
for n in ("transformers", "tokenizers", "vllm", "torch", "numpy"):
    try:
        print("  " + n.ljust(13) + md.version(n))
    except Exception as e:
        print("  " + n.ljust(13) + "读版本失败: " + str(e)[:60])
# 直接验证那个报错的属性是否回来
from transformers import AutoTokenizer
t = AutoTokenizer.from_pretrained(
    "/root/rivermind-data/models/Qwen3-VL-32B-Instruct-AWQ")
have = hasattr(t, "all_special_tokens_extended")
print("  tokenizer 类: " + type(t).__name__)
print("  all_special_tokens_extended 存在: " + str(have))
PYEOF

echo
echo "=== 关键：vLLM 自己能不能过 tokenizer 这一步 ==="
"$VENV/bin/python" - <<'PYEOF'
try:
    from vllm.transformers_utils.tokenizer import get_tokenizer
    tk = get_tokenizer("/root/rivermind-data/models/Qwen3-VL-32B-Instruct-AWQ")
    print("  ✓ get_tokenizer 通过: " + type(tk).__name__)
except Exception as e:
    print("  ★ 仍失败: " + type(e).__name__ + ": " + str(e)[:200])
PYEOF
rm -rf /root/rivermind-data/pip-cache
df -h /root/rivermind-data | tail -1
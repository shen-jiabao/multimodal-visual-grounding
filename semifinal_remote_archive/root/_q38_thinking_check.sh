#!/bin/bash
# 查 chat template 的 thinking 行为（对应"thinking 关闭"这一条）
set -u
D=/root/rivermind-data/models/Qwen3.8-27B

echo "=== ① chat_template.jinja 中与 thinking 相关的行 ==="
grep -nE "thinking|reasoning|enable_thinking|think" "$D/chat_template.jinja" 2>/dev/null | head -25 | sed 's/^/  /' \
  || echo "  （无匹配）"

echo
echo "=== ② tokenizer_config.json 里的 chat_template 开关 ==="
/root/rivermind-data/venv_q38/bin/python - "$D" <<'PY'
import json, os, sys
d = sys.argv[1]
tc = json.load(open(os.path.join(d, "tokenizer_config.json"), encoding="utf-8"))
print("  顶层键:", [k for k in tc if k not in ("added_tokens_decoder",)])
ct = tc.get("chat_template")
if ct:
    print(f"  chat_template 内嵌长度 {len(ct)} 字符；含 'thinking': {'thinking' in ct}")
    # 打印含 thinking 的片段
    for line in ct.splitlines():
        if "thinking" in line or "reasoning" in line:
            print("   |", line.strip()[:160])
else:
    print("  （tokenizer_config 无内嵌 chat_template，模板在 .jinja 文件里）")
for k in ("enable_thinking", "default_thinking", "thinking"):
    if k in tc:
        print(f"  {k} = {tc[k]}")
PY

echo
echo "=== ③ preprocessor_config.json（图像预处理，决定三模态尺寸口径） ==="
cat "$D/preprocessor_config.json" | sed 's/^/  /'
echo "=== DONE ==="

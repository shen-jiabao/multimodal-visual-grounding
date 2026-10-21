#!/bin/bash
# 读 QuantTrio 235B AWQ 的 config.json（确认架构名与量化方式）
set -u
REPO=QuantTrio/Qwen3-VL-235B-A22B-Instruct-AWQ
timeout 40 curl -sL "https://modelscope.cn/models/$REPO/resolve/master/config.json" \
  -o /tmp/c.json 2>/dev/null
echo "  文件大小: $(stat -c%s /tmp/c.json 2>/dev/null || echo 0) B"
/root/rivermind-data/venv/bin/python - <<'PYEOF'
import json
try:
    d = json.load(open("/tmp/c.json", encoding="utf-8"))
except Exception as ex:
    print("  ★解析失败:", type(ex).__name__)
    print("  原始前 300 字符:", open("/tmp/c.json", encoding="utf-8",
                                    errors="replace").read()[:300])
    raise SystemExit
if not isinstance(d, dict) or "quantization_config" not in d and \
        "architectures" not in d:
    print("  （返回可能不是 config 本体，键：", list(d)[:10], "）")
print("  architectures    :", d.get("architectures"))
print("  model_type       :", d.get("model_type"))
q = d.get("quantization_config") or {}
print("  quant_method     :", q.get("quant_method"),
      "| bits:", q.get("bits"), "| group_size:", q.get("group_size"))
print("  tie_word_embeddings:", d.get("tie_word_embeddings"))
t = d.get("text_config") or {}
print("  text num_layers  :", t.get("num_hidden_layers"),
      "| text model_type:", t.get("model_type"))
v = d.get("vision_config") or {}
print("  vision depth     :", v.get("depth"), "| hidden:", v.get("hidden_size"))
PYEOF
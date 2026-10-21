#!/bin/bash
# 下载前验仓：QuantTrio/Qwen3-VL-235B-A22B-Instruct-AWQ 的文件清单、体积、config
# 不需要 GPU。目的：在花 130 GB 带宽与磁盘之前，确认仓库是完整的、且确实是 AWQ。
set -u
REPO=QuantTrio/Qwen3-VL-235B-A22B-Instruct-AWQ
V=/root/rivermind-data/venv/bin/python

echo "=== 1) 文件清单（ModelScope repo/files）==="
timeout 40 curl -s "https://modelscope.cn/api/v1/models/$REPO/repo/files?Revision=master&Root=" \
  -o /tmp/tree.json 2>/dev/null
"$V" - <<'PYEOF'
import json
try:
    d = json.load(open("/tmp/tree.json"))
except Exception as ex:
    print("  ★解析失败:", type(ex).__name__); raise SystemExit
files = d.get("Data", {}).get("Files") or []
if not files:
    print("  ★未取到文件列表，原始返回前 300 字符：")
    print(open("/tmp/tree.json", encoding="utf-8", errors="replace").read()[:300]); raise SystemExit
tot = 0
rows = []
for f in files:
    p = f.get("Path") or f.get("Name", "")
    s = f.get("Size") or 0
    if f.get("Type") == "tree":
        continue
    rows.append((p, s)); tot += s
big = [r for r in rows if r[1] > 200 * 1024 * 1024]
print(f"  文件 {len(rows)} 个，合计 {tot/1024**3:.2f} GB")
print(f"  >200MB 的 {len(big)} 个：")
for p, s in sorted(big, key=lambda x: -x[1])[:12]:
    print(f"    {s/1024**3:6.2f} GB  {p}")
key = [r for r in rows if r[0].split("/")[-1] in
       ("config.json", "model.safetensors.index.json", "preprocessor_config.json",
        "tokenizer_config.json", "generation_config.json")]
print("  关键配置：")
for p, s in key:
    print(f"    {'✓' if s > 0 else '★0字节'} {p}  ({s} B)")
PYEOF

echo
echo "=== 2) config.json 内容（确认是 AWQ + 架构）==="
timeout 40 curl -s "https://modelscope.cn/api/v1/models/$REPO/repo/files?Revision=master&FilePath=config.json" \
  -o /tmp/cfg.json 2>/dev/null
"$V" - <<'PYEOF'
import json
try:
    d = json.load(open("/tmp/cfg.json", encoding="utf-8"))
except Exception as ex:
    t = open("/tmp/cfg.json", encoding="utf-8", errors="replace").read()[:200]
    print("  直接解析失败，原始前 200 字符："); print("   ", repr(t)); raise SystemExit
print("  architectures   :", d.get("architectures"))
print("  model_type      :", d.get("model_type"))
qc = d.get("quantization_config") or {}
print("  quant_method    :", qc.get("quant_method"), "| bits:", qc.get("bits"),
      "| group:", qc.get("group_size"))
print("  num_hidden_layers:", d.get("num_hidden_layers") or d.get("text_config", {}).get("num_hidden_layers"))
vc = d.get("vision_config") or {}
print("  vision depth    :", vc.get("depth"))
PYEOF

echo
echo "=== 3) 磁盘现实检查 ==="
df -h /data /root/rivermind-data 2>/dev/null | sed 's/^/  /'
echo "  （AWQ 4-bit 约 120~130 GB ⇒ 本机 /data 仅 30 GB，装不下；需换多卡且大盘的机器）"

echo
echo "=== 4) 本机 GPU（用于确认"现在跑不了"）==="
ls /dev/nvidia[0-9]* 2>/dev/null | head -4 || echo "  ★无 GPU 设备节点"
echo "=== 结束 ==="
#!/bin/bash
# 下载前验证：ModelScope 上的 QuantTrio/Qwen3-VL-32B-Instruct-AWQ 是否是本任务可用的 AWQ 4bit 权重。
# 必须核对：① 体量放得下 49 GB ② config.json 声明 AWQ 量化 ③ architectures 是 Qwen3VLForConditionalGeneration
MID="QuantTrio/Qwen3-VL-32B-Instruct-AWQ"

echo "===== 1) 文件清单与体量 ====="
curl -s --max-time 30 "https://www.modelscope.cn/api/v1/models/$MID/repo/files?Revision=master&Recursive=1" -o /tmp/f.json
python3 - <<'PYEOF'
import json
d = json.load(open("/tmp/f.json"))
files = (d.get("Data") or {}).get("Files") or []
tot = sum(f.get("Size") or 0 for f in files)
print("  文件 " + str(len(files)) + " 个，总 " + format(tot/1024**3, ".2f") + " GB")
sf = [f for f in files if f.get("Path", "").endswith(".safetensors")]
print("  safetensors " + str(len(sf)) + " 个")
for f in sorted(files, key=lambda x: -(x.get("Size") or 0))[:3]:
    print("    " + f.get("Path", "?") + "  " + format((f.get("Size") or 0)/1024**3, ".2f") + " GB")
key = [f.get("Path") for f in files if f.get("Path") in
       ("config.json", "quantize_config.json", "preprocessor_config.json",
        "generation_config.json", "tokenizer_config.json")]
print("  关键配置文件: " + str(key))
PYEOF

echo
echo "===== 2) config.json 关键字段（架构 / 量化）====="
curl -sL --max-time 30 "https://www.modelscope.cn/models/$MID/resolve/master/config.json" -o /tmp/cfg.json
python3 - <<'PYEOF'
import json
try:
    c = json.load(open("/tmp/cfg.json"))
except Exception as e:
    print("  解析失败:", e); raise SystemExit
print("  architectures      = " + str(c.get("architectures")))
print("  model_type         = " + str(c.get("model_type")))
qc = c.get("quantization_config") or {}
print("  quantization_config= " + json.dumps(qc, ensure_ascii=False)[:200])
print("  torch_dtype        = " + str(c.get("torch_dtype")))
print("  num_hidden_layers  = " + str(c.get("num_hidden_layers")))
vision = c.get("vision_config") or {}
print("  vision depth       = " + str(vision.get("depth")))
PYEOF

echo
echo "===== 3) 量化配置/预处理器是否单独成文件 ====="
for f in quantize_config.json preprocessor_config.json; do
  code=$(curl -sL -o /tmp/x.json -w '%{http_code}' --max-time 20 \
    "https://www.modelscope.cn/models/$MID/resolve/master/$f")
  echo "  $f -> HTTP $code  大小 $(stat -c%s /tmp/x.json 2>/dev/null) 字节"
done
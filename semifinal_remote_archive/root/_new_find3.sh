#!/bin/bash
# 目标：在 ModelScope（实测 30 MB/s，比 hf-mirror 快 30 倍）上找到可用的量化 Qwen3-VL。
# 已知：Qwen/Qwen3-VL-32B-Instruct-AWQ 在 ModelScope 404；bf16 的 32B 约 65 GB，49 GB 盘放不下。
API=https://www.modelscope.cn/api/v1/models

chk() {
  code=$(curl -s -o /tmp/m.json -w '%{http_code}' --max-time 15 "$API/$1")
  if [ "$code" = "200" ]; then
    python3 - "$1" <<'PYEOF'
import json, sys
d = json.load(open("/tmp/m.json"))
dd = d.get("Data") or {}
print("  " + sys.argv[1].ljust(46) + "200  Id=" + str(dd.get("Id", "?")))
PYEOF
  else
    echo "  $1 -> $code"
  fi
}

echo "===== 1) 逐个体检可能的量化仓库 ID ====="
for r in \
  "QuantTrio/Qwen3-VL-32B-Instruct-AWQ" \
  "swift/Qwen3-VL-32B-Instruct-AWQ" \
  "AI-ModelScope/Qwen3-VL-32B-Instruct-AWQ" \
  "LLM-Research/Qwen3-VL-32B-Instruct-AWQ" \
  "kaitchup/Qwen3-VL-32B-Instruct-AWQ" \
  "cyankiwi/Qwen3-VL-32B-Instruct-AWQ" \
  "unsloth/Qwen3-VL-32B-Instruct-AWQ" \
  "cpatonn/Qwen3-VL-32B-Instruct-AWQ" \
  "Qwen/Qwen3-VL-30B-A3B-Instruct-AWQ" \
  "AI-ModelScope/Qwen3-VL-30B-A3B-Instruct-AWQ" \
  "Qwen/Qwen3-VL-32B-Instruct-GPTQ-Int4" \
  "Qwen/Qwen3-VL-8B-Instruct-AWQ" \
  ; do chk "$r"; done

echo
echo "===== 2) 试探搜索接口（打印原始响应的形状）====="
for u in \
  "https://www.modelscope.cn/api/v1/models?PageSize=20&PageNumber=1&Name=Qwen3-VL" \
  "https://www.modelscope.cn/api/v1/dolphin/models?PageSize=20&PageNumber=1&Name=Qwen3-VL" \
  ; do
  echo "--- $u"
  curl -s --max-time 20 "$u" | head -c 220; echo
done

echo
echo "===== 3) ModelScope 上 32B bf16 的真实体量（确认放不下）====="
curl -s --max-time 25 "https://www.modelscope.cn/api/v1/models/Qwen/Qwen3-VL-32B-Instruct/repo/files?Revision=master&Recursive=1" \
  -o /tmp/f.json
python3 - <<'PYEOF'
import json
try:
    d = json.load(open("/tmp/f.json"))
except Exception as e:
    print("  文件清单解析失败:", e); raise SystemExit
files = (d.get("Data") or {}).get("Files") or []
tot = sum(f.get("Size") or 0 for f in files)
print("  文件 " + str(len(files)) + " 个，总 " + format(tot/1024**3, ".2f") + " GB")
for f in sorted(files, key=lambda x: -(x.get("Size") or 0))[:4]:
    print("    " + f.get("Path", "?") + "  " + format((f.get("Size") or 0)/1024**3, ".2f") + " GB")
PYEOF
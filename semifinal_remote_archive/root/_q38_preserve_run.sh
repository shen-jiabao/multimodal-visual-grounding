#!/bin/bash
# 分离执行的 G1 复核（结果落日志，避免本地 SSH 读超时）
# 用法：远端 setsid nohup bash 本脚本 > /dev/null 2>&1 &
set -u
R=/root/rivermind-data
D=$R/models/Qwen3.8-27B
V=$R/venv_q38
MODEL=Qwen/Qwen3.8-27B
OUT=$R/_q38_preserve.log

: > "$OUT"
{
echo "=== ① GPU ==="
if ls /dev/nvidia[0-9]* >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader
  NG=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
  MX=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | sort -n | tail -1)
  echo "GPU 数 $NG｜单卡最大 $(( MX / 1024 )) GB"
  if [ "$NG" -ge 2 ]; then echo "⇒ TP=2"; elif [ "$MX" -ge 75000 ]; then echo "⇒ TP=1"; else echo "★显存不足"; fi
else
  echo "✗ 无 /dev/nvidia*"
fi

echo
echo "=== ② G1 逐片 sha256（对官方 manifest）==="
curl -sL --max-time 90 "https://hf-mirror.com/api/models/$MODEL?blobs=true" -o /tmp/_q38v.json 2>/dev/null
python3 - <<'PY' > /tmp/_q38_man.tsv 2>/dev/null
import json
try:
    d = json.load(open("/tmp/_q38v.json", encoding="utf-8"))
except Exception:
    raise SystemExit(0)
for s in d.get("siblings") or []:
    n = s["rfilename"]
    if not n.endswith(".safetensors"): continue
    sz = s.get("size") or (s.get("lfs") or {}).get("size") or 0
    sha = (s.get("lfs") or {}).get("sha256") or ""
    print(f"{n}\t{sz}\t{sha}")
PY
echo "manifest $(grep -c . /tmp/_q38_man.tsv) 片"
ok=0; bad=0
while IFS=$'\t' read -r name size sha; do
  [ -z "$name" ] && continue
  f="$D/$name"
  if [ ! -f "$f" ]; then echo "✗ 缺 $name"; bad=$((bad+1)); continue; fi
  if [ "$(stat -c%s "$f")" != "$size" ]; then echo "✗ 大小 $name"; bad=$((bad+1)); continue; fi
  g=$(sha256sum "$f" | awk '{print $1}')
  if [ "$g" = "$sha" ]; then ok=$((ok+1)); echo "✓ $name"; else echo "✗ sha $name"; bad=$((bad+1)); fi
done < /tmp/_q38_man.tsv
echo "⇒ 通过 $ok ｜失败 $bad"

echo
echo "=== ③ index.json 覆盖性 ==="
$V/bin/python - "$D" <<'PY'
import json, os, glob, sys
d = sys.argv[1]
idx = os.path.join(d, "model.safetensors.index.json")
if not os.path.exists(idx):
    print("★无 index.json"); raise SystemExit(0)
m = json.load(open(idx, encoding="utf-8"))["weight_map"]
sh = sorted(set(m.values()))
on = sorted(os.path.basename(p) for p in glob.glob(os.path.join(d, "model-*.safetensors")))
print(f"权重条目 {len(m)}｜引用 {len(sh)}｜盘上 {len(on)}｜缺失 {len([s for s in sh if s not in on])}")
PY

echo
echo "=== ④ 环境 ==="
echo "vllm $($V/bin/python -c 'import vllm;print(vllm.__version__)' 2>/dev/null)"
echo "torch $($V/bin/python -c 'import torch;print(torch.__version__)' 2>/dev/null)"
$V/bin/python - <<'PY'
try:
    from vllm.model_executor.models.registry import ModelRegistry
    print("架构登记 Qwen3_5ForConditionalGeneration:", "Qwen3_5ForConditionalGeneration" in ModelRegistry.get_supported_archs())
except Exception as e:
    print("✗", e)
PY

echo
echo "=== ⑤ 磁盘 ==="
df -h $R | tail -1
echo "=== DONE ==="
} >> "$OUT" 2>&1
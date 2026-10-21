#!/bin/bash
# 异质模型 G1：逐片 sha256 独立复核（不依赖下载器自己的账）
set -u
R=/root/rivermind-data
D=$R/models/Qwen2.5-VL-72B-Instruct-AWQ
MODEL=Qwen/Qwen2.5-VL-72B-Instruct-AWQ

echo "=== 0) 目录 ==="
du -sh "$D" | sed 's/^/  /'
df -h $R | tail -1 | sed 's/^/  /'
ls -1 "$D" | sed 's/^/  /'

echo; echo "=== 1) 官方 manifest 重新拉取（独立于下载时的缓存）==="
curl -sL --max-time 90 "https://hf-mirror.com/api/models/$MODEL?blobs=true" -o /tmp/_q285v.json 2>/dev/null
python3 - <<'PY' > /tmp/_q285_manifest.tsv
import json
d = json.load(open("/tmp/_q285v.json", encoding="utf-8"))
for s in d.get("siblings") or []:
    n = s["rfilename"]
    sz = s.get("size") or (s.get("lfs") or {}).get("size") or 0
    sha = (s.get("lfs") or {}).get("sha256") or ""
    print(f"{n}\t{sz}\t{sha}")
PY
echo "  manifest $(grep -c . /tmp/_q285_manifest.tsv) 项"

echo; echo "=== 2) 逐文件校验（大小 + sha256；无 sha256 的只验大小）==="
ok=0; bad=0; nsha=0
while IFS=$'\t' read -r name size sha; do
  [ -z "$name" ] && continue
  f="$D/$name"
  if [ ! -f "$f" ]; then echo "  ✗ 缺失 $name"; bad=$((bad+1)); continue; fi
  s=$(stat -c%s "$f")
  if [ "$s" != "$size" ]; then echo "  ✗ 大小不符 $name ($s≠$size)"; bad=$((bad+1)); continue; fi
  if [ -n "$sha" ]; then
    nsha=$((nsha+1))
    g=$(sha256sum "$f" | awk '{print $1}')
    if [ "$g" = "$sha" ]; then ok=$((ok+1));
    else echo "  ✗ sha256 不符 $name"; bad=$((bad+1)); fi
  else
    ok=$((ok+1))
  fi
done < /tmp/_q285_manifest.tsv
echo "  ⇒ 通过 $ok｜失败 $bad（其中 $nsha 项有官方 sha256）"

echo; echo "=== 3) index.json 覆盖性 ==="
$R/venv/bin/python - "$D" <<'PY'
import json, os, glob, sys
d = sys.argv[1]
idx = os.path.join(d, "model.safetensors.index.json")
m = json.load(open(idx, encoding="utf-8"))["weight_map"]
sh = sorted(set(m.values()))
on = sorted(os.path.basename(p) for p in glob.glob(os.path.join(d, "*.safetensors")))
print(f"  权重条目 {len(m)}｜引用分片 {len(sh)}｜盘上 {len(on)}")
print(f"  缺失 {[s for s in sh if s not in on][:3]}｜多余 {[s for s in on if s not in sh][:3]}")
cfg = json.load(open(os.path.join(d, "config.json"), encoding="utf-8"))
print(f"  architectures={cfg.get('architectures')}")
print(f"  quantization_config={cfg.get('quantization_config')}")
PY

echo; echo "=== 4) GPU ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>&1 | sed 's/^/  /'
echo "=== 结束 ==="

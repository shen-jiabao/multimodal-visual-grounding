#!/bin/bash
# bf16 权重独立复核（第二遍，与 monitor 当时的验证相互独立）
#   ① 14 个分片逐个算 sha256，与官方 manifest 比对
#   ② index.json 覆盖性：14 个分片是否全被引用、是否有缺号
#   ③ 资源余量：磁盘 / 内存（为后续起服做准备）
set -u
R=/root/rivermind-data
D=$R/models/Qwen3-VL-32B-Instruct
MODEL=Qwen/Qwen3-VL-32B-Instruct

echo "=== ① 官方 manifest ==="
curl -sL --max-time 90 "https://hf-mirror.com/api/models/$MODEL?blobs=true" -o /tmp/_v.json 2>/dev/null
python3 - <<'PY' > /tmp/_vm.tsv 2>/dev/null
import json
d = json.load(open("/tmp/_v.json", encoding="utf-8"))
for s in d.get("siblings") or []:
    n = s["rfilename"]
    if not n.endswith(".safetensors"): continue
    sz = s.get("size") or (s.get("lfs") or {}).get("size") or 0
    sha = (s.get("lfs") or {}).get("sha256") or ""
    print(f"{n}\t{sz}\t{sha}")
PY
echo "  manifest 分片 $(grep -c . /tmp/_vm.tsv) 个"

echo; echo "=== ② 逐片 sha256 独立复算 ==="
ok=0; bad=0
while IFS=$'\t' read -r name size sha; do
  [ -z "$name" ] && continue
  f="$D/$name"
  if [ ! -f "$f" ]; then echo "  ✗ 缺失 $name"; bad=$((bad+1)); continue; fi
  s=$(stat -c%s "$f")
  if [ "$s" != "$size" ]; then echo "  ✗ 大小 $name  $s≠$size"; bad=$((bad+1)); continue; fi
  g=$(sha256sum "$f" | awk '{print $1}')
  if [ "$g" = "$sha" ]; then
    ok=$((ok+1)); echo "  ✓ $name  $(( s / 1048576 )) MB"
  else
    bad=$((bad+1)); echo "  ✗ sha256 $name"; echo "      得 $g"; echo "      期 $sha"
  fi
done < /tmp/_vm.tsv
echo "  ⇒ 通过 $ok / 失败 $bad"

echo; echo "=== ③ index.json 覆盖性 ==="
python3 - <<'PY'
import json, os, glob
D = "/root/rivermind-data/models/Qwen3-VL-32B-Instruct"
idx = json.load(open(os.path.join(D, "model.safetensors.index.json"), encoding="utf-8"))
wm = idx.get("weight_map") or {}
shards = sorted({v for v in wm.values()})
print(f"  权重条目 {len(wm)}；引用分片 {len(shards)}")
on_disk = sorted(os.path.basename(p) for p in glob.glob(os.path.join(D, "model-*.safetensors")))
print(f"  盘上分片 {len(on_disk)}")
miss = [s for s in shards if s not in on_disk]
extra = [s for s in on_disk if s not in shards]
print(f"  缺失 {len(miss)} {miss[:3]}；多余 {len(extra)} {extra[:3]}")
print(f"  dtype={idx.get('metadata',{}).get('total_size',0)/1073741824:.1f} GB 声明总量"
      if idx.get("metadata",{}).get("total_size") else "  （无 total_size 元数据）")
PY

echo; echo "=== ④ 起服所需资源 ==="
echo "  内存："; free -g | sed 's/^/    /'
echo "  磁盘："; df -h "$R" | tail -1 | sed 's/^/    /'
echo "  nvidia-smi："; nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>&1 | head -4 | sed 's/^/    /'
echo "  （无输出即无 GPU）"

echo; echo "=== ⑤ vLLM 环境 ==="
for v in venv_vllm venv_vlx venv; do
  p="$R/$v"
  if [ -d "$p" ]; then
    ver=$("$p/bin/python" -c "import vllm;print(vllm.__version__)" 2>/dev/null || echo "无 vllm")
    echo "  ● $v 在（vllm: $ver）"
  else
    echo "  ○ $v 不存在"
  fi
done
pip_bin=$R/venv_vllm/bin/pip
[ -x "$pip_bin" ] && echo "  vllm 依赖数 $("$pip_bin" list 2>/dev/null | wc -l)"
echo "=== 结束 ==="
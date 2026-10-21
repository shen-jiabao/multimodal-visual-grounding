#!/bin/bash
# 目的：① 校准 HF 镜像「401 到底是不存在还是受限」② 拿到真正可用仓库的文件清单与体量
#       ③ 用**确认存在**的大文件实测下载速率（前两次测速都打到了错误页）
API=https://hf-mirror.com/api/models

probe() {  # $1=仓库ID
  code=$(curl -s -o /tmp/r.json -w '%{http_code}' --max-time 20 "$API/$1?blobs=true")
  echo "--- $1 -> HTTP $code"
  if [ "$code" = "200" ]; then
    python3 - <<'PYEOF'
import json
d = json.load(open("/tmp/r.json"))
sib = d.get("siblings") or []
tot = sum(s.get("size") or 0 for s in sib)
saf = [s for s in sib if s["rfilename"].endswith(".safetensors")]
print("    文件 " + str(len(sib)) + " 个，总 " + format(tot/1024**3, ".2f") + " GB"
      + "，" + str(len(saf)) + " 个 safetensors")
big = sorted(sib, key=lambda s: -(s.get("size") or 0))[:3]
for s in big:
    print("      " + s["rfilename"] + "  " + format((s.get("size") or 0)/1024**3, ".2f") + " GB")
print("      config/vision 关键文件: " + str([s["rfilename"] for s in sib
      if s["rfilename"] in ("config.json", "preprocessor_config.json",
                            "quantize_config.json", "generation_config.json")]))
PYEOF
  fi
}

echo "===== 1) 校准：一个肯定不存在的仓库返回什么 ====="
probe "Qwen/ThisRepoDefinitelyDoesNotExist_xyz123"

echo
echo "===== 2) 确认存在的仓库清单与体量 ====="
probe "QuantTrio/Qwen3-VL-32B-Instruct-AWQ"
probe "Qwen/Qwen3-VL-8B-Instruct"

echo
echo "===== 3) 用确认存在的大文件实测速率（QuantTrio 首个分片，限 25 秒）====="
FIRST=$(python3 -c "
import json,subprocess
subprocess.run(['curl','-s','-o','/tmp/q.json','--max-time','20','$API/QuantTrio/Qwen3-VL-32B-Instruct-AWQ?blobs=true'])
d=json.load(open('/tmp/q.json'))
s=[x for x in (d.get('siblings') or []) if x['rfilename'].endswith('.safetensors')]
print(s[0]['rfilename'] if s else '')
" 2>/dev/null)
echo "  首个分片: $FIRST"
if [ -n "$FIRST" ]; then
  curl -sL --max-time 25 -r 0-400000000 -o /dev/null \
    -w '  速率 %{speed_download} B/s  已下 %{size_download} 字节  HTTP %{http_code}\n' \
    "https://hf-mirror.com/QuantTrio/Qwen3-VL-32B-Instruct-AWQ/resolve/main/$FIRST"
fi

echo
echo "===== 4) 同样方式测 ModelScope 的 8B（作对照，确认可下）====="
curl -sL --max-time 25 -r 0-400000000 -o /dev/null \
  -w '  速率 %{speed_download} B/s  已下 %{size_download} 字节  HTTP %{http_code}\n' \
  "https://www.modelscope.cn/models/Qwen/Qwen3-VL-8B-Instruct/resolve/master/model-00001-of-00004.safetensors"

echo
echo "===== 5) 磁盘 ====="
df -h /root/rivermind-data | tail -1
#!/bin/bash
# 网络带宽实测：① pip 真实吞吐 ② 四家 pypi 镜像 ③ hf-mirror ④ modelscope
# 目的：pip 只有 0.8 MB/s，且后续还有 52 GB 权重要走网 ⇒ 必须知道哪个通道快。
set -u
. /root/rivermind-data/_q38_env.sh
R=/root/rivermind-data

bytes_of() {  # pip 在飞的下载总量 = tmp + pipcache
  { du -sb "$R/tmp" 2>/dev/null | cut -f1; du -sb "$R/pipcache" 2>/dev/null | cut -f1; } \
    | awk '{s+=$1} END{print s+0}'
}

echo "=== ① 当前 pip 实测吞吐（采样 20 秒）==="
b1=$(bytes_of); sleep 20; b2=$(bytes_of)
awk -v a="$b1" -v b="$b2" 'BEGIN{
  d=b-a; printf "  20 秒内增长 %.2f MB ⇒ 实测 %.2f MB/s\n", d/1048576, d/1048576/20
}'
echo "  （接近 0 = 正在解包/装包，不是在下；这里是非 0 值即为真实下载速率）"

echo
echo "=== ② 各 pypi 源（torch 大 wheel，15 秒窗口）==="
# ★PyPI simple 索引的 href 形如 ../../packages/xx/torch-...whl#sha256=abc：
#   ① '#' 片段会让 endswith('.whl') 失效 ② wheel tag 是 manylinux_2_28_x86_64，
#   不含 'linux_x86_64'。两处都按实测修正。
PKG=torch
for m in "https://mirrors.aliyun.com/pypi/simple" \
         "https://pypi.tuna.tsinghua.edu.cn/simple" \
         "https://mirrors.ustc.edu.cn/pypi/simple" \
         "https://mirrors.cloud.tencent.com/pypi/simple" \
         "https://repo.huaweicloud.com/repository/pypi/simple" \
         "https://pypi.org/simple"; do
  host=$(echo "$m" | awk -F/ '{print $3}')
  u=$(curl -s --max-time 20 "$m/$PKG/" 2>/dev/null | \
      /root/miniconda3/bin/python3 -c "
import sys, re
from urllib.parse import urljoin
html = sys.stdin.read()
base = '$m/$PKG/'
urls = [urljoin(base, h) for h in re.findall(r'href=\"([^\"]+)\"', html)]
urls = [x.split('#')[0] for x in urls]
cand = [x for x in urls if x.endswith('.whl') and 'cp312' in x and 'x86_64' in x]
print(cand[-1] if cand else '')
")
  if [ -z "$u" ]; then printf "  %-38s ★ 索引取不到链接\n" "$host"; continue; fi
  curl -s -o /dev/null --max-time 15 -w "  $host|%{speed_download}|%{size_download}|%{time_total}\n" "$u" 2>/dev/null \
    | awk -F'|' '{printf "%-40s %6.2f MB/s  15s 内 %7.2f MB（用时 %ss）\n", $1, $2/1048576, $3/1048576, $4}'
done

echo
echo "=== ③ hf-mirror（权重主通道，取 Qwen2.5-0.5B 的 model.safetensors，15 秒窗口）==="
HFU="https://hf-mirror.com/Qwen/Qwen2.5-0.5B-Instruct/resolve/main/model.safetensors"
curl -sL -o /dev/null --max-time 15 -w "  hf-mirror.com|%{speed_download}|%{size_download}|%{time_total}|%{http_code}\n" "$HFU" 2>/dev/null \
  | awk -F'|' '{printf "%-40s %6.2f MB/s  15s 内 %7.2f MB  http=%s\n", $1, $2/1048576, $3/1048576, $5}'

echo
echo "=== ④ modelscope（备选，同文件）==="
MSU="https://modelscope.cn/api/v1/models/Qwen/Qwen2.5-0.5B-Instruct/repo?Revision=master&FilePath=model.safetensors"
curl -sL -o /dev/null --max-time 15 -w "  modelscope.cn|%{speed_download}|%{size_download}|%{time_total}|%{http_code}\n" "$MSU" 2>/dev/null \
  | awk -F'|' '{printf "%-40s %6.2f MB/s  15s 内 %7.2f MB  http=%s\n", $1, $2/1048576, $3/1048576, $5}'

echo
echo "=== ⑤ ModelScope 上是否有 Qwen3.8-27B（权重通道备选）==="
for repo in "Qwen/Qwen3.8-27B" "Qwen/Qwen3.8-27B-Instruct"; do
  j=$(curl -s --max-time 25 "https://modelscope.cn/api/v1/models/$repo/repo/files?Revision=master&Recursive=true")
  echo "$j" | /root/miniconda3/bin/python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
except Exception as e:
    print('  $repo  解析失败: %s' % str(e)[:60]); sys.exit()
files = (d.get('Data') or {}).get('Files')
if not isinstance(files, list):
    print('  $repo  ★ 不存在或未开放（resp=%s）' % str(d)[:80]); sys.exit()
st = [f for f in files if str(f.get('Path','')).endswith('.safetensors')]
tot = sum(int(f.get('Size', 0) or 0) for f in st)
print('  $repo  文件 %d 个，safetensors %d 个，合计 %.2f GiB' % (len(files), len(st), tot/1073741824))
"
done

echo
echo "=== ⑥ 构建日志尾 + 磁盘 ==="
tail -3 "$R/q38_env.log" | sed 's/^/  /'
df -h "$R" | tail -1 | sed 's/^/  /'
echo "=== DONE ==="

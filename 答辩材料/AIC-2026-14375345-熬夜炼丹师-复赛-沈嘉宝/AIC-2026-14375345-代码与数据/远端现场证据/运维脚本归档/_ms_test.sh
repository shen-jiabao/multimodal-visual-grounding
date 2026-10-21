#!/bin/bash
# 测 ModelScope 是否支持 Range 断点下载（决定能否直接套用已验证的并发分片下载器）
set -u
. /root/rivermind-data/_q38_env.sh
M=Qwen/Qwen3.8-27B

echo "=== ① 从 ModelScope API 取一个真实分片名 ==="
F=$(curl -s --max-time 25 "https://modelscope.cn/api/v1/models/$M/repo/files?Revision=master&Recursive=true" | \
    /root/miniconda3/bin/python3 -c "
import sys, json
d = json.load(sys.stdin)
files = (d.get('Data') or {}).get('Files') or []
st = [f['Path'] for f in files if str(f.get('Path','')).endswith('.safetensors')]
print(st[0] if st else '')
")
echo "  分片样例: ${F:-（取不到）}"
[ -z "$F" ] && exit 1

echo
echo "=== ② 两种 URL 形态的 Range 支持与速率（各拉 1 MB 试 206） ==="
U1="https://modelscope.cn/models/$M/resolve/master/$F"
U2="https://modelscope.cn/api/v1/models/$M/repo?Revision=master&FilePath=$F"
for tag in "resolve路径|$U1" "repo接口|$U2"; do
  name=${tag%%|*}; u=${tag#*|}
  echo "  --- $name ---"
  echo "    $u"
  out=$(curl -sL -o /dev/null -r 0-1048575 --max-time 20 -w "%{http_code}|%{size_download}|%{speed_download}" "$u" 2>/dev/null)
  echo "$out" | awk -F'|' '{printf "    Range(0-1MB): http=%s  实收 %.3f MB  %.2f MB/s\n", $1, $2/1048576, $3/1048576}'
  echo "    响应头关键字段:"
  curl -sIL --max-time 20 "$u" 2>/dev/null | grep -iE "^(accept-ranges|content-length|content-range|content-type)" | head -4 | sed 's/^/      /'
done

echo
echo "=== ③ 无 Range 时的整片速率（10 秒窗口）==="
curl -sL -o /dev/null --max-time 10 -w "  %{speed_download}|%{size_download}|%{http_code}\n" "$U1" 2>/dev/null \
  | awk -F'|' '{printf "  %.2f MB/s  10s 内 %.2f MB  http=%s\n", $1/1048576, $2/1048576, $3}'
echo "=== DONE ==="

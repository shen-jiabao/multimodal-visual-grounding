#!/bin/bash
# 查看建环境（_q38_build.sh）进度。
set -u
R=/root/rivermind-data

if pgrep -f _q38_build.sh >/dev/null 2>&1; then
  echo "状态: 运行中  pid $(pgrep -f _q38_build.sh | tr '\n' ' ')"
else
  echo "状态: 已结束"
fi

echo
echo "--- _q38_build.out（末 14 行）---"
tail -14 "$R/_q38_build.out" 2>/dev/null | sed 's/^/  /'

if [ -f "$R/q38_env.log" ]; then
  echo
  echo "--- q38_env.log（末 14 行，pip 详细输出在这里）---"
  tail -14 "$R/q38_env.log" 2>/dev/null | sed 's/^/  /'
fi

echo
echo "--- venv 体量 ---"
du -sh "$R/venv_q38" 2>/dev/null | sed 's/^/  /' || echo "  （尚未生成）"

# 真实下载吞吐：只统计装包器的缓存与临时目录（不含 dataset、不受上传干扰）
if pgrep -f "_q38_build.sh" >/dev/null 2>&1; then
  pip_bytes() {
    { du -sb "$R/tmp" 2>/dev/null | cut -f1
      du -sb "$R/pipcache" 2>/dev/null | cut -f1
      du -sb "$R/uvcache" 2>/dev/null | cut -f1
    } | awk '{s+=$1} END{print s+0}'
  }
  b1=$(pip_bytes); sleep 15; b2=$(pip_bytes)
  awk -v a="$b1" -v b="$b2" 'BEGIN{
    d=b-a; printf "--- 装包器实测吞吐（15 秒）---\n  %.2f MB ⇒ %.2f MB/s\n", d/1048576, d/1048576/15
  }'
  echo "--- uvcache 体量 ---"
  du -sh "$R/uvcache" 2>/dev/null | sed 's/^/  /'
fi

echo "--- 磁盘 ---"
df -h "$R" | tail -1 | sed 's/^/  /'

# 数据上传进度：三个模态目录各有多少文件、合计体量（与 _imglist.txt 的 1295 名对照）
echo
echo "--- 数据上传进度（目标 1295 名 × 3 模态 = 3885 文件）---"
D=$R/dataset/Images
tot=0
for m in visible infrared depth; do
  n=$(ls -1 "$D/$m" 2>/dev/null | wc -l)
  s=$(du -sh "$D/$m" 2>/dev/null | cut -f1)
  echo "  $m: $n 文件 / ${s:-0}"
  tot=$((tot + n))
done
echo "  合计 $tot / 3885"

# 若客户端在飞，报一下它的最近日志（job 目录由 --sync 写入）
if pgrep -f "_rc.py" >/dev/null 2>&1; then
  echo "  ★ 上传客户端进程在运行: $(pgrep -f _rc.py | tr '\n' ' ')"
fi
echo "=== DONE ==="

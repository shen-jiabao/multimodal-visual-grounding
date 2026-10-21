#!/usr/bin/env bash
# 只读取证：本实例时间线（不修改、不删除任何文件）
# 用法：SSH_PW=... python -B _remote_run_file.py _srv_timeline.sh /root/_srv_timeline.sh
set +e
R=/root/autodl-tmp/rivermind-data

echo "===== 0) 主机 / 时间 / 开机时长 ====="
hostname
date; date -u
uptime -s 2>/dev/null || cat /proc/uptime
echo "--- inode 挂载 ---"
df -h / /root/autodl-tmp 2>/dev/null

echo
echo "===== 1) /root 顶层 ====="
ls -la --time-style=full-iso /root
echo "--- .bash_history ---"
ls -la /root/.bash_history 2>&1
tail -n 200 /root/.bash_history 2>/dev/null

echo
echo "===== 2) /root/autodl-tmp 顶层 ====="
ls -la --time-style=full-iso /root/autodl-tmp

echo
echo "===== 3) rivermind-data 顶层 ====="
ls -la --time-style=full-iso "$R"
echo "--- du ---"
du -sh "$R"/* 2>/dev/null

echo
echo "===== 4) output 产物（mtime/ctime）====="
ls -la --time-style=full-iso "$R/output" 2>/dev/null
echo "--- md5 ---"
md5sum "$R"/output/*.json 2>/dev/null

echo
echo "===== 5) dataset ====="
ls -la --time-style=full-iso "$R/dataset" 2>/dev/null
for m in visible infrared depth; do
  n=$(ls "$R/dataset/Images/$m" 2>/dev/null | wc -l)
  echo "  $m: $n"
done
md5sum "$R/dataset/queries/queries.json" 2>/dev/null
ls -la --time-style=full-iso "$R/dataset/queries" 2>/dev/null

echo
echo "===== 6) models / venv ====="
ls -la --time-style=full-iso "$R/models" 2>/dev/null
stat -c '%n | c=%z | m=%y' "$R/models/Qwen3.8-27B" 2>/dev/null
stat -c '%n | c=%z | m=%y' "$R/venv_q38" 2>/dev/null
ls -la --time-style=full-iso "$R/venv_q38/bin" 2>/dev/null | head -6

echo
echo "===== 7) 全树时间戳（排除大目录）====="
find "$R" \
  \( -path "$R/models" -o -path "$R/venv_q38" -o -path "$R/uvcache" \
     -o -path "$R/dataset/Images" -o -path "$R/hf" -o -path "$R/cache" \
     -o -path "$R/tmp" -o -path "$R/__pycache__" \) -prune -o \
  -type f -printf '%TY-%Tm-%Td %TH:%TM:%TS | c=%CY-%Cm-%Cd %CH:%CM:%CS | %10s | %p\n' 2>/dev/null | sort

echo
echo "===== 8) 全盘找 code27b / consensus / predictions ====="
find / -xdev -maxdepth 8 \
  \( -path /proc -o -path /sys -o -path /root/miniconda3 \
     -o -path "$R/uvcache" -o -path "$R/venv_q38" -o -path "$R/models" \) -prune -o \
  \( -name 'code27b*' -o -name 'consensus_vote.py' -o -name 'make_sub.py' \
     -o -name 'predictions_27b*' -o -name 'consensus_pool*' -o -name 'infer.py' \) -print 2>/dev/null \
  | head -40

echo
echo "===== 9) 压缩包 / git 仓库 ====="
find /root /autodl-tmp -maxdepth 5 \
  \( -path "$R/uvcache" -o -path "$R/venv_q38" \) -prune -o \
  \( -name '*.tar.gz' -o -name '*.zip' -o -name '*.tar' -o -name '.git' \) \
  -printf '%TY-%Tm-%Td %TH:%TM | %10s | %p\n' 2>/dev/null | sort | tail -40

echo
echo "===== 10) /root 下近期上传的脚本 ====="
ls -la --time-style=full-iso /root/*.sh /root/*.py 2>/dev/null

echo
echo "===== DONE ====="

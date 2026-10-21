#!/bin/bash
# 只读取证：code27b 服务器侧时间线（不修改、不删除、不写入任何文件）
# 用法：SSH_PW=... python -B _remote_run_file.py _s27b_timeline.sh /root/_s27b_timeline.sh
set +e
W=/root/autodl-tmp/work/code27b

echo "===== 0) 主机 / 时间 / 时区 ====="
hostname 2>/dev/null
date; date -u
cat /etc/timezone 2>/dev/null
timedatectl 2>/dev/null | head -5

echo "===== 1) code27b 是否存在 / 顶层（含 ctime）====="
ls -la --time-style=full-iso "$W" 2>/dev/null || echo "NO_DIR $W"
stat -c 'dir %n%n  ctime=%z%n  mtime=%y%n  birth=%w' "$W" 2>/dev/null

echo "===== 2) 全树时间戳 (mtime | ctime | atime | size | path) ====="
find "$W" -path "$W/outputs" -prune -o -type f \
  -printf '%TY-%Tm-%Td %TH:%TM:%TS | c=%CY-%Cm-%Cd %CH:%CM:%CS | a=%AY-%Am-%Ad %AH:%AM:%AS | %10s | %p\n' \
  2>/dev/null | sort

echo "===== 3) 关键文件 stat（含 birth/inode）====="
for f in README.md requirements.txt make_sub.py src/config.py src/models/qwen3_vl.py \
         src/models/qwen3_vl_parse.py src/inference/matcher.py src/inference/detect_select.py \
         src/inference/chains.py src/inference/predictor.py src/data/preprocess.py \
         src/data/fusion.py src/data/loader.py src/utils/text.py src/eval/metrics.py \
         src/postprocess/search.py scripts/infer.py scripts/consensus_vote.py \
         scripts/pool_dimensions.py configs/server_32b.yaml; do
  stat -c '%n | m=%y | c=%z | birth=%w' "$W/$f" 2>/dev/null
done

echo "===== 4) 是否 git 仓库 / 提交历史 ====="
which git 2>/dev/null
ls -la "$W/.git" 2>/dev/null | head -20
git -C "$W" log --format='%H | %ad | %an <%ae> | %s' --date=iso 2>&1 | head -50
git -C "$W" status -s 2>&1 | head -20

echo "===== 5) outputs / 运行产物 ====="
ls -la --time-style=full-iso "$W/outputs" 2>/dev/null | head -40

echo "===== 6) 上级与同级目录（是否还有别的包 / 别的队）====="
ls -la --time-style=full-iso /root/autodl-tmp/work 2>/dev/null
ls -la --time-style=full-iso /root/autodl-tmp 2>/dev/null | head -40
ls -la --time-style=full-iso /root 2>/dev/null | head -40

echo "===== 7) shell 历史（命令序列证据）====="
wc -l /root/.bash_history 2>/dev/null
tail -n 300 /root/.bash_history 2>/dev/null

echo "===== 8) 上传 / 下载 / 解包痕迹 ====="
find /root -maxdepth 3 \( -name '*.tar.gz' -o -name '*.tgz' -o -name '*.tar' -o -name '*.zip' \) \
  -printf '%TY-%Tm-%Td %TH:%TM:%TS | %10s | %p\n' 2>/dev/null | sort | tail -60

echo "===== 9) 模型目录时间 ====="
ls -la --time-style=full-iso /root/autodl-tmp/models 2>/dev/null | head -20
stat -c '%n | c=%z | m=%y' /root/autodl-tmp/models/Qwen3.8-27B 2>/dev/null

echo "===== 10) 全盘同名副本扫描 ====="
find / -maxdepth 6 -type d -name 'code27b' 2>/dev/null | head -20

echo "===== DONE ====="

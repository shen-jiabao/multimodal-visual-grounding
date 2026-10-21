#!/bin/bash
# 新机一次性准备（无卡模式下即可完成，最省计费）。
#   ① /root/rivermind-data 符号链接到数据盘（让 100 个脚本一字不改可跑）
#   ② 缓存路径全部指向数据盘（系统盘只有 30 G）
#   ③ pip 走阿里镜像（直连 pypi 12s vs 镜像 4.5s）
#   ④ HF 走 hf-mirror（直连不通；镜像 0.36s）
# 幂等：重复执行不会重复追加。
set -u

NEW_ROOT=/root/autodl-tmp
LINK=/root/rivermind-data
MARK="# ==== 2026AiC 新机准备 ===="

echo "=== ① 路径就位 ==="
mkdir -p "$NEW_ROOT/rivermind-data"
if [ -L "$LINK" ]; then
  echo "  已是符号链接: $(readlink "$LINK")"
elif [ -e "$LINK" ]; then
  echo "  ★ $LINK 已存在且**不是**符号链接，为安全起见停手。"
  echo "    请先确认其内容（ls -la $LINK）再决定。"
  exit 9
else
  ln -s "$NEW_ROOT/rivermind-data" "$LINK"
  echo "  ✓ 新建 $LINK → $NEW_ROOT/rivermind-data"
fi

echo
echo "=== ② 目录骨架 + 缓存路径 ==="
mkdir -p "$LINK"/{hf,pipcache,tmp,cache,models,output,dataset}
df -h "$LINK" | tail -1 | sed 's/^/  /'
if ! grep -qF "$MARK" ~/.bashrc 2>/dev/null; then
  cat >> ~/.bashrc <<EOF

$MARK
export PATH=/usr/local/cuda/bin:\$PATH
export HF_HOME=$LINK/hf
export HF_HUB_CACHE=$LINK/hf/hub
export PIP_CACHE_DIR=$LINK/pipcache
export TMPDIR=$LINK/tmp
export XDG_CACHE_HOME=$LINK/cache
export HF_ENDPOINT=https://hf-mirror.com
EOF
  echo "  ✓ 已写入 ~/.bashrc（PATH 含 cuda/bin；四个缓存变量；HF 镜像）"
else
  echo "  已写过，跳过"
fi

echo
echo "=== ③ pip 阿里镜像 ==="
mkdir -p ~/.pip
cat > ~/.pip/pip.conf <<'EOF'
[global]
index-url = https://mirrors.aliyun.com/pypi/simple/
trusted-host = mirrors.aliyun.com
timeout = 60
EOF
echo "  ✓ ~/.pip/pip.conf"
cat ~/.pip/pip.conf | sed 's/^/    /'

echo
echo "=== ④ 生效并复核 ==="
export PATH=/usr/local/cuda/bin:$PATH
export HF_ENDPOINT=https://hf-mirror.com
echo "  nvcc      : $(command -v nvcc)  $(nvcc --version 2>/dev/null | grep -o 'release [0-9.]*')"
echo "  python3   : $(command -v python3 || echo '★不在PATH')"
echo "  R 解析    : $(readlink -f /root/rivermind-data)"
echo "  R 所在盘  : $(df -h /root/rivermind-data | tail -1 | awk '{print $1, $2, "可用", $4}')"
echo "  HF_ENDPOINT=$HF_ENDPOINT"
echo "  pip 源    : $(/root/miniconda3/bin/python3 -m pip config get global.index-url 2>/dev/null || tail -2 ~/.pip/pip.conf | head -1)"

echo
echo "=== DONE ==="

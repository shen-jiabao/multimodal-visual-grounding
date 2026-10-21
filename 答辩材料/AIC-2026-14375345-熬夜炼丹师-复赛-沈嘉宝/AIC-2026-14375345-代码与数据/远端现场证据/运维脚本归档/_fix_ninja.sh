#!/bin/bash
# 修 ninja：后端 JIT 编译需要它，而它不在 PATH 上（PATH 只有 cuda/bin 与 conda/bin）
set -u
. /root/rivermind-data/_q38_env.sh
R=/root/rivermind-data
V=$R/venv_q38

echo "=== 现状 ==="
echo "  venv/bin/ninja : $(ls -d $V/bin/ninja 2>/dev/null || echo '（无）')"
echo "  系统 ninja     : $(command -v ninja || echo '（无）')"
echo "  venv/bin 在 PATH 上吗: $(case ":$PATH:" in *":$V/bin:"*) echo 是;; *) echo '否 ← 问题所在';; esac)"

echo
echo "=== 停掉正在失败的起服 ==="
pkill -f "_serve_run.sh" 2>/dev/null
pkill -f "vllm serve" 2>/dev/null
sleep 3
if pgrep -af "vllm serve|_serve_run.sh" >/dev/null 2>&1; then
  pgrep -af "vllm serve|_serve_run.sh" | sed 's/^/  ★残留 /'
else
  echo "  ✓ 已全部停止"
fi

echo
echo "=== 装 ninja 进 venv（uv 走镜像；已装则秒过）==="
UV_CACHE_DIR="$R/uvcache" UV_LINK_MODE=copy \
  /root/miniconda3/bin/uv pip install --python "$V/bin/python" \
  --default-index https://mirrors.aliyun.com/pypi/simple ninja 2>&1 | tail -6

echo
echo "=== 复验（★把 venv/bin 放到 PATH 最前）==="
export PATH="$V/bin:$PATH"
echo "  which ninja : $(command -v ninja || echo '（仍无）')"
echo "  ninja 版本  : $(ninja --version 2>&1 | head -1)"
echo "=== DONE ==="

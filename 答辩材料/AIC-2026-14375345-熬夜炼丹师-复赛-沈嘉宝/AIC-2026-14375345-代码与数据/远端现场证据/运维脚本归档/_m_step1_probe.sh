#!/bin/bash
# M 件第 1 步：thinking 探针（预登记 M §3.3、§8 R1）+ 记录默认后端失败原因（复现用）
set -u
. /root/rivermind-data/_q38_env.sh
R=/root/rivermind-data
PY=$R/venv_q38/bin/python

echo "=== 0) 记录：默认后端为何失败 ==="
echo "  成功后端：$(grep -a '成功后端' "$R/_q38_serve.log" 2>/dev/null | tail -1 | sed 's/^ *//')"
echo "  --- 日志里与失败相关的行 ---"
grep -anE "ninja|FileNotFoundError|ImportError|RuntimeError|ERROR|✗" "$R/_q38_serve.log" 2>/dev/null \
  | grep -viE "WARNING.*Transformers" | head -14 | cut -c1-165 | sed 's/^/    /'
echo "  --- vllm3x 日志目录里是否有历史后端日志 ---"
ls -1 "$R"/vllm38*.log 2>/dev/null | sed 's/^/    /'

echo
echo "=== 1) 服务自证 ==="
curl -s --max-time 8 http://127.0.0.1:8000/v1/models 2>/dev/null \
  | tr ',' '\n' | grep -E '"id"|"root"' | sed 's/^/  /'

echo
echo "=== 2) thinking 探针（同图同 query，THINKING=default 与 0 各一次）==="
cd "$R"
QWEN_ENGINE=api32b QWEN_API_BASE=http://127.0.0.1:8000/v1 QWEN_API_MODEL=qwen38-27b \
QWEN_DATA=$R/dataset QWEN_CORES=$R/_ordinal_cores_r2.json \
QWEN_MODALITIES=3m QWEN_RULES5=1 QWEN_MAXNEW=1024 \
  timeout 600 $PY -u _m_thinking_probe.py 2>&1 | cut -c1-420 | sed 's/^/  /'
echo "=== DONE ==="

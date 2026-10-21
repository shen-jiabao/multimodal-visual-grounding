#!/bin/bash
# 建 lib64 → lib 符号链接 + 只试缺省后端重启（保留日志，便于抓完整报错）
set -u
R=/root/rivermind-data
V=$R/venv_q38
CU=$V/lib/python3.10/site-packages/nvidia/cu13

echo "=== ① 停掉当前重试 ==="
pkill -f _q38_serve_run.sh 2>/dev/null && echo "  已停 serve_run" || echo "  （无）"
pkill -f "vllm serve" 2>/dev/null && echo "  已停 vllm" || echo "  （无）"
sleep 6
echo "  剩余 vllm $(pgrep -cf 'vllm serve' || echo 0)"

echo; echo "=== ② 建 lib64 符号链接（链接器按惯例找 lib64）==="
for d in "$CU"; do
  if [ -d "$d/lib" ] && [ ! -e "$d/lib64" ]; then
    ln -s lib "$d/lib64" && echo "  ✓ $d/lib64 → lib"
  else
    echo "  （$d：lib64 已存在或 lib 不存在）"
  fi
done
ls -ld "$CU/lib64" 2>&1 | sed 's/^/  /'
echo "  验证：$(ls "$CU/lib64/libcudart.so.13" 2>&1 | tail -1)"

echo; echo "=== ③ 重启（Q38_ATTN=default：只试缺省，不截断日志）==="
: > "$R/_q38_serve.log"
cd $R
Q38_ATTN=default setsid nohup bash "$R/_q38_serve_run.sh" > /dev/null 2>&1 < /dev/null &
sleep 3
echo "  已启动；进程 $(pgrep -cf _q38_serve_run.sh || echo 0)"

echo; echo "=== ④ 观察 200 秒 ==="
for i in $(seq 1 20); do
  sleep 10
  if curl -s --max-time 5 http://127.0.0.1:8000/v1/models 2>/dev/null | grep -q qwen38-27b; then
    echo "  ✓ 就绪（$(( i * 10 )) 秒）"; break
  fi
  pgrep -f "vllm serve" >/dev/null || { echo "  ✗ 进程退出（$(( i * 10 )) 秒）"; break; }
  [ $(( i % 4 )) -eq 0 ] && echo "    …$(( i * 10 ))s｜$(tail -1 "$R/vllm38.log" 2>/dev/null | cut -c1-100)"
done

echo; echo "=== ⑤ vLLM 日志尾 25 行 ==="
tail -25 "$R/vllm38.log" 2>/dev/null | cut -c1-190 | sed 's/^/  /'

echo; echo "=== ⑥ 若失败，取 ninja 完整报错 ==="
grep -nE "Ninja build failed|undefined reference|cannot find -l|collect2|ld returned" "$R/vllm38.log" 2>/dev/null | tail -10 | cut -c1-190 | sed 's/^/  /'
grep -n "Ninja build failed" "$R/vllm38.log" 2>/dev/null | head -1 | cut -d: -f1 | while read -r ln; do
  sed -n "${ln},$(( ln + 30 ))p" "$R/vllm38.log" | cut -c1-190 | sed 's/^/    /'
done
echo "=== DONE ==="
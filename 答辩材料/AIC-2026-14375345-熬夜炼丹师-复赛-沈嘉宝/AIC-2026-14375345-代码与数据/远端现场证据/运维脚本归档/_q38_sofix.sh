#!/bin/bash
# 补齐 pip CUDA 包缺失的 dev 符号链接（libX.so → libX.so.N），这是 -lX 能解析的前提
# 实测：lib/ 里有 libcudart.so.13 但没有 libcudart.so ⇒ '/usr/bin/ld: cannot find -lcudart'
set -u
R=/root/rivermind-data
CU=$R/venv_q38/lib/python3.10/site-packages/nvidia/cu13
LIB=$CU/lib

echo "=== ① 当前缺失情况 ==="
n=0
for f in "$LIB"/lib*.so.*; do
  [ -e "$f" ] || continue
  base=$(basename "$f")
  case "$base" in
    *.so.*.*|*alt*) continue ;;   # 跳过 3 段版本号与 alt 变体
  esac
  stem=${base%%.so.*}             # libcudart
  if [ ! -e "$LIB/$stem.so" ]; then
    echo "  缺 $stem.so（有 $base）"
    n=$(( n + 1 ))
  fi
done
echo "  合计缺 $n 个"

echo; echo "=== ② 补齐 ==="
for f in "$LIB"/lib*.so.*; do
  [ -e "$f" ] || continue
  base=$(basename "$f")
  case "$base" in *.so.*.*|*alt*) continue ;; esac
  stem=${base%%.so.*}
  if [ ! -e "$LIB/$stem.so" ]; then
    ln -s "$base" "$LIB/$stem.so" && echo "  ✓ $stem.so → $base"
  fi
done

echo; echo "=== ③ 复验手工链接 ==="
echo 'int main(){return 0;}' > /tmp/_t.c
if c++ /tmp/_t.c -L"$LIB" -lcudart -o /tmp/_t 2>/tmp/_lderr; then
  echo "  ✓ -lcudart 现在可解析"
else
  echo "  ✗ 仍失败：$(head -2 /tmp/_lderr | tr '\n' ' ')"
fi
ls -l "$LIB/libcudart.so" 2>&1 | sed 's/^/  /'

echo; echo "=== ④ 重启（只试缺省后端；同样清 JIT 缓存）==="
pkill -f _q38_serve_run.sh 2>/dev/null || true
pkill -f "vllm serve" 2>/dev/null || true
sleep 5
rm -rf /root/.cache/flashinfer
: > "$R/_q38_serve.log"
cd $R
Q38_ATTN=default setsid nohup bash "$R/_q38_serve_run.sh" > /dev/null 2>&1 < /dev/null &
sleep 3
echo "  已启动；进程 $(pgrep -cf _q38_serve_run.sh || echo 0)"

echo; echo "=== ⑤ 观察 240 秒 ==="
for i in $(seq 1 24); do
  sleep 10
  if curl -s --max-time 5 http://127.0.0.1:8000/v1/models 2>/dev/null | grep -q qwen38-27b; then
    echo "  ✓ **就绪**（$(( i * 10 )) 秒）"; break
  fi
  pgrep -f "vllm serve" >/dev/null || { echo "  ✗ 进程退出（$(( i * 10 )) 秒）"; break; }
  [ $(( i % 4 )) -eq 0 ] && echo "    …$(( i * 10 ))s｜$(tail -1 "$R/vllm38.log" 2>/dev/null | cut -c1-95)"
done

echo; echo "=== ⑥ 结果 ==="
curl -s --max-time 8 http://127.0.0.1:8000/v1/models 2>/dev/null | head -c 240 | sed 's/^/  /'
echo
grep -nE "cannot find -l|Ninja build failed" "$R/vllm38.log" 2>/dev/null | tail -4 | cut -c1-180 | sed 's/^/  ✗ /'
grep -E "model weights took|KV cache size|Encoder cache|Available KV|Starting vLLM API" "$R/vllm38.log" 2>/dev/null | tail -6 | cut -c1-150 | sed 's/^/  /'
echo "=== DONE ==="
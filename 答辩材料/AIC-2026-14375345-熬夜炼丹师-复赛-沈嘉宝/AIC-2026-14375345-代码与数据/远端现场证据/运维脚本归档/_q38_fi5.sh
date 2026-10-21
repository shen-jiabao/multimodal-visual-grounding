#!/bin/bash
# 补 -L 链接路径（-lcudart 找不到）+ 清 JIT 缓存 + 重启
set -u
R=/root/rivermind-data
V=$R/venv_q38
CU=$V/lib/python3.10/site-packages/nvidia/cu13

echo "=== ① 停 ==="
pkill -f _q38_serve_run.sh 2>/dev/null && echo "  已停 serve_run" || echo "  （无）"
pkill -f "vllm serve" 2>/dev/null && echo "  已停 vllm" || echo "  （无）"
sleep 6

echo; echo "=== ② 更新环境文件：补 LDFLAGS/LIBRARY_PATH ==="
cat > "$R/_q38_env.sh" <<EOS
# Qwen3.8-27B 起服环境（_q38_fi5.sh）
# 实测三处坑：
# 1) 系统无 nvcc/无 /usr/local/cuda ⇒ 已把 pip cu13 链到 /usr/local/cuda
# 2) nvcc 13.2.86 vs CUDA runtime 头 13.0.96（CUDART_VERSION=13000）⇒ CCCL 拒编
#    ⇒ 用 FLASHINFER_EXTRA_CUDAFLAGS 关掉一致性检查
# 3) flashinfer JIT 用 c++ 链接，未带 -L ⇒ '/usr/bin/ld: cannot find -lcudart'
#    ⇒ 用 FLASHINFER_EXTRA_LDFLAGS + LIBRARY_PATH 显式给出
export CUDA_HOME="/usr/local/cuda"
export CUDA_PATH="/usr/local/cuda"
export PATH="/usr/local/cuda/bin:\$PATH"
export FLASHINFER_NVCC="/usr/local/cuda/bin/nvcc"
export FLASHINFER_EXTRA_CUDAFLAGS="-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK"
export FLASHINFER_EXTRA_LDFLAGS="-L/usr/local/cuda/lib64 -L/usr/local/cuda/lib"
export LIBRARY_PATH="/usr/local/cuda/lib64:/usr/local/cuda/lib:\${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/local/cuda/lib:\${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
EOS
sed 's/^/  /' "$R/_q38_env.sh"

echo; echo "=== ③ 清 flashinfer JIT 缓存（强制重生成 build.ninja，带动新 -L） ==="
du -sh /root/.cache/flashinfer 2>/dev/null | sed 's/^/  清前 /'
rm -rf /root/.cache/flashinfer && echo "  ✓ 已清 /root/.cache/flashinfer"

echo; echo "=== ④ 手工验证 -lcudart 现在可解析 ==="
echo 'int main(){return 0;}' > /tmp/_t.c
c++ /tmp/_t.c -L/usr/local/cuda/lib64 -lcudart -o /tmp/_t 2>&1 | head -3 | sed 's/^/  /' \
  && echo "  ✓ 链接成功（-lcudart 可解析）"

echo; echo "=== ⑤ 重启（只试缺省后端） ==="
: > "$R/_q38_serve.log"
cd $R
Q38_ATTN=default setsid nohup bash "$R/_q38_serve_run.sh" > /dev/null 2>&1 < /dev/null &
sleep 3
echo "  已启动；进程 $(pgrep -cf _q38_serve_run.sh || echo 0)"

echo; echo "=== ⑥ 观察 240 秒 ==="
for i in $(seq 1 24); do
  sleep 10
  if curl -s --max-time 5 http://127.0.0.1:8000/v1/models 2>/dev/null | grep -q qwen38-27b; then
    echo "  ✓ 就绪（$(( i * 10 )) 秒）"; break
  fi
  pgrep -f "vllm serve" >/dev/null || { echo "  ✗ 进程退出（$(( i * 10 )) 秒）"; break; }
  [ $(( i % 4 )) -eq 0 ] && echo "    …$(( i * 10 ))s｜$(tail -1 "$R/vllm38.log" 2>/dev/null | cut -c1-95)"
done

echo; echo "=== ⑦ 结果 ==="
curl -s --max-time 8 http://127.0.0.1:8000/v1/models 2>/dev/null | head -c 220 | sed 's/^/  /'
echo
echo "  --- 报错（若有）---"
grep -nE "cannot find -l|Ninja build failed|OutOfMemory|Unable to|ERROR" "$R/vllm38.log" 2>/dev/null | tail -8 | cut -c1-185 | sed 's/^/  /'
echo "  --- 关键成功行 ---"
grep -E "model weights took|KV cache size|Encoder cache|Starting vLLM|Available KV" "$R/vllm38.log" 2>/dev/null | tail -6 | cut -c1-150 | sed 's/^/  /'
echo "=== DONE ==="
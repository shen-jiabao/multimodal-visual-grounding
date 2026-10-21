#!/bin/bash
# Qwen3.8-27B 起服务（预登记 I）
#
# 显存（实测权重 51.77 GiB = 55.6 GB，18 分片，逐片 sha256 已过）：
#   2×40G → TP=2（每卡约 27.8 GB）  ★当前目标
#   1×80G → TP=1
#   1×40G → **装不下，脚本直接拒启**
#
# ★ 与 32B bf16 那次的关系：那是"同族精度修复"（唯一为正机制的延续）；
#   本件是**跨代新架构**、无接地先验、且 27B < 32B ⇒ **不给正预期**。
#
# ★ 已知坑（32B 那次实测，本脚本已固化）：
#   ① `--limit-mm-per-prompt {"image":1,"video":0}` —— 默认会按"最大尺寸视频"做编码器
#      探测跑，一次性分配 5.77 GiB 而 OOM；限掉 video 后 budget 从 151250 降到 16384 tokens
#   ② `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` —— 缓解碎片
#   ③ attention backend 逐个重试（该模型 GDN 由 vLLM 原生实现，不依赖 flash_attn）
#   ④ **目录名断言** —— 必须含 Qwen3.8-27B、不得含 32B-Instruct（防起错模型使读数无意义）
set -u

R=/root/rivermind-data
VENV=${Q38_VENV:-$R/venv_q38}          # ★新 venv（vLLM 0.23.0，G0 已过）
DIR=${Q38_DIR:-$R/models/Qwen3.8-27B}
NAME=${Q38_NAME:-qwen38-27b}
PORT=${Q38_PORT:-8000}
LOG=${Q38_LOG:-$R/vllm38.log}
UTIL=${Q38_UTIL:-0.90}
MAXLEN=${Q38_MAXLEN:-8192}
MAXBATCH=${Q38_MAXBATCH:-2048}
MMLIMIT=${Q38_MMLIMIT:-'{"image":1,"video":0}'}
EXTRA=${Q38_EXTRA:-}
ALLOC=${Q38_ALLOC:-expandable_segments:True}
ATTN=${Q38_ATTN:-auto}
# ★ custom all-reduce（TP 通信的优化路径）
#   实测坑：本实例（2×A100-SXM4-40GB，NV12 NVLink，P2P=True，驱动 580.126.09，torch 2.11+cu130）
#   起服时 TP1 在 "Profiling CUDA graph memory" 阶段崩溃：
#     Failed: Cuda error /workspace/csrc/custom_all_reduce.cuh:455 'invalid argument'
#     → Worker proc VllmWorker-1 died → Engine core initialization failed
#   （伴随的 shm_broadcast "No available shared memory broadcast block found in 60 seconds"
#     是**后果**不是原因——TP1 先死，TP0 在等它。）
#   拓扑与 P2P 均正常，故判为该内核在 cu130/A100 组合上的兼容性问题。
#   vLLM 源码 custom_all_reduce.py:104/152/163 本身即在不适配时提示"显式指定
#   disable_custom_all_reduce=True"，故此处按其官方建议关闭，退回 NCCL all-reduce。
#   这是**基础设施路径**的替换，不改动提示词/解码/并发（M5 单变量前提不受影响）。
NOCAR=${Q38_NOCAR:-1}
# ★ 目录名断言的两个子串改成可配（默认值 = 原 q38 行为，逐字不变）
#   用途：同一套起服机制复用到异质模型（`_prereg_K.md` 的 Qwen2.5-VL-72B-AWQ）——
#   只换 MUST/FORBID 两个字符串，起服机制本身不复制第二份（本项目纪律：判据只实现一处）。
MUST=${Q38_MUST:-Qwen3.8-27B}
FORBID=${Q38_FORBID:-32B-Instruct}

echo "=== 0) 目录名断言（★防起错模型） ==="
echo "  目标 $DIR"
case "$DIR" in *"$FORBID"*) echo "  ★含 $FORBID ⇒ 中止（本件必须用 $MUST）"; exit 9 ;; esac
case "$DIR" in *"$MUST"*) echo "  ✓ 含 $MUST" ;; *) echo "  ★目录名不含 $MUST ⇒ 中止"; exit 9 ;; esac

echo; echo "=== 1) GPU 前置检查 + 自动定 TP ==="
if ! ls /dev/nvidia[0-9]* >/dev/null 2>&1; then echo "  ★无 /dev/nvidia* ⇒ 未分配 GPU，中止"; exit 9; fi
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader | sed 's/^/  /'
NG=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
MX=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | sort -n | tail -1)
echo "  GPU 数 $NG｜单卡最大 $(( MX / 1024 )) GB"
if [ "$NG" -ge 2 ]; then TP=2
elif [ "$MX" -ge 75000 ]; then TP=1
else
  echo "  ★权重 55.6 GB：需 **2×40G** 或 1×80G；当前 $NG×$(( MX / 1024 ))G ⇒ 装不下，中止"
  exit 9
fi
echo "  ⇒ TP=$TP｜util=$UTIL｜maxlen=$MAXLEN｜batch=$MAXBATCH｜mm=$MMLIMIT"
echo "  custom all-reduce：$([ "$NOCAR" = "1" ] && echo '关闭（退回 NCCL）' || echo '启用')"

echo; echo "=== 2) 权重完整性（本地，不联网） ==="
"$VENV/bin/python" - "$DIR" <<'PY'
import json, os, sys
d = sys.argv[1]
idx = os.path.join(d, "model.safetensors.index.json")
if not os.path.exists(idx): print("  ★无 index.json"); sys.exit(9)
m = json.load(open(idx, encoding="utf-8"))["weight_map"]
sh = sorted(set(m.values()))
miss = [s for s in sh if not os.path.exists(os.path.join(d, s))]
tot = sum(os.path.getsize(os.path.join(d, s)) for s in sh if os.path.exists(os.path.join(d, s)))
print(f"  分片 {len(sh)}｜缺失 {len(miss)}｜合计 {tot/1024**3:.2f} GiB｜权重条目 {len(m)}")
if miss: print("  ★缺失:", miss[:5]); sys.exit(9)
PY
[ $? -eq 9 ] && { echo "  ★权重不完整，拒启"; exit 9; }

echo; echo "=== 3) 环境 ==="
"$VENV/bin/python" -c "import vllm,torch;print('  vllm',vllm.__version__,'| torch',torch.__version__)" 2>&1 | sed 's/^/  /'
"$VENV/bin/python" - <<'PY'
try:
    from vllm.model_executor.models.registry import ModelRegistry
    print("  架构登记 Qwen3_5ForConditionalGeneration:",
          "Qwen3_5ForConditionalGeneration" in ModelRegistry.get_supported_archs())
except Exception as e:
    print("  ✗", e)
PY

# ★ nvcc / flashinfer JIT
#   实测两级坑：
#     ① vLLM/flashinfer 的 JIT 要 nvcc，而系统无 nvcc、无 /usr/local/cuda
#        ⇒ 已把 pip 版 CUDA 链到 **/usr/local/cuda**（flashinfer `get_cuda_path()` 的默认回退路径），
#          复验：不设任何环境变量时 `get_cuda_path() == /usr/local/cuda` ✓
#     ② 光设 CUDA_HOME 不够——worker 子进程未继承该变量
#        ⇒ 故**双保险**：符号链接（默认路径）+ 环境文件（显式变量）
PIPCUDA=$(ls -d "$VENV"/lib/python3.*/site-packages/nvidia/cu13 2>/dev/null | head -1)
ENVF="$R/_q38_env.sh"
if [ -f "$ENVF" ]; then . "$ENVF"; echo "  已载入环境文件 $ENVF"; fi
if [ ! -e /usr/local/cuda ] && [ -x "$PIPCUDA/bin/nvcc" ]; then
  mkdir -p /usr/local && ln -s "$PIPCUDA" /usr/local/cuda && echo "  ✓ 补建 /usr/local/cuda 符号链接"
fi
echo "  CUDA_HOME=${CUDA_HOME:-未设}｜nvcc $(command -v nvcc || echo '不在 PATH')"
echo "  /usr/local/cuda → $(readlink -f /usr/local/cuda 2>/dev/null || echo '不存在')"
echo "  flashinfer 探测：$("$VENV/bin/python" -c 'from flashinfer.jit.cpp_ext import get_cuda_path;print(get_cuda_path())' 2>&1 | tail -1)"

if [ -n "$ATTN" ] && [ "$ATTN" != "auto" ]; then CAND=("$ATTN")
elif [ "$ATTN" = "default" ]; then CAND=("")
else CAND=("" "FLASHINFER" "TRITON_ATTN" "XFORMERS"); fi
echo "  attention 候选：$(printf '[%s] ' "${CAND[@]}")"

echo; echo "=== 4) 起 vLLM（逐后端重试；后台分离） ==="
pkill -f "vllm serve" 2>/dev/null && { echo "  已停旧 vllm"; sleep 8; } || echo "  （无旧 vllm）"
ready=0; used=""
for c in "${CAND[@]}"; do
  label="${c:-缺省}"; echo "  --- 尝试后端：$label ---"; : > "$LOG"
  cd /root
  ARGS=(serve "$DIR" --served-model-name "$NAME" --host 0.0.0.0 --port "$PORT"
        --tensor-parallel-size "$TP" --trust-remote-code
        --max-model-len "$MAXLEN" --gpu-memory-utilization "$UTIL" --max-num-seqs 8
        --limit-mm-per-prompt "$MMLIMIT" --max-num-batched-tokens "$MAXBATCH")
  [ "$NOCAR" = "1" ] && ARGS+=(--disable-custom-all-reduce)
  [ -n "$EXTRA" ] && ARGS+=($EXTRA)
  export PYTORCH_CUDA_ALLOC_CONF="$ALLOC"
  if [ -n "$c" ]; then VLLM_ATTENTION_BACKEND="$c" setsid nohup "$VENV/bin/vllm" "${ARGS[@]}" > "$LOG" 2>&1 < /dev/null &
  else setsid nohup "$VENV/bin/vllm" "${ARGS[@]}" > "$LOG" 2>&1 < /dev/null & fi
  echo "    已启动｜日志 $LOG"
  for i in $(seq 1 60); do            # 每后端最多等 10 分钟（27B 加载比 32B 略快）
    sleep 10
    if curl -s --max-time 5 "http://127.0.0.1:$PORT/v1/models" 2>/dev/null | grep -q "$NAME"; then
      ready=1; used="$label"; echo "    ✓ 就绪（$(( i * 10 )) 秒）"; break
    fi
    pgrep -f "vllm serve" >/dev/null || { echo "    ✗ 进程退出（$(( i * 10 )) 秒内）"; break; }
    [ $(( i % 6 )) -eq 0 ] && echo "      …$(( i * 10 ))s｜$(tail -1 "$LOG" 2>/dev/null | cut -c1-95)"
  done
  [ "$ready" = "1" ] && break
  echo "    ✗ 后端 $label 失败，报错行："
  grep -iE "error|out of memory|not supported|ImportError|ModuleNotFound|unrecognized" "$LOG" 2>/dev/null | tail -6 | sed 's/^/      /'
  pkill -f "vllm serve" 2>/dev/null; sleep 10
done
if [ "$ready" = "0" ]; then
  echo "  ★全部后端失败。日志尾 30 行："; tail -30 "$LOG"
  echo "  可调 Q38_UTIL / Q38_MAXLEN / Q38_MMLIMIT / Q38_ATTN 后重试"; exit 1
fi
echo "  ⇒ 成功后端：$used"
grep -E "Resolved architecture|model weights took|Encoder cache|Available KV cache|KV cache size" "$LOG" 2>/dev/null | tail -8 | sed 's/^/    /'

echo; echo "=== 5) ★起服后自证（确认是 Qwen3.8-27B，非 32B） ==="
curl -s --max-time 10 "http://127.0.0.1:$PORT/v1/models" | tr ',' '\n' | grep -E '"id"|models/Qwen' | sed 's/^/  /'

echo; echo "=== 6) 客户端环境（本机无旧 venv 时回退 venv_q38 —— 它也含 openai/pillow） ==="
CLI_PY=$R/venv/bin/python
[ -x "$CLI_PY" ] || CLI_PY=$VENV/bin/python
echo "  客户端解释器: $CLI_PY"
"$CLI_PY" -c "import openai,PIL;print('  ✓ openai',openai.__version__,'| pillow',PIL.__version__)" 2>&1 | sed 's/^/  /'

echo; echo "=== 7) 下一步 ==="
cat <<'NOTE'
  ① 20 条冒烟（G2 可解析 ≥16/20、G3 方向 ≥12/20；8~11 放行但登记"接地能力弱"；<8 停）
  ② 过闸 → 全量 4587 条非序数 → 装配（--base result_semi_n32bf16_nonord，锚 ord 689 / 非序数 3741）
  ③ 读数：得分 × 5690 − 689 对 3741；Δ = (得分 − 0.7786) × 5690
NOTE
echo "=== 结束 ==="
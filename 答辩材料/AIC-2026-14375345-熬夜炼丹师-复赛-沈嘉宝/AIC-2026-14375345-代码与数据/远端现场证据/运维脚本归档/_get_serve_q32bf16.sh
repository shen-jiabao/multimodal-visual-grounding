#!/bin/bash
# Qwen3-VL-32B-Instruct（bf16）起服务 —— G 路线（预登记 _prereg_G.md）
#
# 与 AWQ 版的关系：**同一模型、同族机制、去量化**。G 的目的是测"精度修复"的边际值，
# 而不是换模型（换模型的先验在 C 路线）。故**必须确认起的是 bf16 目录，不是 AWQ**。
#
# 显存（实测权重 62.1 GB / 14 分片，非估算）：
#   1×80G → TP=1（权重 62 GB，util 0.92 留约 11 GB 给 KV/激活）
#   2×40G → TP=2（每卡 31 GB，util 0.90 每卡留约 5 GB）
#   1×40G → **装不下，脚本直接拒启**
#
# ★ 与 235B 脚本的差异（都是踩过的坑）：
#   ① 无网下载步骤 —— 权重已下完并已逐片核 sha256（_q32bf16_verify.sh 复核 14/14）
#   ② 加了 **目录名断言**（必须含 `32B-Instruct` 且**不含** `AWQ`）——避免起错模型使读数无意义
#   ③ 加了 **attention backend 探测** —— 本 venv **无 flash_attn**，须显式给后端
#   ④ 加了 **起服后自证模型身份** —— 从服务端读回 model id 与 max_model_len 再落日志
set -u

R=/root/rivermind-data
VENV=${Q32_VENV:-$R/venv}
DIR=${Q32_DIR:-$R/models/Qwen3-VL-32B-Instruct}
NAME=${Q32_NAME:-qwen3-vl-32b-bf16}
PORT=${Q32_PORT:-8000}
LOG=${Q32_LOG:-$R/vllm32bf16.log}
UTIL=${Q32_UTIL:-}
ATTN=${Q32_ATTN:-auto}
MAXLEN=${Q32_MAXLEN:-32768}
EXTRA=${Q32_EXTRA:-}          # 追加参数（空格分隔），如 "--enforce-eager"
ALLOC=${Q32_ALLOC:-expandable_segments:True}   # 缓解碎片（实测 OOM 时提示此项）
# ★ 实测：2×40G 上权重 31.07 GiB/卡（容量 39.49），失败点是 vLLM 的
#   determine_available_memory() 探测跑，它按 max_num_batched_tokens 造 dummy 激活，
#   默认值一次要 5.77 GiB ⇒ 顶爆。故把批 token 砍小、并限定每提示只有 1 张图。
MAXBATCH=${Q32_MAXBATCH:-}
MMLIMIT=${Q32_MMLIMIT:-}
# 别名：给同一份权重多挂一个服务名。用途——`_probe_models.py`（归档参照实现）把
# API_MODEL 写死成 `qwen3-vl-32b`，接线自校验需要"我方管线 vs 归档实现"打同一端点，
# 故给它挂别名，而不是去改归档代码（改了就失去"参照"的意义）。
ALIAS=${Q32_ALIAS:-}

echo "=== 0) 目录名断言（★关键：起错模型则整件事退化）==="
echo "  目标目录 $DIR"
case "$DIR" in
  *AWQ*) echo "  ★目录名含 AWQ ⇒ 中止（G 必须用 bf16）"; exit 9 ;;
esac
case "$DIR" in
  *32B-Instruct*) echo "  ✓ 目录名含 32B-Instruct，无 AWQ" ;;
  *) echo "  ★目录名不含 32B-Instruct ⇒ 中止"; exit 9 ;;
esac

echo; echo "=== 1) GPU 前置检查 + 自动定 TP ==="
if ! ls /dev/nvidia[0-9]* >/dev/null 2>&1; then
  echo "  ★无 /dev/nvidia* 设备节点 ⇒ 中止。当前实例未分配 GPU。"; exit 9
fi
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader | sed 's/^/  /'
NG=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
MAXMB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | sort -n | tail -1)
echo "  GPU 数 $NG ｜单卡最大显存 $(( MAXMB / 1024 )) GB"
if [ "$NG" -ge 2 ]; then
  TP=2
elif [ "$MAXMB" -ge 75000 ]; then
  TP=1
else
  echo "  ★单卡 $(( MAXMB / 1024 )) GB < 75 GB 且 GPU 数 = $NG"
  echo "    bf16 权重 62.1 GB ⇒ 需要 2×40G 或 1×80G。中止。"; exit 9
fi
[ -z "$UTIL" ] && { [ "$TP" = "1" ] && UTIL=0.92 || UTIL=0.90; }
echo "  ⇒ TP=$TP ｜ util=$UTIL ｜ max-len=$MAXLEN"

echo; echo "=== 2) 权重完整性（本地，不联网） ==="
"$VENV/bin/python" - "$DIR" <<'PY'
import json, os, sys
d = sys.argv[1]
idx = os.path.join(d, "model.safetensors.index.json")
if not os.path.exists(idx):
    print("  ★无 index.json ⇒ 中止"); sys.exit(9)
m = json.load(open(idx, encoding="utf-8"))["weight_map"]
sh = sorted(set(m.values()))
miss = [s for s in sh if not os.path.exists(os.path.join(d, s))]
tot = sum(os.path.getsize(os.path.join(d, s)) for s in sh if os.path.exists(os.path.join(d, s)))
print(f"  分片 {len(sh)}｜缺失 {len(miss)}｜合计 {tot/1024**3:.1f} GB｜权重条目 {len(m)}")
if miss:
    print("  ★缺失:", miss[:5]); sys.exit(9)
PY
[ $? -eq 9 ] && { echo "  ★权重不完整，拒绝启动"; exit 9; }
"$VENV/bin/python" -c "import transformers as t; v=t.__version__; print('  transformers', v); assert v.split('.')[0] < '5', '★需 <5.0'"
[ $? -ne 0 ] && { echo "  ★transformers 版本不合规"; exit 9; }

echo; echo "=== 3) attention backend 候选（本 venv 无 flash_attn） ==="
"$VENV/bin/python" - <<'PY'
import importlib
for m in ("flash_attn", "flashinfer", "xformers", "triton"):
    try:
        mod = importlib.import_module(m)
        print(f"  {m:12s} {getattr(mod,'__version__','ok')}")
    except Exception:
        print(f"  {m:12s} ✗")
PY
if [ "$ATTN" = "default" ] || [ "$ATTN" = "none" ]; then
  CAND=("")                       # 只用缺省后端，不重试其它
elif [ -n "$ATTN" ] && [ "$ATTN" != "auto" ]; then
  CAND=("$ATTN")
else
  # 按序重试：**先试缺省（不加环境变量）** —— 此前 AWQ 32B 在本 venv 起服成功过，
  # 多半用的就是缺省路径；失败再依次退 FLASHINFER / TRITON_ATTN / XFORMERS。
  CAND=("" "FLASHINFER" "TRITON_ATTN" "XFORMERS")
fi
echo "  候选序列：$(printf '[%s] ' "${CAND[@]}")"

echo; echo "=== 4) 起 vLLM（逐个后端重试；后台分离） ==="
if pgrep -f "vllm serve" >/dev/null; then
  echo "  已有 vllm serve 在跑 ⇒ 先停掉"
  pkill -f "vllm serve"; sleep 8
fi
ready=0; used=""
for c in "${CAND[@]}"; do
  label="${c:-缺省}"
  echo "  --- 尝试后端：$label ---"
  : > "$LOG"
  cd /root
  ARGS=(serve "$DIR" --served-model-name "$NAME" ${ALIAS:+"$ALIAS"} --host 0.0.0.0 --port "$PORT"
        --tensor-parallel-size "$TP" --mm-encoder-tp-mode data --trust-remote-code
        --max-model-len "$MAXLEN" --gpu-memory-utilization "$UTIL" --max-num-seqs 8)
  [ -n "$EXTRA" ] && ARGS+=($EXTRA)
  [ -n "$MAXBATCH" ] && ARGS+=(--max-num-batched-tokens "$MAXBATCH")
  [ -n "$MMLIMIT" ] && ARGS+=(--limit-mm-per-prompt "$MMLIMIT")
  export PYTORCH_CUDA_ALLOC_CONF="$ALLOC"
  echo "    util=$UTIL maxlen=$MAXLEN batch=${MAXBATCH:-缺省} mm=${MMLIMIT:-缺省} extra='${EXTRA:-无}' alloc=$ALLOC"
  if [ -n "$c" ]; then
    VLLM_ATTENTION_BACKEND="$c" setsid nohup "$VENV/bin/vllm" "${ARGS[@]}" \
      > "$LOG" 2>&1 < /dev/null &
  else
    setsid nohup "$VENV/bin/vllm" "${ARGS[@]}" > "$LOG" 2>&1 < /dev/null &
  fi
  echo "    已启动｜日志 $LOG"

  for i in $(seq 1 48); do            # 每后端最多等 8 分钟
    sleep 10
    if curl -s --max-time 5 "http://127.0.0.1:$PORT/v1/models" 2>/dev/null | grep -q "$NAME"; then
      ready=1; used="$label"; echo "    ✓ 就绪（$(( i * 10 )) 秒）"; break
    fi
    if pgrep -f "vllm serve" >/dev/null; then :; else
      echo "    ✗ 进程已退出（$(( i * 10 )) 秒内）"; break
    fi
    [ $(( i % 6 )) -eq 0 ] && echo "      …$(( i * 10 ))s｜$(tail -1 "$LOG" 2>/dev/null | cut -c1-100)"
  done
  [ "$ready" = "1" ] && break
  echo "    ✗ 后端 $label 失败，报错行："
  grep -iE "error|backend|not supported|ImportError|ModuleNotFound" "$LOG" 2>/dev/null \
    | tail -5 | sed 's/^/      /'
  pkill -f "vllm serve" 2>/dev/null; sleep 10
done

if [ "$ready" = "0" ]; then
  echo "  ★全部后端尝试失败。日志尾 30 行："; tail -30 "$LOG"
  echo "  可用 Q32_ATTN=<后端> 或 Q32_UTIL / Q32_MAXLEN 调整后重试"; exit 1
fi
echo "  ⇒ 成功后端：$used"
echo "  日志关键行："
grep -E "Loading|Resolved architecture|model weights took|GPU KV cache|Using .* backend" "$LOG" 2>/dev/null | tail -8 | sed 's/^/    /'

echo; echo "=== 6) 起服后自证（★确认起的是 bf16，不是 AWQ） ==="
curl -s --max-time 10 "http://127.0.0.1:$PORT/v1/models" | sed 's/^/  /'
echo
echo "  本机权重真身（从服务端无法直读，改从日志确认模型路径）："
grep -m3 -n "$DIR" "$LOG" | sed 's/^/    /'

echo; echo "=== 7) 下一步（★顺序不可颠倒） ==="
cat <<'NOTE'
  ① 接线自校验：python -B _smoke_qwen32b.py      ← 与 AWQ 同一条，先证线路通
  ② 20 条目检：确认出框落在查询目标上，非整幅/角落
  ③ 全量 4587 条非序数：复用 _launch_qwen32b.sh，**只改输出与日志两个路径**
     （提示词/解码/并发逐字不变 —— 这是"单变量"的前提）
  ④ 装配：python -B _assembly_32b.py --base result_semi_n32_orddirect
     读数：得分 × 5690 − 689 对 3730
  ⑤ ★交叉验算：bf16 产物 vs AWQ 产物（B8E2F9E2…）的一致率；
     **若 100% 一致，说明内核没用上 bf16 ⇒ 停下查，不报数**
NOTE
echo "=== 结束 ==="
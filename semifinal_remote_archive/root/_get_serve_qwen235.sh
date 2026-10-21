#!/bin/bash
# Qwen3-VL-235B-A22B 获取 + 起服务（预登记 C 的具体实现）
#
# 全部路径与并行度都经环境变量参数化 —— 因为**必须换多卡机器**，路径会变。
#   QWEN235_REPO   仓库（缺省官方 FP8；AWQ 版见下）
#   QWEN235_DIR    本地目录（缺省 /data/models/<repo 名>）
#   QWEN235_TP     张量并行度（FP8 约 235 GB ⇒ 4×80G 用 TP=4；AWQ 约 120 GB ⇒ 4×40G 用 TP=4）
#   QWEN235_UTIL   gpu_memory_utilization（只影响 KV cache，不改模型数学）
#
# 显存与体积（★已实测核验，非估算）：
#   QuantTrio/Qwen3-VL-235B-A22B-Instruct-AWQ   ← 本脚本缺省
#       实测仓库 55 个文件 / **116.64 GB** / **42 个 safetensors 分片**（各 2.79 GB）
#       config: quant_method=**awq**  bits=**4**  group_size=128
#       architectures=**Qwen3VLMoeForConditionalGeneration**（MoE，94 层文本）
#       ⇒ vLLM 0.11.0 已登记该架构（registry.py:272）⇒ 可加载
#       ⇒ 显存 ≈120 GB ⇒ **4×40G** 或 2×80G；**磁盘需 ≥140 GB**
#   Qwen/Qwen3-VL-235B-A22B-Instruct-FP8        备选：≈235 GB → 4×80G / 8×40G
# 用法：直接 `bash _get_serve_qwen235.sh`（缺省已是 AWQ 版 + TP=4）
set -u

REPO=${QWEN235_REPO:-QuantTrio/Qwen3-VL-235B-A22B-Instruct-AWQ}
DIR=${QWEN235_DIR:-/data/models/$(basename "$REPO")}
TP=${QWEN235_TP:-4}
UTIL=${QWEN235_UTIL:-0.90}
VENV=${QWEN235_VENV:-/root/rivermind-data/venv}
LOG=${QWEN235_LOG:-/root/rivermind-data/vllm235.log}
PORT=${QWEN235_PORT:-8000}

echo "=== 0) GPU 前置检查（没 GPU 就别下 235 GB）==="
if ! ls /dev/nvidia[0-9]* >/dev/null 2>&1; then
  echo "  ★无 /dev/nvidia* 设备节点 ⇒ 中止。当前实例未分配 GPU。"; exit 9
fi
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | sed 's/^/  /'
echo "  仓库 $REPO ｜ 目标目录 $DIR ｜ TP=$TP ｜ util=$UTIL"

echo
echo "=== 1) 磁盘检查（先确认够，再下）==="
mkdir -p "$(dirname "$DIR")"
df -h "$(dirname "$DIR")" | tail -1 | sed 's/^/  /'
avail_g=$(df -BG --output=avail "$(dirname "$DIR")" | tail -1 | tr -dc '0-9')
echo "  可用 ${avail_g} GB"
if [ "${avail_g:-0}" -lt 140 ]; then
  echo "  ★可用空间 < 140 GB ⇒ 中止（FP8 需约 235 GB，AWQ 需约 130 GB）"; exit 9
fi

echo
echo "=== 2) 下载（ModelScope；HF 不可达已实测）==="
if [ -f "$DIR/config.json" ] && [ "$(du -sm "$DIR" | cut -f1)" -gt 50000 ]; then
  echo "  已有 ≥50 GB，跳过下载"
else
  "$VENV/bin/python" -m pip show modelscope >/dev/null 2>&1 \
    || "$VENV/bin/python" -m pip install -q modelscope
  "$VENV/bin/python" - "$REPO" "$DIR" <<'PYEOF'
import sys
from modelscope import snapshot_download
repo, local = sys.argv[1], sys.argv[2]
p = snapshot_download(repo, local_dir=local)
print("  下载完成:", p)
PYEOF
  rc=$?; [ $rc -ne 0 ] && { echo "  ★下载失败 rc=$rc"; exit $rc; }
fi
echo "  目录大小: $(du -sh "$DIR" | cut -f1)"

echo
echo "=== 3) 权重完整性（分片核对）==="
"$VENV/bin/python" - "$DIR" <<'PYEOF'
import json, os, sys
d = sys.argv[1]
idx = os.path.join(d, "model.safetensors.index.json")
if os.path.exists(idx):
    m = json.load(open(idx))["weight_map"]
    sh = sorted(set(m.values()))
    miss = [s for s in sh if not os.path.exists(os.path.join(d, s))]
    tot = sum(os.path.getsize(os.path.join(d, s)) for s in sh
              if os.path.exists(os.path.join(d, s)))
    print(f"  分片 {len(sh)} 个，缺失 {len(miss)} 个，合计 {tot/1024**3:.1f} GB")
    if miss:
        print("  ★缺失:", miss[:5]); sys.exit(9)
else:
    print("  ★无 index.json —— 可能未下完")
    sys.exit(9)
PYEOF
[ $? -eq 9 ] && { echo "  ★权重不完整，拒绝启动"; exit 9; }

echo
echo "=== 4) 起 vLLM（后台分离；注意 transformers 必须 <5.0）==="
"$VENV/bin/python" -c "import transformers as t; v=t.__version__; print('  transformers', v); assert v.split('.')[0] < '5', '★需 <5.0';"
[ $? -ne 0 ] && { echo "  ★transformers 版本不合规"; exit 9; }

if pgrep -f "vllm serve" >/dev/null; then
  echo "  已有 vllm serve 在跑，跳过启动"
else
  cd /root
  setsid nohup "$VENV/bin/vllm" serve "$DIR" \
    --served-model-name qwen3-vl-235b \
    --host 0.0.0.0 --port "$PORT" \
    --tensor-parallel-size "$TP" \
    --mm-encoder-tp-mode data \
    --trust-remote-code \
    --max-model-len 32768 \
    --gpu-memory-utilization "$UTIL" \
    --max-num-seqs 8 \
    > "$LOG" 2>&1 < /dev/null &
  echo "  已分离启动，日志 $LOG"
fi

echo
echo "=== 5) 等就绪（最多 15 分钟；235B 加载比 32B 慢得多）==="
for i in $(seq 1 90); do
  sleep 10
  if curl -s --max-time 5 "http://127.0.0.1:$PORT/v1/models" 2>/dev/null | grep -q "qwen3-vl-235b"; then
    echo "  ✓ 就绪（等待 $((i*10)) 秒）"; break
  fi
  [ "$i" = "90" ] && { echo "  ★15 分钟未就绪，日志尾："; tail -25 "$LOG"; exit 1; }
  [ $((i % 6)) -eq 0 ] && echo "    …已等 $((i*10))s｜$(tail -1 "$LOG" | cut -c1-100)"
done
echo
echo "=== 6) 下一步（★顺序不可颠倒）==="
cat <<'NOTE'
  ① 接线自校验：先跑「20 条人工目检」，确认出框落在查询目标上而非整幅/角落
  ② 小样 300 条（清单 _c_sample300.json 已冻结）：
       QWEN_ENGINE=api32b QWEN_PHASE=A QWEN_ONLY_NONORD=1 \
       QWEN_QIDS=<本目录>/_c_sample300.json \
       QWEN_DATA=<数据目录> QWEN_CORES=<本目录>/_ordinal_cores_r2.json \
       QWEN_OUT_DIRECT=<输出>/qwen235_sample.json QWEN_CONC=8 \
       setsid nohup <venv>/bin/python -u <本目录>/_qwen_full_pipeline.py > c_sample.log 2>&1 &
     ⚠ 注意**必须同时设 QWEN_OUT_SOM**（管线每张图都落盘 som，缺目录会崩——踩过一次）
  ③ 判定（本地，`_d_measure.py --sample _c_sample300.json`）：改动率 ≥ 16.8% 且 IoU 分布健康
  ④ 判据过才全量 4587 条 → `_assembly_32b.py --base result_semi_n32_orddirect` 出件
NOTE
echo "=== 结束 ==="
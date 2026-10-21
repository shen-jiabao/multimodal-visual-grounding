#!/bin/bash
# M 件 329 条无框的**单条回补**（MULTIBOX=0 ⇒ 一次只问一条，模型不可能"跳号"）
# 除 MULTIBOX 外，其余开关与全量逐字相同（模态/规则/max_new/thinking/坐标口径）。
set -u
R=/root/rivermind-data
PY=$R/venv_q38/bin/python

echo "=== 0) 前置：服务与 GPU ==="
svc=$(curl -s --max-time 8 http://127.0.0.1:8000/v1/models 2>/dev/null)
if echo "$svc" | grep -q qwen38-27b; then
  echo "  ✓ 服务就绪"
else
  echo "  ★服务未就绪 ⇒ 请先跑 _serve_launch.sh"; exit 1
fi
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader 2>&1 | sed 's/^/  /'

echo
echo "=== 1) 生成待回补 qid 清单（m_direct.json 里无框的）==="
$PY - "$R" <<'PY'
import json, os, sys
R = sys.argv[1]
d = json.load(open(os.path.join(R, "output", "m_direct.json"),
                   encoding="utf-8"))
nb = sorted(k for k, v in d.items() if not v.get("bbox"))
p = os.path.join(R, "_m_retry_qids.json")
json.dump(nb, open(p, "w"), ensure_ascii=False)
print(f"  无框 {len(nb)} 条 → {p}")
PY

echo
echo "=== 2) 发车（单条模式）==="
if pgrep -f "_qwen_full_pipeline.py" >/dev/null 2>&1; then
  echo "  ★已有管线在跑 ⇒ 中止，避免争显存"; exit 1
fi
cd "$R"
env QWEN_ENGINE=api32b QWEN_API_BASE=http://127.0.0.1:8000/v1 \
    QWEN_API_MODEL=qwen38-27b QWEN_CONC=8 QWEN_PHASE=A QWEN_ONLY_NONORD=0 \
    QWEN_DATA=$R/dataset QWEN_CORES=$R/_ordinal_cores_r2.json \
    QWEN_OUT_DIRECT=$R/output/m_retry.json \
    QWEN_OUT_SOM=$R/output/m_retry_som.json \
    QWEN_QIDS=$R/_m_retry_qids.json \
    QWEN_MODALITIES=3m QWEN_RULES5=1 QWEN_MULTIBOX=0 QWEN_MAXNEW=1024 \
    QWEN_THINKING=0 QWEN_MERGE_SPACE=whole \
    setsid nohup $PY -u "$R/_qwen_full_pipeline.py" \
    > "$R/m_retry.log" 2>&1 < /dev/null &
sleep 25

echo "  进程: $(pgrep -f _qwen_full_pipeline.py | tr '\n' ' ')"
echo "  --- 自证（回读管线首行）---"
grep -m1 "\[pipe\] 模态=" "$R/m_retry.log" 2>/dev/null | sed 's/^/    /'
echo "  --- 日志首 8 行 ---"
head -8 "$R/m_retry.log" 2>/dev/null | cut -c1-150 | sed 's/^/    /'

pl=$(grep -m1 "\[pipe\] 模态=" "$R/m_retry.log" 2>/dev/null)
if echo "$pl" | grep -q "模态=3m" && echo "$pl" | grep -q "multi_box=False"; then
  echo "  ✓ 自检通过（3m + multi_box=False = 单条模式）"
else
  echo "  ★★自检失败 ⇒ 立即停止"
  pkill -f "_qwen_full_pipeline.py" 2>/dev/null
  exit 9
fi
echo "=== DONE ==="

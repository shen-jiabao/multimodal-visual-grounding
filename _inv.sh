#!/bin/bash
# 远端资源盘点（只读）
set -u
echo "===== 预测结果条目数 ====="
cd /root/output || exit 1
for f in predictions.json predictions_v2_highres_slow.json predictions_grounding.json \
         predictions_vlxseek.json checkpoint_v3_slow_enriched.json \
         enriched_queries_qwen25.json sanity30.json fusion_g4.json; do
  if [ -f "$f" ]; then
    n=$(python3 - "$f" <<'PY'
import json,sys
try:
    d=json.load(open(sys.argv[1]))
    print(len(d))
except Exception as e:
    print("ERR:"+type(e).__name__)
PY
)
    printf "  %-36s %s entries\n" "$f" "$n"
  fi
done

echo
echo "===== 目录体积 ====="
du -sh /root/LocateAnything-3B /root/GroundingDINObase /root/Rex-Omni-FP16 \
       /root/VLX-Seek /root/infer /root/output /root/dataset /root/chunks \
       /models/LocateAnything-3B /models/Qwen2.5-3B-Instruct \
       /models/LocateAnything-3B.zip /models/git-2.45.2 2>/dev/null

echo
echo "===== 数据集 ====="
ls /root/dataset/
echo "--- 顶层 ---"
ls "/root/dataset/初赛数据集-基于大模型的多模态视觉理解与推理/"
echo "--- Images ---"
ls "/root/dataset/初赛数据集-基于大模型的多模态视觉理解与推理/Images/"
for d in "/root/dataset/初赛数据集-基于大模型的多模态视觉理解与推理/Images/"*/; do
  printf "  %-12s %s files\n" "$(basename "$d")" "$(ls "$d" | wc -l)"
done
echo "--- queries ---"
ls "/root/dataset/初赛数据集-基于大模型的多模态视觉理解与推理/queries/" 2>/dev/null

echo
echo "===== Rex-Omni-FP16 ====="
ls -la /root/Rex-Omni-FP16
echo
echo "===== VLX-Seek ====="
ls -la /root/VLX-Seek | head -30
echo
echo "===== GroundingDINObase ====="
ls -la /root/GroundingDINObase
echo
echo "===== LocateAnything-3B ====="
ls -la /root/LocateAnything-3B; ls -la /root/LocateAnything-3B/* | head -20

echo
echo "===== conda 环境 ====="
ls -d /root/miniconda3/envs/* /root/vlxseek_env /root/wedetect_env 2>/dev/null

echo
echo "===== 正在运行的进程 ====="
ps -eo pid,etime,pcpu,pmem,cmd --sort=-pcpu | head -10
echo
echo "===== GPU ====="
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv

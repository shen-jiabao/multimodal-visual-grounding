#!/bin/bash
# 环境实录（关服前必取）：把"跑出这些产物时所在的环境"固化成可核查的记录。
#
# 为什么必须取：事后重建只能得到**另一次解析结果**（pip 版本解析会变、驱动会变），
# 不能证明当时跑出产物的环境。半决赛要求交"模型文件与环境"，这份就是原始证据。
#
# 采集：系统 / GPU驱动CUDA / 解释器 / 关键包版本 / pip freeze 全量 /
#       模型分片 sha256 / vLLM 服务命令行 / 数据盘点与校验和。
set -u
PY=/root/rivermind-data/venv/bin/python
MODEL=${QWEN_MODEL_DIR:-/root/rivermind-data/models/Qwen3-VL-32B-Instruct-AWQ}
DATA=/root/rivermind-data/dataset
CORES=/root/rivermind-data/_ordinal_cores_v3.json
OUT=/root/_env_record.txt

if [ ! -d "$MODEL" ]; then
  MODEL=$(ls -d /root/rivermind-data/models/*/ 2>/dev/null | head -1)
fi

{
echo "########## 环境实录 ##########"
echo "采集时间: $(date -Is)"
echo "主机名  : $(hostname)"
echo
echo "===== 1) 系统 ====="
uname -a
head -4 /etc/os-release 2>/dev/null || true
echo
echo "===== 2) GPU / 驱动 / CUDA ====="
nvidia-smi 2>&1 | head -12
nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap \
  --format=csv 2>&1
nvcc --version 2>&1 | tail -4 || echo "（无 nvcc，属正常：vLLM 用运行时 CUDA）"
echo
echo "===== 3) 解释器 ====="
"$PY" -V
echo "解释器路径: $PY"
"$PY" -m pip -V 2>&1
echo
echo "===== 4) 关键包版本 ====="
"$PY" - <<'PYEOF'
import importlib.metadata as md
for n in ("vllm", "transformers", "torch", "torchvision", "openai",
          "pillow", "numpy", "paramiko", "accelerate", "triton"):
    try:
        print("%-16s %s" % (n, md.version(n)))
    except Exception:
        print("%-16s (未安装)" % n)
try:
    import torch
    print("torch.version.cuda :", torch.version.cuda)
    print("cudnn              :", torch.backends.cudnn.version())
    print("device             :", torch.cuda.get_device_name(0))
    print("compute capability :", torch.cuda.get_device_capability(0))
except Exception as ex:
    print("torch 探针失败:", repr(ex))
PYEOF
echo
echo "===== 5) pip freeze（全量）====="
"$PY" -m pip freeze 2>&1
echo
echo "===== 6) 模型文件清单与分片 sha256 ====="
echo "目录: $MODEL"
ls -l "$MODEL"
echo "--- sha256（每个文件）---"
find "$MODEL" -type f | sort | while read -r f; do sha256sum "$f"; done
echo "--- config.json ---"
cat "$MODEL/config.json" 2>/dev/null || echo "（无 config.json）"
echo
echo "===== 7) vLLM 服务命令行 ====="
ps -ef | grep -i "[v]llm" || echo "（无 vllm 进程）"
curl -s --max-time 5 http://127.0.0.1:8000/v1/models || echo "（API 未响应）"
echo
echo "===== 8) 数据盘点 ====="
sha256sum "$DATA/queries/queries.json"
sha256sum "$CORES"
echo "visible 图数: $(ls "$DATA/Images/visible" | wc -l)"
echo "--- 图片分片 sha256 ---"
find "$DATA/Images/visible" -type f | sort | while read -r f; do sha256sum "$f"; done
} > "$OUT" 2>&1

echo "环境实录已写 $OUT"
echo "行数 $(wc -l < "$OUT") | 大小 $(du -h "$OUT" | cut -f1)"
echo "----- 开头 30 行 -----"
head -30 "$OUT"
echo "----- 结尾 12 行 -----"
tail -12 "$OUT"
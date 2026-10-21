#!/bin/bash
# 开卡确认 + 现有约定清单
set -u
. /root/rivermind-data/_q38_env.sh
R=/root/rivermind-data

echo "=== ① GPU ==="
echo "  /dev/nvidia* : $(ls /dev/nvidia[0-9]* 2>/dev/null | tr '\n' ' ' || echo '（无）')"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,driver_version,compute_cap \
  --format=csv,noheader 2>&1 | sed 's/^/  /'
echo "  torch: $($R/venv_q38/bin/python -c "import torch;print('avail',torch.cuda.is_available(),'|',torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-')" 2>&1 | tail -1)"

echo
echo "=== ② 已上传的脚本/文档 ==="
ls -1 "$R"/*.py "$R"/*.sh 2>/dev/null | xargs -n1 basename | sed 's/^/  /'

echo
echo "=== ③ 数据集里的 queries.json 字段（确认三模态键名） ==="
$R/venv_q38/bin/python - "$R" <<'PY'
import json, os, sys
q = json.load(open(os.path.join(sys.argv[1], "dataset", "queries",
                                "queries.json"), encoding="utf-8"))
k = next(iter(q))
print("  样本 qid:", k)
print("  字段:", json.dumps(q[k], ensure_ascii=False))
PY
echo "=== DONE ==="

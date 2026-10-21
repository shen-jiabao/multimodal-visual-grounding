#!/bin/bash
# LA 起跑前的交接核实：逐项确认，不凭"我上传过了"。
VPY=/srv/venv-la2/bin/python
echo "== 1) _locate_full.py 是否含 bbox_all（融合规则的前提）=="
grep -c bbox_all /root/_locate_full.py
grep -n "bbox_all" /root/_locate_full.py | head -4

echo
echo "== 2) LA 启动器在位且含 GPU 空闲闸门 =="
ls -l /root/_launch_locate_semi.sh
grep -n "GPU 必须空闲" /root/_launch_locate_semi.sh

echo
echo "== 3) venv 依赖实测可导入 =="
$VPY - <<'PY'
import transformers, peft, decord, lmdb
from transformers import AutoProcessor
print("  transformers", transformers.__version__,
      "| peft", peft.__version__, "| decord", decord.__version__,
      "| lmdb", lmdb.__version__)
print("  AutoProcessor OK")
PY

echo
echo "== 4) 模型权重 =="
du -sh /srv/models/LocateAnything-3B
ls /srv/models/LocateAnything-3B/*.safetensors | wc -l
ls /srv/models/LocateAnything-3B/*.incomplete 2>/dev/null | wc -l

echo
echo "== 5) 8B 主任务当前状态 =="
grep -E '^\[[0-9]+/' /srv/output/qwen_full.log | tail -1
echo -n "  [ERR] 计数: "; grep -c '\[ERR\]' /srv/output/qwen_full.log

echo
echo "== 6) 融合/校验脚本在位（本地，仅列出）=="
ls -l /root/_probe_locale.py

#!/bin/bash
# 核对上传物 + 权威核心词表落位。
echo "=== 1) 已上传的脚本与参照物 ==="
for f in /root/_qwen_full_pipeline.py /root/_smoke_qwen32b.py /root/_probe_models.py \
         /root/_launch_qwen32b.sh /root/_probe_qids.json /root/_probe_32b.json \
         /root/rivermind-data/_ordinal_cores_v3.json; do
  if [ -f "$f" ]; then echo "  $(stat -c%s "$f" | xargs printf '%8d')  $f"; else echo "  ★缺失 $f"; fi
done
echo
echo "=== 2) 核心词表：键集是否与本地 r2 一致（PHASE=A 只用到键集）==="
/root/rivermind-data/venv/bin/python - <<'PYEOF'
import json
p = "/root/rivermind-data/_ordinal_cores_v3.json"
d = json.load(open(p, encoding="utf-8"))
print("  当前文件条数 " + str(len(d)))
ks = sorted(d)
print("  首键 " + ks[0] + "  末键 " + ks[-1])
print("  样例核心词: " + str([d[k] for k in ks[:5]]))
PYEOF
echo
echo "=== 3) vLLM 服务状态 ==="
curl -s --max-time 8 http://127.0.0.1:8000/v1/models | head -c 250; echo
pgrep -af "vllm serve" | head -1
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
tail -4 /root/rivermind-data/vllm_serve.log
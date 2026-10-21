#!/bin/bash
# 全量推理开跑前检查：模型、数据、核心词、脚本常量、显存。
# 任何一项不符就**不要开跑**——本项目的纪律是"未验证的产物不落盘"。
echo '===== 1) 模型权重 ====='
ls /srv/models/Qwen3-VL-8B-Instruct/*.safetensors 2>/dev/null | wc -l
ls /srv/models/Qwen3-VL-8B-Instruct/ | grep -c incomplete
du -sh /srv/models/Qwen3-VL-8B-Instruct

echo
echo '===== 2) 数据集 ====='
echo -n 'visible 文件数: '; ls /srv/dataset/Images/visible | wc -l
echo -n 'visible 体量:   '; du -sh /srv/dataset/Images/visible | cut -f1
echo -n 'queries.json:   '; ls -l /srv/dataset/queries/queries.json | awk '{print $5" bytes"}'

echo
echo '===== 3) 核心词 ====='
/root/miniconda3/envs/myconda/bin/python - <<'PY'
import json
c = json.load(open("/srv/output/_ordinal_cores_v3.json", encoding="utf-8"))
q = json.load(open("/srv/dataset/queries/queries.json", encoding="utf-8"))
print(f"  核心词 {len(c)} 条 | 键集 ⊂= queries: {set(c) <= set(q)}")
print(f"  queries {len(q)} 条")
PY

echo
echo '===== 4) 引用图是否已全部上传且完整 ====='
/root/miniconda3/envs/myconda/bin/python - <<'PY'
import json, os
from PIL import Image
D = "/srv/dataset"
q = json.load(open(os.path.join(D, "queries", "queries.json"), encoding="utf-8"))
need = {os.path.basename(v["visible"]) for v in q.values()}
vd = os.path.join(D, "Images", "visible")
have = set(os.listdir(vd))
miss = need - have
print(f"  引用图 {len(need)} | 已上传 {len(need & have)} | 缺失 {len(miss)}")
bad = []
for nm in sorted(need & have):
    try:
        Image.open(os.path.join(vd, nm)).load()
    except Exception:
        bad.append(nm)
print(f"  可完整读取 {len(need & have) - len(bad)} | 损坏/半截 {len(bad)}")
if bad[:5]:
    print("  损坏样例:", bad[:5])
PY

echo
echo '===== 5) 管线脚本常量 ====='
grep -nE '^(MODEL_PATH|DATA|OUT_DIRECT|OUT_SOM|CORES|SAVE_EVERY)' /root/_qwen_full_pipeline.py

echo
echo '===== 6) 显存 ====='
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader

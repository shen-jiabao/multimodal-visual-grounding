#!/bin/bash
# 用现存方案产物出提交件：annotate 版 → submission.json + zip + SHA256
set -e
cd /root/autodl-tmp/work/code27b || exit 1
/root/miniconda3/bin/python - <<'PYEOF'
import json, hashlib, zipfile

queries = json.load(open('/root/autodl-tmp/work/queries.json', encoding='utf-8'))
preds = json.load(open('outputs/predictions_27b_annotate.json', encoding='utf-8'))
ref = json.load(open('predictions_27b_ref.json', encoding='utf-8'))

out = {}
n_fb = 0
for k, qv in queries.items():
    b = preds.get(k)
    if b is None:
        b = [0.49, 0.49, 0.51, 0.51]
        n_fb += 1
    bb = [round(x, 4) for x in b]
    if bb == [0.49, 0.49, 0.51, 0.51]:
        n_fb += 1
    out[k] = {
        "visible": qv["visible"],
        "infrared": qv["infrared"],
        "depth": qv["depth"],
        "query": qv["query"],
        "bbox": bb,
    }
print("entries", len(out), "| fallback(含产物内)", n_fb)

with open('outputs/submission_annotate.json', 'w', encoding='utf-8') as f:
    json.dump(out, f, ensure_ascii=False, indent=2)

z = 'outputs/result_semi_27b_annotate.zip'
with zipfile.ZipFile(z, 'w', zipfile.ZIP_DEFLATED) as zf:
    zf.write('outputs/submission_annotate.json', 'submission.json')
h = hashlib.sha256(open(z, 'rb').read()).hexdigest().upper()
print("zip", z)
print("SHA256", h)

# 逐条核对：出件 vs ref（基准提交包同源）差异数应 = 651
diff = sum(1 for k in out if out[k]['bbox'] != [round(x, 4) for x in (
    ref[k]['bbox'] if isinstance(ref[k], dict) else ref[k])])
print("vs ref diff(4dp)", diff)
PYEOF
ls -la outputs/result_semi_27b_annotate.zip

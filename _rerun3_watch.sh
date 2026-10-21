#!/bin/bash
# 完成监视器：等 infer 进程结束 → 装配提交件 → 与 ref/annotate 三方对比
PID="$1"
[ -z "$PID" ] && { echo "usage: $0 <infer_pid>"; exit 1; }
while kill -0 "$PID" 2>/dev/null; do sleep 120; done
cd /root/autodl-tmp/work/code27b || exit 1
{
echo "[$(date)] infer 进程 $PID 已结束"
[ -s outputs/predictions_rerun3.json ] || echo "WARN: predictions_rerun3.json 缺失或为空"
/root/miniconda3/bin/python - <<'PYEOF'
import json, hashlib, zipfile

q = json.load(open('/root/autodl-tmp/work/queries.json', encoding='utf-8'))
p = json.load(open('outputs/predictions_rerun3.json', encoding='utf-8'))
ref = json.load(open('predictions_27b_ref.json', encoding='utf-8'))
an = json.load(open('outputs/predictions_27b_annotate.json', encoding='utf-8'))
fb = [0.49, 0.49, 0.51, 0.51]


def bx(d, k):
    v = d[k]
    return v['bbox'] if isinstance(v, dict) else v


out = {}
nmiss = 0
for k, qv in q.items():
    b = p.get(k)
    if b is None:
        b = fb
        nmiss += 1
    out[k] = {"visible": qv["visible"], "infrared": qv["infrared"],
              "depth": qv["depth"], "query": qv["query"],
              "bbox": [round(x, 4) for x in b]}
print("entries", len(out), "| missing->fb", nmiss,
      "| fb_total", sum(1 for k in out if out[k]['bbox'] == fb))

json.dump(out, open('outputs/submission_rerun3.json', 'w', encoding='utf-8'),
          ensure_ascii=False, indent=2)
z = 'outputs/result_semi_27b_rerun3.zip'
with zipfile.ZipFile(z, 'w', zipfile.ZIP_DEFLATED) as zf:
    zf.write('outputs/submission_rerun3.json', 'submission.json')
print("SHA256", hashlib.sha256(open(z, 'rb').read()).hexdigest().upper())

r4 = {k: [round(x, 4) for x in bx(ref, k)] for k in ref}
a4 = {k: [round(x, 4) for x in bx(an, k)] for k in an}
o4 = {k: out[k]['bbox'] for k in out}
dref = sum(1 for k in o4 if o4[k] != r4.get(k))
dan = sum(1 for k in o4 if o4[k] != a4.get(k))
both = sum(1 for k in o4 if o4[k] != r4.get(k) and o4[k] != a4.get(k))
print("diff vs ref:", dref, "| diff vs annotate:", dan,
      "| 三版各异:", both)
PYEOF
echo "[$(date)] DONE_MARKER"
} >> outputs/RERUN3_DONE.txt 2>&1

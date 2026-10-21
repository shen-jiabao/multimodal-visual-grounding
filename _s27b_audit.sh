#!/bin/bash
# 盘点现存方案：make_sub.py 逻辑 + annotate 产物完整性 + 与基准对比
cd /root/autodl-tmp/work/code27b || exit 1
echo "=== make_sub.py ==="
cat make_sub.py
echo "=== audit ==="
/root/miniconda3/bin/python - <<'PYEOF'
import json
q = json.load(open('/root/autodl-tmp/work/queries.json', encoding='utf-8'))
d = json.load(open('outputs/predictions_27b_annotate.json', encoding='utf-8'))
ref = json.load(open('predictions_27b_ref.json', encoding='utf-8'))
ks = list(d)
print('结构探测 first =', ks[0], '->', json.dumps(d[ks[0]], ensure_ascii=False)[:200])

def bbox(v):
    return v['bbox'] if isinstance(v, dict) else v

print('queries', len(q), '| preds', len(d), '| ref', len(ref))
miss = [k for k in q if k not in d]
print('missing', len(miss), miss[:5])
fb = [k for k, v in d.items() if bbox(v) == [0.49, 0.49, 0.51, 0.51]]
print('fallback', len(fb), fb[:5])
bad = [k for k, v in d.items()
       if not (len(bbox(v)) == 4 and 0 <= min(bbox(v)) and max(bbox(v)) <= 1
               and bbox(v)[0] <= bbox(v)[2] and bbox(v)[1] <= bbox(v)[3])]
print('illegal', len(bad), bad[:5])
common = set(d) & set(ref)
same = sum(1 for k in common if bbox(d[k]) == bbox(ref[k]))
print('vs ref: common', len(common), 'same', same, 'diff', len(common) - same)
PYEOF

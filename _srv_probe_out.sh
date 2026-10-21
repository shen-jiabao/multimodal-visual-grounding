#!/usr/bin/env bash
# 远端产物内容特征（只读）
set +e
PY=/root/miniconda3/bin/python
D=/root/autodl-tmp/rivermind-data/output

echo "=== head of m_direct_filled.json ==="
head -c 300 $D/m_direct_filled.json; echo
echo "=== head of m_direct.json ==="
head -c 300 $D/m_direct.json; echo

echo
echo "=== stats ==="
$PY - <<'EOF'
import json
for name in ["m_direct.json","m_direct_filled.json","m_retry.json"]:
    p="/root/autodl-tmp/rivermind-data/output/"+name
    try:
        d=json.load(open(p))
    except Exception as e:
        print(name,"ERR",e); continue
    ks=list(d)[:3]
    print(name,"n=",len(d),"sample=",[(k,d[k]) for k in ks])
EOF

echo
echo "=== queries.json stats ==="
$PY - <<'EOF'
import json
p="/root/autodl-tmp/rivermind-data/dataset/queries/queries.json"
d=json.load(open(p))
print("type",type(d).__name__,"n",len(d))
if isinstance(d,list):
    print("keys",list(d[0]))
    print("first",json.dumps(d[0],ensure_ascii=False)[:400])
else:
    k=list(d)[0]
    print("first key",k,"->",json.dumps(d[k],ensure_ascii=False)[:400])
EOF

echo
echo "=== done ==="

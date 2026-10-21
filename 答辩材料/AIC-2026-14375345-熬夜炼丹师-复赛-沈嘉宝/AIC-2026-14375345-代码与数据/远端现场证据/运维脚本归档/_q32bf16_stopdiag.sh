#!/bin/bash
# 停全量 + 诊断空框成因
#   判据：>15% 即停（已触发，20.0%）。区分两类：
#     ① 模型真的输出 [] / 无框 —— 已知系统性行为（AWQ 时亦 ~13.2%），无害
#     ② 解析失败 / 被 maxlen=8192 截断 —— 真问题，必须修
set -u
R=/root/rivermind-data
OUT=$R/output/qwen32bf16_direct.json

echo "=== 1) 停管线（保留产物，可续跑） ==="
pid=$(pgrep -f _qwen_full_pipeline.py | head -1)
if [ -n "$pid" ]; then kill "$pid" 2>/dev/null && echo "  已发 SIGTERM 给 $pid"; sleep 5; fi
pgrep -f _qwen_full_pipeline.py >/dev/null && { pkill -9 -f _qwen_full_pipeline.py; echo "  已强杀"; } || echo "  已停止"
echo "  vllm 仍保留：$(pgrep -cf 'vllm serve' || true) 个（不占额外资源，后续续跑要用）"

echo; echo "=== 2) 记录结构（先看清键） ==="
$R/venv/bin/python - "$OUT" <<'PY'
import json, sys
d = json.load(open(sys.argv[1], encoding="utf-8"))
ks = {}
for v in d.values():
    for k in v:
        ks[k] = ks.get(k, 0) + 1
print("  字段出现次数：", dict(sorted(ks.items(), key=lambda x: -x[1])))
# 打印一条有框、一条无框的完整记录
have = [ (k,v) for k,v in d.items() if v.get("bbox") ]
none = [ (k,v) for k,v in d.items() if not v.get("bbox") ]
if have:
    k, v = have[0]
    print(f"\n  样例（有框）{k}: {json.dumps(v, ensure_ascii=False)[:420]}")
if none:
    k, v = none[0]
    print(f"\n  样例（无框）{k}: {json.dumps(v, ensure_ascii=False)[:420]}")
PY

echo; echo "=== 3) 空框条目的原始回答（区分「模型输出[]」与「解析失败」） ==="
$R/venv/bin/python - "$OUT" <<'PY'
import json, sys, collections
d = json.load(open(sys.argv[1], encoding="utf-8"))
none = {k: v for k, v in d.items() if not v.get("bbox")}
print(f"  空框 {len(none)} / {len(d)}")
c = collections.Counter()
for k, v in list(none.items())[:20]:
    raw = (v.get("raw") or v.get("answer") or v.get("text") or "")
    r = str(raw).replace("\n", " ")[:110]
    print(f"    {k}  reason={v.get('reason','-')}  raw={r!r}")
    c[("empty_raw" if not r else "has_raw")] += 1
print(f"  小计 {dict(c)}")
PY

echo; echo "=== 4) 空框组 vs 有框组：问句长度对比（看是否与截断相关） ==="
$R/venv/bin/python - "$OUT" "$R/dataset" <<'PY'
import json, os, sys
out, data = sys.argv[1], sys.argv[2]
d = json.load(open(out, encoding="utf-8"))
q = json.load(open(os.path.join(data, "queries", "queries.json"), encoding="utf-8"))
def stat(keys):
    ls = [len(q[k]["query"]) for k in keys if k in q]
    if not ls: return "-"
    ls.sort()
    return f"n={len(ls)} p50={ls[len(ls)//2]} max={ls[-1]}"
e = [k for k, v in d.items() if not v.get("bbox")]
o = [k for k, v in d.items() if v.get("bbox")]
print(f"  空框组 问句长度 {stat(e)}")
print(f"  有框组 问句长度 {stat(o)}")
PY

echo; echo "=== 5) 服务端是否报错（截断/超长） ==="
grep -iE "truncat|exceed|max_model_len|longer than|400|500" "$R/vllm32bf16.log" 2>/dev/null | tail -6 | cut -c1-160 | sed 's/^/  /' || echo "  （无）"
echo "  vllm 日志行数 $(wc -l < $R/vllm32bf16.log)"
echo "=== 结束 ==="
#!/bin/bash
# H 第一步：量 154 条水平轴子类的**枚举率**（决定 H 立不立）
#
# 背景（M5 已测）：该子类 err_154 = 40（准确率 74.0%），判据落「20~40 边缘」带，
#   而真正决定 H 上限的是**枚举率**——枚举不出来，任何编号选择都无从谈起。
#   M5 测不到它（归零法只给条数不给身份），必须真跑一次枚举 = H 的第一步。
#
# 预写判据（**跑之前写定**）：
#   movable_h ≈ err_154 × enum_ok_rate
#     movable_h < 10   ⇒ 不立 H
#     10 ≤ movable_h ≤ 20 ⇒ 边缘（立 H 则预登记要更严）
#     movable_h > 20   ⇒ 值得立 H
#   参照：序数 SoM 实测枚举率约 50%（204/407 枚举失败）
set -u
R=/root/rivermind-data
PY=$R/venv/bin/python
QIDS=$R/_axis154.json
SOM=$R/output/axis154_som.json
LOG=$R/axis154_enum.log
MODEL=${QWEN_API_MODEL:-qwen3-vl-32b-bf16}

echo "=== 1) 生成 154 条水平轴子类清单 ==="
$PY - "$R/dataset/queries/queries.json" "$R/_ordinal_cores_r2.json" "$QIDS" <<'PY'
import json, re, sys
q = json.load(open(sys.argv[1], encoding="utf-8"))
cores = set(json.load(open(sys.argv[2], encoding="utf-8")))
PAT = re.compile(r"\b(leftmost|rightmost|topmost|rearmost)\b", re.I)
hit = sorted(k for k, v in q.items() if k not in cores and PAT.search(v.get("query") or ""))
cnt = {}
for k in hit:
    for w in ("leftmost", "rightmost", "topmost", "rearmost"):
        if re.search(r"\b" + w + r"\b", q[k]["query"], re.I):
            cnt[w] = cnt.get(w, 0) + 1
print(f"  命中 {len(hit)} 条｜成分 {cnt}")
if len(hit) != 154:
    print(f"  ★条数 {len(hit)} ≠ 期望 154 ⇒ 停下查口径"); sys.exit(2)
json.dump(hit, open(sys.argv[3], "w"), ensure_ascii=False)
print(f"  已写 {sys.argv[3]}")
PY
[ $? -eq 2 ] && exit 2

echo; echo "=== 2) 跑 Phase B 枚举（仅这 154 条；分离，落文件） ==="
cd $R
: > "$LOG"
QWEN_ENGINE=api32b QWEN_API_BASE=http://127.0.0.1:8000/v1 QWEN_API_MODEL=$MODEL \
QWEN_PHASE=B QWEN_QIDS=$QIDS QWEN_DATA=$R/dataset QWEN_CORES=$R/_ordinal_cores_r2.json \
QWEN_OUT_SOM=$SOM QWEN_OUT_DIRECT=$R/output/_unused_direct.json QWEN_CONC=8 \
  setsid nohup $PY -u $R/_qwen_full_pipeline.py > "$LOG" 2>&1 < /dev/null &
echo "  已启动 PID=$(pgrep -f _qwen_full_pipeline.py | head -1)"

echo; echo "=== 3) 等待（最多 12 分钟） ==="
for i in $(seq 1 72); do
  sleep 10
  pgrep -f _qwen_full_pipeline.py >/dev/null || { echo "  ✓ 结束（$(( i * 10 )) 秒）"; break; }
  [ $(( i % 6 )) -eq 0 ] && echo "    …$(( i * 10 ))s｜$(tail -1 "$LOG" 2>/dev/null | cut -c1-100)"
done
echo "  日志尾 6 行："; tail -6 "$LOG" 2>/dev/null | cut -c1-150 | sed 's/^/    /'

echo; echo "=== 4) 枚举率统计 ==="
$PY - "$SOM" <<'PY'
import json, sys, os, collections
p = sys.argv[1]
if not os.path.exists(p):
    print("  ★无产物"); raise SystemExit(0)
d = json.load(open(p, encoding="utf-8"))
print(f"  产物 {len(d)} 条")
n = collections.Counter()
for k, v in d.items():
    n[v.get("n_cands", 0)] += 1
print(f"  n_cands 分布：{dict(sorted(n.items()))}")
tot = len(d)
ok = sum(c for k, c in n.items() if k >= 2)
fail = sum(c for k, c in n.items() if k <= 1)
print(f"  枚举成功（n_cands≥2）{ok}/{tot} = {ok/max(1,tot):.1%}")
print(f"  枚举失败（n_cands≤1）{fail}/{tot} = {fail/max(1,tot):.1%}  ← 参照：序数 SoM 为 50.1%")
reason = collections.Counter(v.get("reason", "ok") for v in d.values())
print(f"  reason 分布：{dict(reason)}")
e = 40                      # M5 实测 err_154
r = ok / max(1, tot)
mov = e * r
print(f"\n  ★movable_h ≈ err_154({e}) × 枚举率({r:.1%}) = {mov:.1f}")
print("  判据（跑前写定）：<10 不立 H｜10~20 边缘（预登记更严）｜>20 值得立 H")
if mov < 10:   print("  ⇒ 判定：**不立 H**")
elif mov <= 20: print("  ⇒ 判定：**边缘**（立 H 则预登记须更严）")
else:          print("  ⇒ 判定：**值得立 H**")
PY

echo; echo "=== 5) GPU ==="
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | sed 's/^/  /'
echo "=== 结束 ==="
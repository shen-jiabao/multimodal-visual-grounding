#!/bin/bash
# 20 条目检（冒烟）：跑完即打印闸门① 的可机检结果 + 逐条明细
# 闸门（_prereg_F.md §4，跑前已定死）：① 可解析 ≥16/20  ② 方向正确 ≥12/20（须人眼看图）
set -u
R=/root/rivermind-data
PY=$R/venv_vlx/bin/python
OUT=$R/_vlx_smoke20_out.json
VIS=$R/vlx_vis

# attention 实现：官方在 builder.py / qwen3_5_vl_encoder.py 三处硬编码
# flash_attention_2。本机无 nvcc（CUDA toolkit 未装）⇒ flash_attn 无法从源码编译。
# 故回退 sdpa（数学等价的注意力实现，非机制改动）。已留痕：原文件哈希见
# _vlx_attn_fix.sh 输出。不设 VLX_ATTN 则行为与官方一致（会因缺 flash_attn 而崩）。
export VLX_ATTN="${VLX_ATTN:-sdpa}"

echo "=== 0. 现场核验（防跑错版本） ==="
sha256sum "$R/_vlx_direct.py" | sed 's/^/  /'
echo "  期望 ABA02E7CA81916200D080921DD80EEDC77E3DB0D3D16EAEBE8FAE08B52EE6889"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader 2>&1 | sed 's/^/  GPU /'
rm -f "$OUT"           # 冒烟一律从零跑，避免旧产物混入
cd "$R"

echo; echo "=== 1. 跑 20 条（--temp 0.0 贪心；--vis 出目检图） ==="
"$PY" -u _vlx_direct.py --qids _vlx_smoke20.json --out "$OUT" --vis --vis-dir "$VIS" 2>&1 \
  | sed 's/^/  /'
rc=${PIPESTATUS[0]}
echo "  退出码 $rc"

echo; echo "=== 2. 闸门① 机检 + 逐条明细 ==="
"$PY" - <<'PYEOF'
import json
from pathlib import Path
R = Path("/root/rivermind-data")
res = json.loads((R / "_vlx_smoke20_out.json").read_text(encoding="utf-8"))
n = len(res)
ok = [k for k, v in res.items() if v.get("bbox")]
err = [k for k, v in res.items() if v.get("err")]
print(f"  总 {n} 条")
print(f"  ★闸门① 可解析（result_bbox_list 非空）: {len(ok)}/{n} "
      f"（要求 ≥16）→ {'通过' if len(ok) >= 16 else '★不通过'}")
print(f"  异常 {len(err)} 条")
nc = [v.get("n_cand") or 0 for v in res.values()]
if nc:
    print(f"  候选数 min {min(nc)} / max {max(nc)}（WeDetect top-100）")
print()
print("  --- 逐条（供闸门② 人眼看图：框是否落在查询描述的目标附近）---")
for k, v in res.items():
    b = v.get("bbox")
    bs = ("[%.3f %.3f %.3f %.3f]" % tuple(b)) if b else "—"
    print(f"  {k}  cand={v.get('n_cand'):>3}  obj={v.get('object_index')}  {bs}")
    print(f"       Q: {v.get('query', '')[:88]}")
    if v.get("err"):
        print(f"       ✗ {v['err']}")
PYEOF

echo; echo "=== 3. 目检图 ==="
ls -1 "$VIS" 2>/dev/null | head -25 | sed 's/^/  /'
echo "  共 $(ls -1 "$VIS" 2>/dev/null | wc -l) 张，位于 $VIS"
echo "=== 结束 ==="

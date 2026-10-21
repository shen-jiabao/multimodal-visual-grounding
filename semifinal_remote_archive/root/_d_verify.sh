#!/bin/bash
# D 小样开跑前的清点与烟测：图齐否 / 服务在否 / 深度与红外可读否 / 三图请求通否
set -u
D=/root/rivermind-data/dataset
V=/root/rivermind-data/venv/bin/python

echo "=== 1) 图片清点（应 visible 1295 / infrared 707 / depth 707）==="
for m in visible infrared depth; do
  echo "  Images/$m : $(ls "$D/Images/$m" 2>/dev/null | wc -l) 张"
done

echo; echo "=== 2) vLLM 服务 ==="
curl -s --max-time 6 http://127.0.0.1:8000/v1/models | head -c 220; echo

echo; echo "=== 3) 深度/红外读取烟测（前 3 张）==="
"$V" - <<'PYEOF'
import os, sys
sys.path.insert(0, "/root/rivermind-data")
import _qwen_full_pipeline as P
D = "/root/rivermind-data/dataset"
for n in sorted(os.listdir(f"{D}/Images/visible"))[:3]:
    ir = P.load_ir(f"{D}/Images/infrared/{n}")
    dp = P.load_depth(f"{D}/Images/depth/{n}")
    # 深度灰度的非零占比 = 有效深度区域占比（应接近实测 72%）
    px = list(dp.convert("L").getdata())
    nz = sum(1 for v in px if v > 0) / len(px)
    print(f"  {n}  IR={ir.mode}{ir.size}  Depth={dp.mode}{dp.size} "
          f"非零占比={nz:.1%}")
PYEOF

echo; echo "=== 4) 三图请求烟测（1 条真实推理）==="
"$V" - <<'PYEOF'
import json, os, sys
os.environ["QWEN_MODALITIES"] = "3"
sys.path.insert(0, "/root/rivermind-data")
import _qwen_full_pipeline as P
D = "/root/rivermind-data/dataset"
q = json.load(open(f"{D}/queries/queries.json", encoding="utf-8"))
gen = P.make_gen_api()
# 取小样里的第一条（保证是 D 作用范围内的条目）
sample = json.load(open("/root/rivermind-data/_d_sample300.json", encoding="utf-8"))
k = sample[0]
n = os.path.basename(q[k]["visible"])
imgs = [P.Image.open(f"{D}/Images/visible/{n}").convert("RGB"),
        P.load_ir(f"{D}/Images/infrared/{n}"),
        P.load_depth(f"{D}/Images/depth/{n}")]
r = P._direct_one(gen, imgs, q[k]["query"])
print("  qid  :", k)
print("  query:", q[k]["query"][:72])
print("  结果 :", json.dumps(r, ensure_ascii=False)[:300])
print("  判定 :", "✓ 出框" if r.get("bbox") else "★ 未出框")
print("  断言 : 三图提示词尾段一致 =",
      P.THREE_IMG_PROMPT.split("{q}")[1] == P.FULL_DIRECT_PROMPT.split("{q}")[1])
PYEOF
echo; echo "=== 烟测结束 ==="
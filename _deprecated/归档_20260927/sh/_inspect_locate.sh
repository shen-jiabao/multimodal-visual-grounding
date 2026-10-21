#!/bin/bash
# 只读：定位 LocateAnything / VLX 推理入口及其 caption 注入接口（不跑模型）
set -u
echo "===== /root 下 python/sh 脚本 ====="
ls -la /root/*.py /root/*.sh 2>/dev/null
echo
echo "===== LocateAnything-3B 目录结构 ====="
find /root/LocateAnything-3B -maxdepth 2 -name "*.py" 2>/dev/null | head -40
echo
echo "===== 检索 LocateAnywhere/locate 调用 demo/入口 ====="
grep -rln --include=*.py -iE "locateanywhere|locate-anything|from locate|import locate" /root/LocateAnything-3B /root 2>/dev/null | grep -v /output | head -30
echo
echo "===== 可能在 /root 的推理/跑批脚本（含 predict / infer / highres / slow）====="
find /root -maxdepth 3 -type f \( -name "*.py" -o -name "*.sh" \) 2>/dev/null \
    | grep -iE "predict|infer|locate|run|highres|slow|demo|eval" | head -50
echo
echo "===== /root/output 关键结果 schema（各文件前 1 条的键）====="
python3 - <<'PY'
import json,glob,os
for f in ["/root/output/predictions_v2_highres_slow.json",
          "/root/output/predictions_grounding.json",
          "/root/output/predictions_vlxseek.json"]:
    if not os.path.exists(f):
        print(f, "  [缺失]"); continue
    d=json.load(open(f))
    k=next(iter(d))
    print(f, " 节点数", len(d))
    print("   sample:", json.dumps({k:d[k]}, ensure_ascii=False)[:300])
PY
echo
echo "===== LOCATE 环境（如模型相关目录/README）====="
ls -la /root/LocateAnything-3B 2>/dev/null | head
find /root/LocateAnything-3B -maxdepth 2 -iname "readme*" 2>/dev/null
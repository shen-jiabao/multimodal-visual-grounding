#!/bin/bash
# 枚举重跑实时进度（可反复执行，无副作用）。
set -u
PY=/root/miniconda3/envs/myconda/bin/python
echo "=== 进程 ==="
pgrep -af _enum_retry.py || echo "已退出"
echo "=== 日志末 3 行 ==="
tail -n 3 /srv/output/enum_retry.log
echo "=== 覆盖统计 ==="
$PY - <<'PYEOF'
import json
from pathlib import Path
raw = json.loads(Path("/srv/output/_enum_raw.json").read_text(encoding="utf-8"))
tg = json.loads(Path("/srv/output/_enum_targets.json").read_text(encoding="utf-8"))
q = json.loads(Path("/srv/dataset/queries/queries.json").read_text(encoding="utf-8"))

def pairs(d):
    # 必须拼成与 worker 写入 raw 时相同的字符串键 f"{visible}||{core_lower}"。
    # 曾误返回 (visible, core) 元组集合去和字符串键求交集 → 恒为空（假警报）。
    return {f'{q[k]["visible"]}||{c.lower()}' for k, c in d.items()}

pt, pc = pairs(tg["target"]), pairs(tg["control"])
have = set(raw)
print(f"已完成 {len(raw)} / 262 对")
print(f"  目标集 {len(pt & have)}/{len(pt)} | 控制组 {len(pc & have)}/{len(pc)}")
ks = ("p1", "p2", "p3", "t1", "t2", "t3", "t4")
n2 = sum(1 for v in raw.values()
         if max(len(v.get(k) or []) for k in ks) >= 2)
err = [k for k, v in raw.items() if "err" in v]
print(f"  至少一种提示给出 >=2 框: {n2} | 整对失败: {len(err)}")
PYEOF
echo "=== GPU ==="
nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader

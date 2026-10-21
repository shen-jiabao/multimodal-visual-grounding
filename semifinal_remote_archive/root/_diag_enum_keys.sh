#!/bin/bash
# 诊断：raw 的键与 targets/queries 推出的键为何不匹配。
set -u
/root/miniconda3/envs/myconda/bin/python - <<'PYEOF'
import json
from pathlib import Path

raw = json.loads(Path("/srv/output/_enum_raw.json").read_text(encoding="utf-8"))
tg = json.loads(Path("/srv/output/_enum_targets.json").read_text(encoding="utf-8"))
q = json.loads(Path("/srv/dataset/queries/queries.json").read_text(encoding="utf-8"))

k0 = next(iter(raw))
print("raw 键样例:", repr(k0))
print("raw 记录字段:", list(raw[k0].keys()))
cid = next(iter(tg["control"]))
print("控制组 qid:", repr(cid))
print("  queries 里该 qid 的 visible:", repr(q[cid]["visible"]))
print("  targets 里该 qid 的 core:", repr(tg["control"][cid]))
expect = f'{q[cid]["visible"]}||{tg["control"][cid].lower()}'
print("  推出的键:", repr(expect))
print("  该键在 raw 中:", expect in raw)
same_visible = sum(1 for v in raw.values() if v.get("rel") == q[cid]["visible"])
print("  raw 中 rel 等于该 visible 的记录数:", same_visible)
print("queries 条数:", len(q), "| targets target/control:", len(tg["target"]), len(tg["control"]))
# 拿 raw 里第一条的 rel/core 反查
print("raw 首条的 rel/core:", repr(raw[k0].get("rel")), repr(raw[k0].get("core")))
PYEOF

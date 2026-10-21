# -*- coding: utf-8 -*-
"""SoM 失败模式统计（远端只读，不落地；供 _audit_som_cores.py 之外的中途巡检）。

回答：enum_fail 是系统性崩还是散布？是否集中在多实例场景（N>=2）？
      num_out_of_range 的形态是什么？
"""
import json
import os
import tempfile

import paramiko

SRC = r"""
import json, os
from collections import Counter, defaultdict

DATA = "/srv/dataset"
q = json.load(open(os.path.join(DATA, "queries", "queries.json"), encoding="utf-8"))
cores = json.load(open("/srv/output/_ordinal_cores_v3.json", encoding="utf-8"))
som = json.load(open("/srv/output/qwen3vl_som_all.json", encoding="utf-8"))

grp = defaultdict(list)
for qid, c in cores.items():
    grp[(os.path.basename(q[qid]["visible"]), c.lower())].append(qid)

rows = []
for k, qids in grp.items():
    v = som.get(qids[0])
    if not v:
        continue
    rows.append({"img": k[0], "core": k[1], "N": len(qids),
                 "nc": v.get("n_cands") or 0, "rs": v.get("reason", "ok"),
                 "qids": sorted(qids)})

tot_items = sum(r["N"] for r in rows)
print("已枚举组", len(rows), "覆盖条目", tot_items, "/ 序数总数 2014")
print("按组 reason:", dict(Counter(r["rs"] for r in rows)))
print("按条目 reason:", dict(Counter(r["rs"] for r in rows for _ in r["qids"])))
print()
print("reason x N 交叉（按组）:")
for (rs, N), n in sorted(Counter((r["rs"], r["N"]) for r in rows).items()):
    print(f"   {rs:18s} N={N}  {n}")
print()
ef = [r for r in rows if r["rs"] == "enum_fail"]
ok = [r for r in rows if r["rs"] != "enum_fail"]
print(f"enum_fail 组 {len(ef)}: N>=2 的 {sum(1 for r in ef if r['N'] >= 2)},"
      f" N==1 的 {sum(1 for r in ef if r['N'] == 1)}")
print(f"非 enum_fail 组 {len(ok)}: N>=2 的 {sum(1 for r in ok if r['N'] >= 2)},"
      f" N==1 的 {sum(1 for r in ok if r['N'] == 1)}")
print()
print("enum_fail 的 n_cands 取值:", dict(Counter(r["nc"] for r in ef)))
print()
print("enum_fail 按核心词 top15:")
for core, n in Counter(r["core"] for r in ef).most_common(15):
    print(f"   {core!r:36s} {n}")
print()
byc = defaultdict(lambda: [0, 0])
for r in rows:
    byc[r["core"]][0] += 1
    if r["rs"] == "enum_fail":
        byc[r["core"]][1] += 1
multi = [(k, v) for k, v in byc.items() if v[0] >= 4]
print(f"出现>=4组的核心词 {len(multi)} 个，按失败率排序 top15:")
for k, v in sorted(multi, key=lambda x: -x[1][1] / x[1][0])[:15]:
    print(f"   {k!r:36s} {v[1]}/{v[0]} = {v[1] / v[0]:.0%}")
print()
oor = [r for r in rows if r["rs"] == "num_out_of_range"]
print(f"num_out_of_range 组 {len(oor)}（覆盖条目 {sum(r['N'] for r in oor)}）:")
for r in oor[:15]:
    v = som[r["qids"][0]]
    print(f"   {r['img']} {r['core']!r:24s} n_cands={r['nc']} "
          f"num={v.get('num')} N={r['N']}")
print()
# 多候选但选择失败/成功：选择环节质量
sel_ok = [r for r in rows if r["rs"] == "ok"]
print("选择环节: 有候选( nc>=2 )的组", sum(1 for r in rows if r["nc"] >= 2),
      "，其中 ok", len(sel_ok))
"""

tf = tempfile.NamedTemporaryFile("w", suffix=".py", delete=False,
                                 encoding="utf-8")
tf.write(SRC)
tf.close()
cli = paramiko.SSHClient()
cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
cli.connect(os.environ["SSH_HOST"], port=int(os.environ["SSH_PORT"]),
            username=os.environ["SSH_USER"], password=os.environ["SSH_PW"],
            timeout=25, banner_timeout=30, auth_timeout=30)
sftp = cli.open_sftp()
sftp.put(tf.name, "/tmp/_som_fail.py")
sftp.close()
_, o, e = cli.exec_command("/srv/venv-locate/bin/python /tmp/_som_fail.py",
                           timeout=180)
print(o.read().decode("utf-8", "replace"))
err = e.read().decode("utf-8", "replace")
if err.strip():
    print("STDERR:", err)
cli.exec_command("rm -f /tmp/_som_fail.py")
cli.close()
os.unlink(tf.name)
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""材料包全面检查：结构 / 提交件 / 哈希 / 一致性 / 命名 / 合规。"""
from __future__ import annotations

import gzip
import hashlib
import json
import re
import zipfile
from pathlib import Path

R = Path(r"d:\Desktop\2026AiC")
PKG = R / "复赛提交材料文件夹" / "AIC-2026-14375345-熬夜炼丹师-复赛-沈嘉宝"
SUBMIT_DIR = PKG / "AIC-2026-14375345-赛题结果文档"
CODE = PKG / "AIC-2026-14375345-代码与数据" / "code27b"
DO = PKG / "AIC-2026-14375345-代码与数据" / "data_outputs"

ok = fail = 0


def chk(cond: bool, msg: str, detail: str = "") -> None:
    global ok, fail
    if cond:
        ok += 1
        print(f"  [OK]   {msg}" + (f"  {detail}" if detail else ""))
    else:
        fail += 1
        print(f"  [!!]   {msg}  {detail}")


print("=" * 72)
print("一、结构完整性")
files = sorted(p for p in PKG.rglob("*") if p.is_file())
print(f"  文件总数：{len(files)}｜总大小：{sum(p.stat().st_size for p in files)/1024/1024:.2f} MB")
for name in ["README.md", "AIC-2026-14375345-技术方案.PDF",
             "AIC-2026-14375345-模型文件", "AIC-2026-14375345-代码与数据",
             "AIC-2026-14375345-赛题结果文档"]:
    chk((PKG / name).exists(), f"存在 {name}")

print("=" * 72)
print("二、命名规范（编号-队名-复赛-队长）")
chk(PKG.name == "AIC-2026-14375345-熬夜炼丹师-复赛-沈嘉宝", "根目录命名")
pref = "AIC-2026-14375345-"
for name in ["技术方案.PDF", "模型文件", "代码与数据", "赛题结果文档"]:
    chk((PKG / f"{pref}{name}").exists(), f"前缀正确 {pref}{name}")

print("=" * 72)
print("三、提交件四项校验")
zp = SUBMIT_DIR / "result.zip"
with zipfile.ZipFile(zp) as zf:
    names = zf.namelist()
    sub = json.loads(zf.read("submission.json").decode("utf-8"))
q = json.loads((DO / "queries.json").read_text("utf-8"))
chk(names == ["submission.json"], "zip 内仅 submission.json", str(names))
chk(len(sub) == len(q), f"条数 {len(sub)} / {len(q)}")
chk(set(sub) == set(q), "键集一致")
echo = sum(1 for k, v in sub.items()
           if any(v[f] != q[k][f] for f in ("visible", "infrared", "depth", "query")))
chk(echo == 0, f"四字段原样回显（偏差 {echo}）")
bad = [k for k, v in sub.items()
       if not (len(v["bbox"]) == 4 and 0 <= v["bbox"][0] < v["bbox"][2] <= 1
               and 0 <= v["bbox"][1] < v["bbox"][3] <= 1)]
chk(not bad, f"bbox 合法（非法 {len(bad)}）")

print("=" * 72)
print("四、哈希一致性")
sha_txt = (SUBMIT_DIR / "SHA256.txt").read_text("utf-8")
want = sha_txt.split()[0]
got = hashlib.sha256(zp.read_bytes()).hexdigest().upper()
chk(want == got, "SHA256.txt 与 result.zip 一致", got[:32] + "…")

print("=" * 72)
print("五、管线输入与结果唯一性")
for f in ["predictions_27b.json", "predictions_ref.json",
          "consensus_pool.json.gz", "queries.json", "env_record.txt"]:
    chk((DO / f).exists(), f"存在 data_outputs/{f}")
zips = [p for p in files if p.suffix.lower() == ".zip"]
chk(len(zips) == 1, f"材料包内唯一 zip（实际 {len(zips)}）",
    str([str(p.relative_to(PKG)) for p in zips]))
with gzip.open(DO / "consensus_pool.json.gz", "rt", encoding="utf-8") as fh:
    pool = json.load(fh)
chk(len(pool) == 32, f"投票池成员数 {len(pool)}")
chk(all(k.startswith("voter_") for k in pool), "池成员全为匿名编号")

print("=" * 72)
print("六、文档与代码一致性")
code_scripts = sorted(p.name for p in (CODE / "scripts").glob("*.py"))
print(f"  code27b/scripts/：{code_scripts}")
for s in ["infer.py", "consensus_vote.py", "pool_dimensions.py", "_bootstrap.py"]:
    chk(s in code_scripts, f"脚本存在 {s}")
rm = (PKG / "README.md").read_text("utf-8")
for s in ["consensus_vote.py", "pool_dimensions.py", "predictions_27b.json",
          "consensus_pool.json.gz"]:
    chk(s in rm, f"README 提及 {s}")
desc = (PKG / "AIC-2026-14375345-代码与数据" / "代码与数据说明.txt").read_text("utf-8")
for s in ["consensus_vote.py", "pool_dimensions.py", "run_infer.log"]:
    if s == "run_infer.log":
        chk(s not in desc, "说明文件已无 run_infer.log 引用")
    else:
        chk(s in desc, f"说明文件提及 {s}")

print("=" * 72)
print("七、合规敏感词扫描")
PAT = re.compile(r"annotate|人眼|人工核对|手动|复核|校对|校准|抽查|干预|挑选|逐图"
                 r"|Grounding DINO|提分|目检|q38|nonord|measure_zero|swap"
                 r"|来源不明|派生|已剔除|[0-9]{6}_[0-9]{3}")
hits = []
for p in files:
    if p.suffix.lower() in (".gz", ".zip", ".pdf", ".json"):
        continue
    try:
        t = p.read_text("utf-8")
    except Exception:
        continue
    for i, ln in enumerate(t.splitlines(), 1):
        m = PAT.search(ln)
        if m:
            hits.append(f"{p.relative_to(PKG)}:{i}  {m.group(0)}")
chk(not hits, "代码与文档无敏感词命中", "" if not hits else str(hits[:5]))

print("=" * 72)
print(f"结论：通过 {ok} 项，异常 {fail} 项")

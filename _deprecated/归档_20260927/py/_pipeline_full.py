# -*- coding: utf-8 -*-
"""端到端自动化：Step1（远端 GD 全量）完成 → 拉取 → 融合链 → 序数焊接
→ 校验 → 打最终提交 zip。

真实接口（已逐一读取源码确认，非记忆）：
  M = GD 三模态基线     预测_输入 : result_gd_3modal_v2_.json（远端预测 = 新基线）
  L2= Locate RGB        predictions_v2_highres_slow.json
  G = GroundingDINO RGB predictions_grounding.json
  Q = 问句槽位引        queries/queries.json
  SEL= 序数人工选择       ordinal_select_result.json

融合链（转写自 @_fusion_g4.py / _fusion_g4_r2r1p.py / _apply_r1pp2.py）：
  G4 : 替换若 iou(L2,G)>=0.5 且 iou(L2,M)<0.5,iou(M,G)<0.5
       → fused=0.6L2+0.4G，需 area<=0.35 且 <=1.25×M.area，否则保留 M
  R2 : G4 中集合/复数类 且 缩小替换 → 回退 M
  R1′: 序数词 且 M.area>0.2 → 采用 L2
  R1″: 纯空间(非序数) 且 M.area>0.35 且 L2.area<=0.35 → 采用 L2
写法：单项函数转写自对应脚本，逐字保留其规则/护栏，绝不做文件名覆写。
"""

import contextlib
import hashlib
import io
import json
import os
import re
import sys
import zipfile
from pathlib import Path

R = Path(__file__).resolve().parent

# ---------------- 路径配置（显式） ----------------
SRC_REMOTE_OUT = "/root/infer/result_gd_3modal_full.json"   # 远端全量预测产物
LOCAL_M = R / "result_gd_3modal_full.json"                  # 新基线（下载到本地）
L2 = R / "predictions_v2_highres_slow.json"
G = R / "predictions_grounding.json"
Q = R / "初赛数据集-基于大模型的多模态视觉理解与推理" / "queries" / "queries.json"
TEST = R / "初赛数据集-基于大模型的多模态视觉理解与推理" / "test.json"
SEL = R / "ordinal_select_result.json"
OUT_JSON = R / "result_gd_3modal_fullseq.json"
OUT_ZIP = R / "result_gd_3modal_fullseq.zip"
DEP = R / "_deprecated"

# 序数焊接：确认接受的 qid（排除 010019_003 TRUNC）
ACCEPTED = [
    "000023_002", "000023_003", "000023_004", "001773_005",
    "010033_004", "012826_001", "014716_003", "016333_002", "016774_001",
]
WRITE_TRUNC = False
TRUNC_SET = {"010019_003"}


# ---------------- 基础几何 ----------------
def iou(a, b):
    """转写 @_fusion_g4.py L12。"""
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0., x2 - x1), max(0., y2 - y1)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 1e-9 else 0.


def area(b):
    """转写 @_fusion_g4.py L20。"""
    return (b[2] - b[0]) * (b[3] - b[1])


def norm(b):
    """转写 @_fusion_g4_r2r1p.py L37。滤波加 x1<x2 下确界。"""
    x1, y1, x2, y2 = b
    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1
    x1, y1 = max(0., min(1., x1)), max(0., min(1., y1))
    x2, y2 = max(0., min(1., x2)), max(0., min(1., y2))
    return [x1, y1, min(1., max(x2, x1 + 1e-3)), min(1., max(y2, y1 + 1e-3))]


ORD = re.compile(r"\b(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|last)\b", re.I)
COLL = re.compile(r"all the|group of|crowd|people|men|women|children|students|tourists|"
                  r"\bpair\b|\btwo\b|\bthree\b|\bfour\b|\bfive\b|\bboth\b", re.I)
SPAT = re.compile(r"leftmost|rightmost|topmost|bottommost|nearest|closest|farthest|furthest|"
                  r"\bleft\b|\bright\b|\btop\b|\bbottom\b|behind|in front|above|below|"
                  r"beside|next to|between|foreground|background|upper|lower", re.I)
CONF = ["000282_004", "000282_008", "002004_001", "002004_002"]


# ---------------- Step1：等待远端全量并下载 ----------------
def remote_wait_download(name: str) -> None:
    """轮询远端直到 9555 条完成且进程结束，拉回 LOCAL_M。"""
    import paramiko
    host = os.environ.get("SSH_HOST", "px-cloud1.matpool.com")
    port = int(os.environ.get("SSH_PORT", "27675"))
    user = os.environ.get("SSH_USER", "root")
    pw = os.environ.get("SSH_PW")
    if not pw:
        raise SystemExit("[ERR] 未设置 SSH_PW")
    def make_cli():
        c = paramiko.SSHClient()
        c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        c.connect(host, port=port, username=user, password=pw,
                  timeout=25, banner_timeout=30, auth_timeout=30)
        return c

    cli = make_cli()
    code = (f"import json,os;p='{SRC_REMOTE_OUT}';"
            f"d=json.load(open(p)) if os.path.exists(p) else {{}};"
            f"print(len(d))")

    def exec_retry(cmd, rtimeout):
        """执行远端命令；瞬态 socket 断连则重连并重试一次。"""
        nonlocal cli
        import paramiko as _pk
        for attempt in range(3):
            try:
                _, so, _ = cli.exec_command(cmd, timeout=rtimeout)
                return so.read().decode("utf-8", "replace")
            except (_pk.SSHException, OSError) as ex:
                print(f"[warn] 执行失败(attempt {attempt+1}): {ex}，重连", flush=True)
                with contextlib.suppress(Exception):
                    cli.close()
                cli = make_cli()
        raise SystemExit(f"[ERR] 连续重试失败，cmd={cmd}")

    last = -1
    while True:
        so = exec_retry(f"python3 -c \"{code}\"", 120)
        n = int(so.strip() or 0)
        po = exec_retry("pgrep -f predict_gd_3modal.py", 60)
        running = bool(po.strip())
        if n != last:
            print(f"[monitor] {n}/9555 running={running}", flush=True)
            last = n
        if n >= 9555 or not running:
            print(f"[monitor] 完成: {n}/9555 running={running}", flush=True)
            break
        time.sleep(120 if running else 30)
    if n < 9555 and not running:
        print(f"[ERR] 远端进程已结束但只有 {n}/9555 条——不下载，避免半成品", flush=True)
        raise SystemExit(2)
    print(f"[sftp] 下载 {SRC_REMOTE_OUT} -> {LOCAL_M}", flush=True)
    for attempt in range(3):
        try:
            sftp = cli.open_sftp()
            sftp.get(SRC_REMOTE_OUT, str(LOCAL_M))
            sftp.close()
            break
        except OSError as ex:
            print(f"[warn] 下载失败(attempt {attempt+1}): {ex}，重连", flush=True)
            with contextlib.suppress(Exception):
                cli.close()
            cli = make_cli()
    cli.close()
    print(f"[sftp] 已下载 {LOCAL_M.stat().st_size} 字节", flush=True)


import time  # noqa: E402


# ---------------- 融合链（逐字转写自源码） ----------------
def step_g4(m, l2, g):
    """转写 @_fusion_g4.py。"""
    out = {k: dict(m[k]) for k in m}
    rep = 0
    guards = 0
    for k in keys_of(m, l2, g):
        l2b, mb, gb = l2[k]["bbox"], m[k]["bbox"], g[k]["bbox"]
        i_lm, i_lg, i_mg = iou(l2b, mb), iou(l2b, gb), iou(mb, gb)
        if i_lg >= 0.5 and i_lm < 0.5 and i_mg < 0.5:
            fused = norm([0.6 * l2b[i] + 0.4 * gb[i] for i in range(4)])
            af, ab = area(fused), area(mb)
            if af <= 0.35 and af <= 1.25 * ab:
                out[k] = dict(m[k])
                out[k]["bbox"] = [round(float(v), 6) for v in fused]
                rep += 1
            else:
                guards += 1
    print(f"[G4] 替换 {rep}  护栏拦下 {guards}")
    return out


def keys_of(m, l2, g):
    """G4 原始脚本只处理三交集，这里同样但绝不丢条目。"""
    inter = sorted(set(m) & set(l2) & set(g))
    return inter


def step_r2r1p(fused, m, l2, q):
    """转写 @_fusion_g4_r2r1p.py：R1′ 优先，R2 其次。"""
    out = {}
    applied = {}
    for k, rec in fused.items():
        b = rec["bbox"]
        rule = None
        if k in l2 and ORD.search(q[k]["query"]) and area(m[k]["bbox"]) > 0.2:
            b, rule = norm(l2[k]["bbox"]), "R1p_ordinal"
        elif (iou(fused[k]["bbox"], m[k]["bbox"]) < 0.999
              and area(fused[k]["bbox"]) < area(m[k]["bbox"])
              and COLL.search(q[k]["query"])):
            b, rule = m[k]["bbox"], "R2_collective"
        entry = dict(rec); entry["bbox"] = [round(float(v), 6) for v in b]
        out[k] = entry
        if rule:
            applied[rule] = applied.get(rule, 0) + 1
    print(f"[R2R1'] {json.dumps(applied)}")
    return out


def step_r1pp2(cur, base, l2, q):
    """转写 @_apply_r1pp2.py：R1″ 精炼版。"""
    cand = [k for k in cur
            if (not ORD.search(q[k]["query"])) and SPAT.search(q[k]["query"])
            and area(base[k]["bbox"]) > 0.35]
    adopt = [k for k in cand if area(l2[k]["bbox"]) <= 0.35]
    out = {}
    for k, rec in cur.items():
        entry = dict(rec)
        if k in adopt:
            entry["bbox"] = [round(float(v), 6) for v in norm(l2[k]["bbox"])]
        out[k] = entry
    print(f"[R1\"] 候选 {len(cand)} → 采纳 {len(adopt)}")
    return out


# ---------------- 序数焊接 ----------------
def weld_ordinal(fused, sel, test):
    """转写自 @_weld_ordinal.py 的 ACCEPTED 规则。"""
    data = {k: dict(v) for k, v in fused.items()}
    before = len(data)
    assert before == 9555 and set(data) == set(test), "源与 test.json 键不一致"
    qids = list(ACCEPTED)
    if WRITE_TRUNC:
        qids.extend(TRUNC_SET)
    for qid in qids:
        entry = sel.get(qid)
        if entry and entry.get("bbox") is not None and qid in data:
            data[qid]["bbox"] = [round(float(v), 6) for v in entry["bbox"]]
    assert len(data) == before and set(data) == set(test), "焊接后键集/条目数异常"
    return data


# ---------------- 校验 + 打 zip ----------------
def verify(final):
    """键集==test，9555，bbox 合法，字段齐全，大框计数。"""
    test = json.loads(TEST.read_text(encoding="utf-8"))
    assert len(final) == len(test) == 9555, f"条目数 {len(final)} vs {len(test)}"
    assert set(final) == set(test), "键集不一致"
    FIELDS = {"visible", "infrared", "depth", "query", "bbox"}
    bad = [k for k, v in final.items()
           if set(v) != FIELDS
           or not (0.0 <= v["bbox"][0] < v["bbox"][2] <= 1.0
                   and 0.0 <= v["bbox"][1] < v["bbox"][3] <= 1.0)]
    assert not bad, f"字段/bbox 非法: {bad[:5]}"
    big = sum(1 for v in final.values() if area(v["bbox"]) > 0.35)
    print(f"[verify] 9555 键集一致，bbox 全部合法，大框(>0.35)={big}")


def build_zip(final):
    """内部单一文件 result_gd_3modal.json，转写 @predict.py L127。"""
    with zipfile.ZipFile(OUT_ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
        data = json.dumps(final, ensure_ascii=False).encode("utf-8")
        zf.writestr("result_gd_3modal.json", data)
        sha = hashlib.sha256(data).hexdigest().upper()
    OUT_JSON.write_text(json.dumps(final, ensure_ascii=False), encoding="utf-8")
    print(f"[zip] {OUT_ZIP.name} 内部 result_gd_3modal.json SHA256={sha}")
    print(f"[out] {OUT_JSON.name} SHA256={hashlib.sha256(OUT_JSON.read_bytes()).hexdigest().upper()}")


def archive_lock(name: str) -> None:
    DEP.mkdir(exist_ok=True)
    src = R / name
    if src.exists():
        dest = DEP / (Path(name).stem + ".prev" + Path(name).suffix)
        dest.write_bytes(src.read_bytes())
        print(f"[dep] 归档 {name} -> {dest.name}")


# ---------------- main runner ----------------
def run_fusion(base_path, name="<base>"):
    """以给定基线跑完整链路（dry-run 用旧 v2 基线校验）。"""
    m = json.loads(base_path.read_text(encoding="utf-8"))
    l2 = json.loads(L2.read_text(encoding="utf-8"))
    g = json.loads(G.read_text(encoding="utf-8"))
    q = json.loads(Q.read_text(encoding="utf-8"))
    sel = json.loads(SEL.read_text(encoding="utf-8"))
    n = len(m)
    print(f"[{name}] 基线 n={n}")
    assert n == 9555, f"基线条目 {n} != 9555"

    fused = step_g4(m, l2, g)
    fused = step_r2r1p(fused, m, l2, q)
    fused = step_r1pp2(fused, m, l2, q)
    final = weld_ordinal(fused, sel, json.loads(TEST.read_text(encoding="utf-8")))
    verify(final)
    build_zip(final)
    return final


def main(argv) -> int:
    """入口：--src 指定基线(缺省用远端完成后下载)；--no-download 跳过 Step1。"""
    src = None
    if "--no-download" in argv:
        src = LOCAL_M
    for a in argv:
        if a.startswith("--src="):
            src = Path(a.split("=", 1)[1])
    if src is None:
        if not LOCAL_M.exists():
            remote_wait_download("full")
        src = LOCAL_M
    if not src.exists():
        raise SystemExit(f"[ERR] 基线不存在: {src}")
    run_fusion(src, src.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
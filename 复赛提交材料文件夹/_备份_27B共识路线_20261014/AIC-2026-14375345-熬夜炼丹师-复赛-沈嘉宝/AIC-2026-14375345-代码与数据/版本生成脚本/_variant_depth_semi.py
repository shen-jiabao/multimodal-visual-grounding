# -*- coding: utf-8 -*-
"""深度序数轴变体：把"上下/左右"换成"远近"，在候选表上按深度取第 k 个。

机制（与 y 轴同族，属**修正输入**而非放宽判据）
- 序号仍由问句文本推算（`_fuse_qwen.det_ordinal`），只是排序轴改为深度：对每个候选框
  在 16 位深度图上取**有效像素中位数**作为该框的深度代表值，近→远排序，NFR 取第 k、
  FRN 取倒数第 k。垂直轴（y）与深度轴（z）在文本上同族，故先验可直接沿用（y 轴实测
  +30%、截断层 +21.2%）。
- 深度计算属**数据层**：`_fuse_qwen` 只接受 `zkeys` 作为排序键，自身不读图——保持
  "判据只有一处实现"。

守卫（宁可不动，不猜）
- 深度图整幅无效、或有效深度框不足 2 个 → 放弃该条（排序无意义，猜了就是主动改错）
- 深度轴约束见 `_fuse_qwen.DEPTH_GUARD`（"closest" 不指远近的三类句式）

基线取已交 0.7598 的截断层产物；两层的条目集不相交（深度轴条目不含水平/垂直线索，
故不在截断组内），可线性叠加。

用法
    python -B _variant_depth_semi.py            # 只评估
    python -B _variant_depth_semi.py --zip      # 另出提交件
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
import sys
import zipfile
from pathlib import Path

R = Path(r"d:\Desktop\2026AiC")
sys.path.insert(0, str(R))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

import _fuse_qwen as F  # noqa: E402
import _sub_path as SP  # noqa: E402

SEMI = R / "复赛数据集-基于大模型的多模态视觉理解与推理"
SUB = R / "submission_zips" / "semifinal"
CORES = R / "_ordinal_cores_r2.json"
BASE_STEM = "result_semi_enum_trunc"
AXES = ("x", "y", "z")
MIN_MM = 300        # 与主管线一致：<300mm 视为无效深度
MIN_VALID = 2       # 有效深度框少于该值即放弃该条


def read_depth(rel: str) -> np.ndarray | None:
    """读 16 位深度图（cv2.imdecode 处理中文路径）。"""
    raw = np.fromfile(SEMI / rel, dtype=np.uint8)
    d = cv2.imdecode(raw, cv2.IMREAD_UNCHANGED)
    if d is None:
        return None
    return cv2.cvtColor(d, cv2.COLOR_BGR2GRAY) if d.ndim == 3 else d


def box_depth(depth: np.ndarray, box: list) -> float:
    """框内有效深度的中位数（mm）；无有效像素返回 inf（排到最远）。"""
    h, w = depth.shape[:2]
    x1, y1, x2, y2 = box
    xs, xe = max(int(x1 * w), 0), min(max(int(x2 * w), int(x1 * w) + 1), w)
    ys, ye = max(int(y1 * h), 0), min(max(int(y2 * h), int(y1 * h) + 1), h)
    patch = depth[ys:ye, xs:xe].astype(np.float64)
    v = patch[patch >= MIN_MM]
    return float(np.median(v)) if v.size else float("inf")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=BASE_STEM)
    ap.add_argument("--zip", action="store_true")
    ap.add_argument("--stem", default="result_semi_depth_axis")
    ap.add_argument("--show", type=int, default=12)
    a = ap.parse_args()

    q = json.loads((SEMI / "queries" / "queries.json").read_text(encoding="utf-8"))
    cores = json.loads(CORES.read_text(encoding="utf-8"))
    s8b = json.loads((R / "_semi_qwen3vl_som_all.json").read_text(encoding="utf-8"))
    base = json.loads(SP.find(a.base).read_text(encoding="utf-8"))

    # 先按文本筛出深度轴条目（不读图），再逐条算深度
    cands = []
    for qid in sorted(cores):
        rec = s8b.get(qid) or {}
        if not rec.get("bbox_all"):
            continue
        ax, ori = F.ordinal_axis(q[qid]["query"], AXES)
        if ax != "z":
            continue
        cands.append((qid, rec, ori))
    print(f"深度轴条目（文本可判、有候选表）：{len(cands)} 条")

    depth_cache: dict = {}
    chg, skip = [], 0
    print(f"\n{'qid':12s} {'n':>3s} {'k':>2s} {'ori':>4s} {'有效':>4s}  深度(mm, 排序后)")
    for qid, rec, ori in cands:
        rel = q[qid]["visible"].replace("/visible/", "/depth/")
        if rel not in depth_cache:
            depth_cache[rel] = read_depth(rel)
        depth = depth_cache[rel]
        if depth is None:
            skip += 1
            continue
        boxes = rec["bbox_all"]
        zk = [box_depth(depth, b) for b in boxes]
        if sum(1 for z in zk if z != float("inf")) < MIN_VALID:
            skip += 1
            print(f"{qid:12s} {len(boxes):3d} {'-':>2s} {ori:>4s} {0:4d}  ← 深度图无效，放弃")
            continue
        k = F.ordinal_index(q[qid]["query"])
        new, rule = F.det_ordinal("det", rec, q[qid]["query"], AXES, zk)
        old = base[qid]["bbox"]
        changed = new is not None and F.iou(new, old) < F.IOU_KEEP
        nvalid = sum(1 for z in zk if z != float("inf"))
        if changed:
            chg.append((qid, new, old, len(boxes), k, ori, zk, rule))
        if len(chg) <= a.show or changed:
            zs = " ".join(f"{x:.0f}" if x != float("inf") else "inf"
                          for x in sorted(zk))
            print(f"{qid:12s} {len(boxes):3d} {k if k else '-':>2} {ori:>4s} {nvalid:4d}  "
                  f"{zs}  {'★改' if changed else ('=' if rule=='ORD_DET' else rule)}")

    print(f"\n★ 会改变 {len(chg)} 条 | 放弃 {skip} 条 | Δ 上限 {len(chg)/5690:.4f}"
          f" | ×21~30% 期望 ≈ {len(chg)*0.21/5690:.4f}~{len(chg)*0.30/5690:.4f}")
    for qid, new, old, n, k, ori, zk, rule in chg:
        print(f"    {qid} n={n} k={k} {ori}  旧{[round(x,3) for x in old]} → "
              f"新{[round(x,3) for x in new]}  | {q[qid]['query'][:48]}")

    if a.zip and chg:
        out = {k2: dict(v) for k2, v in base.items()}
        for qid, new, _o, _n, _k, _r, _z, _rl in chg:
            out[qid]["bbox"] = new
        (SP.out_path(a.stem, ".json")).write_text(
            json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        p = SP.out_path(a.stem, ".zip")
        payload = json.dumps(out, ensure_ascii=False,
                             separators=(",", ":")).encode("utf-8")
        with zipfile.ZipFile(p, "w", zipfile.ZIP_DEFLATED) as zf:
            zi = zipfile.ZipInfo(f"{a.stem}.json", date_time=(1980, 1, 1, 0, 0, 0))
            zi.compress_type = zipfile.ZIP_DEFLATED
            zf.writestr(zi, payload)
        print(f"\n已写 {p.name}  SHA256 "
              f"{hashlib.sha256(p.read_bytes()).hexdigest().upper()}")
        r = subprocess.run([sys.executable, "-B", str(R / "_validate_semi.py"), str(p)],
                           capture_output=True, text=True)
        print("  " + "\n  ".join(ln for ln in r.stdout.splitlines()
                                  if ln.strip().startswith(("[", "✓", "✗"))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

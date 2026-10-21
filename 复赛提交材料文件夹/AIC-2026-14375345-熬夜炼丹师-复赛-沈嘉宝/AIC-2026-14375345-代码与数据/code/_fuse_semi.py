# -*- coding: utf-8 -*-
"""复赛融合：Qwen3-VL-8B（direct + SoM）与 LocateAnything-3B 按互印证规则合成提交件。

判据复用：`iou` / `norm` / `area` / `valid_cand` 一律从 `_fuse_qwen` 导入，
不在此重复实现（本项目吃过"判据两处、各自漂移"的亏）。

取框规则（对 5 条多框条目回测后确定，见对话记录）：
  1) LA 多框先去重（IoU > DEDUP_IOU 视为同框重复）
     —— 消灭 `000512_002` 那类"同一框重复 41 次刷满 max_new_tokens"的退化
  2) 去重后只剩 1 框 → 采用
  3) 去重后仍多框 → 取与「对照答案」IoU 最高者
     对照答案：序数问句取 8B 的 **SoM** 框（实测 LA 与 8B.som 对齐、与 8B.direct 不对齐）
               非序数取 8B 的 **direct** 框
  4) 最高 IoU < LA_MIN_IOU → LA 该条不予采信，交回 8B
  5) 无对照答案（8B 该条无框）→ 多框时取面积最大者，并标注规则名以便回溯
  6) LA 无结果 / 无框 → 用 8B
  7) 两者皆无框 → 记 FILL_* 并在报告中计数（不静默填充）

用法：
    python _fuse_semi.py --d8b qwen3vl_all.json --s8b qwen3vl_som_all.json \
        --la locate_bbox_all.json --out result_fuse_semi.json --zip
"""
from __future__ import annotations
# ===== 提交包路径适配（打包时自动注入；算法逻辑未改动）=====
import os as _os
import sys as _sys
from pathlib import Path as _Path
_HERE = _Path(__file__).resolve().parent
if str(_HERE) not in _sys.path:
    _sys.path.insert(0, str(_HERE))
ROOT = _Path(_os.environ.get("AIC_ROOT") or _HERE.parent)
ROOT_S = str(ROOT)
# ===== 注入结束 =====

import argparse
import json
import sys
import zipfile
from collections import Counter
from pathlib import Path

R = Path(ROOT_S)
sys.path.insert(0, str(R))

from _fuse_qwen import area, iou, norm, ordered  # noqa: E402

SEMI = R / "复赛数据集-基于大模型的多模态视觉理解与推理"
DEDUP_IOU = 0.99      # 两框 IoU 超过此值视为同框重复
LA_MIN_IOU = 0.30     # LA 框与对照答案的最低印证阈值
FIELDS = {"bbox", "depth", "infrared", "query", "visible"} 


def dedup(boxes: list) -> list:
    """去掉近似重复的框，保持原顺序（首个出现者保留）。

    用 `ordered` 而非 `valid_cand`：去重只关心"是否同一个框"，与面积无关。
    若用 `valid_cand`（含 MAX_AREA=0.35 上限），`000512_002`「细白横线」这类
    合法的整幅宽框会被全部滤掉 → 最终输出 null，反而必然丢分。
    """
    out = []
    for b in boxes:
        if not ordered(b):
            continue
        if any(iou(b, o) > DEDUP_IOU for o in out):
            continue
        out.append(b)
    return out


def pick_la(la_boxes: list, ref, is_ord: bool) -> tuple[object, str]:
    """按互印证规则从 LA 的框里挑一个；返回 (框或 None, 规则名)。"""
    boxes = dedup(la_boxes)
    if not boxes:
        return None, "LA_EMPTY"
    if len(boxes) == 1:
        return norm(boxes[0]), "LA_SINGLE"
    if ref is None:
        # 无对照答案：取面积最大者（中立兜底），规则名单独标记便于回溯
        best = max(boxes, key=area)
        return norm(best), "LA_MULTI_NOREF_MAXAREA"
    scored = sorted(((iou(b, ref), b) for b in boxes), key=lambda t: -t[0])
    best_iou, best = scored[0]
    if best_iou < LA_MIN_IOU:
        return None, "LA_MULTI_NO_AGREE"
    return norm(best), ("LA_MULTI_AGREE_SOM" if is_ord else "LA_MULTI_AGREE_DIRECT")


def fuse_8b_only(queries: dict, cores: set, d8b: dict,
                 s8b: dict) -> tuple[dict, Counter, list]:
    """仅用 8B 产物生成提交：**序数优先 SoM 框，其余用 direct 框**。

    对应初赛的 SoM 用法（序数走 SoM、非序数走 direct）。用途是 LA 跑完前的
    安全基线提交——避免"要等 6–8 小时才有一份可交产物"。
    """
    data, stats, nobox = {}, Counter(), []
    for qid, rec in queries.items():
        is_ord = qid in cores
        d_box = (d8b.get(qid) or {}).get("bbox")
        s_box = (s8b.get(qid) or {}).get("bbox")
        if is_ord and ordered(s_box):
            cand, rule = norm(s_box), "8B_SOM"
        elif ordered(d_box):
            cand = norm(d_box)
            rule = "8B_DIRECT_ORD_FALLBACK" if is_ord else "8B_DIRECT"
        else:
            cand, rule = None, "FILL_NONE"
        if cand is None:
            nobox.append(qid)
        data[qid] = {
            "visible": rec["visible"],
            "infrared": rec["infrared"],
            "depth": rec["depth"],
            "query": rec["query"],
            "bbox": cand,
        }
        stats[rule] += 1
    return data, stats, nobox


def fuse(queries: dict, cores: set, d8b: dict, s8b: dict,
         la: dict) -> tuple[dict, Counter, list]:
    """生成提交数据；返回 (data, 规则计数, 无框条目清单)。"""
    data, stats, nobox = {}, Counter(), []
    for qid, rec in queries.items():
        is_ord = qid in cores
        d_box = (d8b.get(qid) or {}).get("bbox")
        s_box = (s8b.get(qid) or {}).get("bbox")
        # 对照答案：序数优先 SoM 框，缺失则退回 direct 框
        ref = (s_box if is_ord else d_box) or d_box
        la_boxes = (la.get(qid) or {}).get("bbox_all") or []
        if not la_boxes and (la.get(qid) or {}).get("bbox"):
            la_boxes = [la[qid]["bbox"]]

        cand, rule = pick_la(la_boxes, ref, is_ord)
        if cand is None:
            # 规则 4/6：LA 不可采信或不可用 → 交回 8B
            cand = norm(ref) if ordered(ref) else None
            rule = "KEEP_8B_" + rule
        if cand is None:
            nobox.append(qid)
            rule = "FILL_NONE"
        data[qid] = {
            "visible": rec["visible"],
            "infrared": rec["infrared"],
            "depth": rec["depth"],
            "query": rec["query"],
            "bbox": cand,
        }
        stats[rule] += 1
    return data, stats, nobox


def validate(data: dict, queries: dict) -> list:
    """硬约束校验，返回问题清单（空即通过）。"""
    bad = []
    if len(data) != len(queries):
        bad.append(f"条数 {len(data)} != queries {len(queries)}")
    if set(data) != set(queries):
        bad.append("键集与 queries 不一致")
    nf = nbad = nnone = nnan = 0
    for qid, v in data.items():
        if set(v) != FIELDS:
            nf += 1
        b = v.get("bbox")
        if b is None:
            nnone += 1
            continue
        if not (isinstance(b, list) and len(b) == 4):
            nbad += 1
            continue
        if not all(isinstance(x, (int, float)) and x == x for x in b):
            nnan += 1
            continue
        x1, y1, x2, y2 = b
        if not (0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0):
            nbad += 1
    for tag, n in (("字段集异常", nf), ("bbox 非四元/越界/逆序", nbad),
                   ("bbox 含 NaN", nnan), ("bbox 为 null", nnone)):
        if n:
            bad.append(f"{tag}: {n}")
    return bad


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--d8b", required=True, help="8B direct 产物 qwen3vl_all.json")
    ap.add_argument("--s8b", required=True, help="8B SoM 产物 qwen3vl_som_all.json")
    ap.add_argument("--la", help="LA 产物 locate_bbox_all.json；"
                                 "不给则以「仅 8B」模式产出基线提交")
    ap.add_argument("--ir", help="红外补轮产物 retry_ir.json；"
                                 "仅对「其他来源无框」的条目生效")
    ap.add_argument("--retry19", help="8B 换模态/换问法补跑产物 retry19.json；"
                                      "优先级低于 --ir，仅补仍未填的")
    ap.add_argument("--swap", help="定向替换来源（如 retry_thermal.json）；"
                                   "作用于**已有框**的条目，需配 --swap-qids")
    ap.add_argument("--swap-qids", help="允许被替换的 qid 清单（JSON 数组）")
    ap.add_argument("--fill-full", action="store_true",
                    help="仍无框者用整幅框 [0,0,1,1] 兜底，使提交件不含 null")
    ap.add_argument("--queries", default=str(SEMI / "queries" / "queries.json"))
    ap.add_argument("--cores", default=str(R / "_ordinal_cores_r2.json"))
    ap.add_argument("--out", required=True, help="输出 JSON 路径")
    ap.add_argument("--zip", action="store_true", help="同时打包 zip")
    a = ap.parse_args()

    queries = json.loads(Path(a.queries).read_text(encoding="utf-8"))
    cores = set(json.loads(Path(a.cores).read_text(encoding="utf-8")))
    d8b = json.loads(Path(a.d8b).read_text(encoding="utf-8"))
    s8b = json.loads(Path(a.s8b).read_text(encoding="utf-8"))
    print(f"queries {len(queries)} | 序数 {len(cores)} | "
          f"8B.direct {len(d8b)} | 8B.som {len(s8b)}", end="")
    if a.la:
        la = json.loads(Path(a.la).read_text(encoding="utf-8"))
        print(f" | LA {len(la)}")
        data, stats, nobox = fuse(queries, cores, d8b, s8b, la)
    else:
        print(" | 模式: 仅 8B（LA 未提供）")
        data, stats, nobox = fuse_8b_only(queries, cores, d8b, s8b)

    print("\n=== 规则计数 ===")
    for k, v in stats.most_common():
        print(f"  {k:28s} {v}")

    # 补充来源：仅填「其他来源无框」的条目，不覆盖已有结论。
    # 顺序即优先级——先到的先占位，后面的只补仍未填的。
    fills = ((a.ir, "IR_FILL"), (a.retry19, "QWEN_RETRY"))
    for path, label in fills:
        if not path:
            continue
        src = json.loads(Path(path).read_text(encoding="utf-8"))
        n = 0
        for qid, v in src.items():
            if qid in data and data[qid]["bbox"] is None \
                    and ordered(v.get("bbox")):
                data[qid]["bbox"] = norm(v["bbox"])
                n += 1
        print(f"  {label:28s} {n}   ← {Path(path).name}")

    # 定向替换：作用于**已有框**的条目（与上面的"补空"语义不同）
    if a.swap:
        sw = json.loads(Path(a.swap).read_text(encoding="utf-8"))
        allow = (set(json.loads(Path(a.swap_qids).read_text(encoding="utf-8")))
                 if a.swap_qids else set(sw))
        n_sw = 0
        for qid, v in sw.items():
            if qid not in allow or qid not in data:
                continue
            if ordered(v.get("bbox")):
                data[qid]["bbox"] = norm(v["bbox"])
                n_sw += 1
        print(f"  {'SWAP':28s} {n_sw}   ← {Path(a.swap).name}"
              f"（限定 {len(allow)} 条）")

    # 整幅兜底：使提交件不含 null（初赛真实提交件亦无 null）
    n_ff = 0
    if a.fill_full:
        for qid, v in data.items():
            if v["bbox"] is None:
                v["bbox"] = [0.0, 0.0, 1.0, 1.0]
                n_ff += 1
        print(f"  {'FILL_FULL':28s} {n_ff}   ← 整幅框兜底")

    nobox = [k for k, v in data.items() if v["bbox"] is None]

    bad = validate(data, queries)
    print("\n=== 硬约束校验 ===")
    if bad:
        for b in bad:
            print("  ✗", b)
    else:
        print("  全部通过")

    if nobox:
        print(f"\n警告：{len(nobox)} 条两模型皆无框，bbox 为 null：{nobox[:8]}")

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    print(f"\n已写 {out}（{out.stat().st_size/1e6:.2f} MB）")

    if a.zip:
        z = out.with_suffix(".zip")
        with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
            zi = zipfile.ZipInfo(out.name, date_time=(1980, 1, 1, 0, 0, 0))
            zf.writestr(zi, out.read_bytes())
        print(f"已写 {z}（{z.stat().st_size/1e6:.2f} MB）")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())

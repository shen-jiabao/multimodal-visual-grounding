# -*- coding: utf-8 -*-
"""Locate 全量结果 与 已锁定基线 择优融合（保底不劣于现有提交）。

输入（接口已核实）：
  locate_bbox_all.json   Locate 全量输出：{qid: {bbox(单框|None), n_all_boxes, raw}}
  result_gd_3modal_fullseq.json  已锁定基线（9555，含 fields）
  queries.json                   问句（用于类别判定）
输出：
  result_locate_merge.json + result_locate_merge.zip（内部 result_locate_merge.json）

融合规则（保守，优先级从高到低）：
  R0 基线兜底：Locate 无框(None) → 保留基线框。
  R1 面积护栏：Locate 框面积 >0.35（合并大框）→ 保留基线（与项目记忆护栏一致）。
  R2 不一致保护：Locate 框与基线 IoU 极低(<0.2) 且 Locate 框面积相对基线 <50% 或 >150%
     → 视为异常，回退基线。避免"框漂移"覆盖掉基线的高质量预测。
  R3 其余（Locate 有框、面积合法、形状接近）→ 采用 Locate。
统计：每问句记录采用了谁，汇总替换数/回退数/类别分布。
"""
import hashlib
import json
import re
import zipfile
from pathlib import Path

R = Path(__file__).resolve().parent
LOC = R / "locate_bbox_all.json"              # 全量 Locate 输出（拉回本地）
BASE = R / "result_gd_3modal_fullseq.json"    # 已锁定基线
QUERIES = R / "初赛数据集-基于大模型的多模态视觉理解与推理" / "queries" / "queries.json"
OUT_JSON = R / "result_locate_merge.json"
OUT_ZIP = R / "result_locate_merge.zip"

ORD = re.compile(r"\b(first|second|third|fourth|fifth|sixth|seventh|"
                 r"eighth|ninth|tenth|last)\b", re.I)
# 极值空间词：Locate 定位更对（leftmost/rightmost 轿车等），不归入"序数回退基线"
EXTREME = re.compile(r"\b(leftmost|rightmost|topmost|bottommost)\b", re.I)
SPAT = re.compile(r"\b(behind|in front|next to|near |below|above|beneath|"
                  r"under |closest|farthest|furthest|between|left of|right of|"
                  r"front|back|foreground|background|upper|lower|far |near)\b", re.I)
AREA_LIMIT = 0.35
IoU_LOW = 0.2
# 护栏样本：人工确认的关键样本，任何情况下不得改动（项目硬约束）。
GUARD_SAMPLES = {"000282_004", "000282_008", "002004_001", "002004_002"}
# 区域/场景类问句：Locate 框大是正确语义（湖/草地/天空/地面/围栏等），豁免 R1/R2。
REGION_RE = re.compile(
    r"\b(lake|water|pond|river|sea|ocean|sky|cloud|grass|lawn|ground|floor|"
    r"road|street|sidewalk|path|field|sand|snow|wall|fence|barrier|railing|"
    r"bamboo|tree|door|area|region|scenery|background|foreground|landscape)\b",
    re.I)
# 连续结构类：长条/护栏/栏杆等 Locate 单框易只圈一段 → 无条件保留基线。
STRUCTURE_RE = re.compile(
    r"\b(railing|fence|fencing|barrier|guardrail|handrail|railing)\b", re.I)
# 近景类问句："最近/靠近"的对象占画面大是正确语义，豁免 R2 放大回退（不含序数）。
NEAR_RE = re.compile(r"\b(closest|nearest|near (the|a)|in the (front|foreground))\b",
                     re.I)


def iou(a, b):
    if not a or not b:
        return 0.0
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0., x2 - x1), max(0., y2 - y1)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 1e-9 else 0.0


def area(b):
    return (b[2] - b[0]) * (b[3] - b[1])


def norm(b):
    x1, y1, x2, y2 = b
    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1
    x1, y1 = max(0., min(1., x1)), max(0., min(1., y1))
    x2, y2 = max(0., min(1., x2)), max(0., min(1., y2))
    return [x1, y1, min(1., max(x2, x1 + 1e-3)), min(1., max(y2, y1 + 1e-3))]


def main() -> int:
    if not LOC.exists():
        print(f"[ERR] {LOC} 不存在（先拉回全量结果）")
        return 2
    loc = json.loads(LOC.read_text(encoding="utf-8"))
    base = json.loads(BASE.read_text(encoding="utf-8"))
    queries = json.loads(QUERIES.read_text(encoding="utf-8"))
    assert set(base) == set(queries), "基线/问句键不一致"
    assert len(base) == 9555

    out = {}
    stat = {"use_locate": 0, "back_to_base": 0}
    stat_r = {"R0_noframe": 0, "R1_bigarea": 0, "R2_mismatch": 0,
              "R3_ordinal_keep_base": 0}
    stat_cat = {}
    multi_notes = []          # 多框样本记账：{qid, n_boxes, used_locate}
    for k, rec in base.items():
        q = queries[k]["query"]
        lb = loc.get(k, {}).get("bbox")
        nb = loc.get(k, {}).get("n_all_boxes", 1)
        bb = rec["bbox"]
        entry = dict(rec)
        if k in GUARD_SAMPLES:
            # 护栏硬约束：不得改动，直接保留基线
            entry["bbox"] = bb
            out[k] = entry
            stat["back_to_base"] += 1
            continue
        if STRUCTURE_RE.search(q):
            # 连续结构类（栏杆/护栏/围栏等）：Locate 单框易只圈一段 → 保留基线
            entry["bbox"] = bb
            out[k] = entry
            stat["back_to_base"] += 1
            continue
        use_locate = True
        is_region = bool(REGION_RE.search(q))
        is_near = bool(NEAR_RE.search(q))
        is_ordspat = bool(ORD.search(q) or SPAT.search(q))
        if ORD.search(q):
            # 序数类（first/second/.../last）：Locate 单框推理未命中正确实例
            # （全量单框不走枚举取第N），基线序数层已验证 → 保留基线。
            # 极值空间词(leftmost 等)不在此列，走正常择优。
            use_locate = False
            stat_r["R3_ordinal_keep_base"] += 1
        elif lb is None:
            use_locate = False
            stat_r["R0_noframe"] += 1                      # R0 兜底
        elif area(lb) > AREA_LIMIT and not is_region:
            use_locate = False
            stat_r["R1_bigarea"] += 1                      # R1 面积护栏（区域类豁免）
        elif iou(lb, bb) < IoU_LOW and not is_region:
            al, ab = area(lb), area(bb)
            # Locate 优势=框更准更细（面积常显著<基线），缩小是修正方向，
            # 仅当"面积显著放大(>1.5x基线)"才判漂移回退——放大才是异常。
            # 豁免：区域/场景类、近景类、序数/空间类（Locate 修正序数是核心价值）。
            if al > 1.5 * ab and not is_near and not is_ordspat:
                stat_r["R2_mismatch"] += 1                 # R2 放大漂移
                use_locate = False
            else:
                use_locate = True                          # 缩小/接近：保留 Locate
        else:
            use_locate = True                              # R3 正常

        entry["bbox"] = [round(float(v), 6) for v in norm(lb)] if use_locate else bb
        out[k] = entry
        stat["use_locate" if use_locate else "back_to_base"] += 1
        # 多框记账：全量只取首框，此处单独标记供抽查，不静默吞掉
        if nb and nb > 1:
            multi_notes.append({"qid": k, "n_boxes": nb,
                                "used_locate": use_locate})
        # 类别统计
        cat = "ord/spat" if (ORD.search(q) or SPAT.search(q)) else "other"
        stat_cat.setdefault(cat, {"total": 0, "locate": 0})
        stat_cat[cat]["total"] += 1
        if use_locate:
            stat_cat[cat]["locate"] += 1

    assert len(out) == 9555 and set(out) == set(base), "键集异常"
    bad = sum(1 for v in out.values()
              if not (0 <= v["bbox"][0] < v["bbox"][2] <= 1
                      and 0 <= v["bbox"][1] < v["bbox"][3] <= 1))
    assert bad == 0, f"含非法 bbox {bad}"

    OUT_JSON.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    with zipfile.ZipFile(OUT_ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("result_locate_merge.json",
                    json.dumps(out, ensure_ascii=False).encode("utf-8"))

    print(f"融合完成 {len(out)} 条")
    print(f"  采用 Locate: {stat['use_locate']}  回退基线: {stat['back_to_base']}")
    print(f"  回退原因: {stat_r}")
    nmulti = len(multi_notes)
    if nmulti:
        nmu_loc = sum(1 for m in multi_notes if m["used_locate"])
        print(f"  多框样本(n_all_boxes>1): {nmulti} 个，其中采用首框Locate {nmu_loc} "
              f"/ 回退基线 {nmulti-nmu_loc}  (清单见 _merge_multibox_notes.json)")
        (R / "_merge_multibox_notes.json").write_text(
            json.dumps(multi_notes, ensure_ascii=False, indent=1),
            encoding="utf-8")
    for c, d in stat_cat.items():
        print(f"  类别[{c}] 总量{d['total']}  Locate占{d['locate']} "
              f"({100*d['locate']/d['total']:.1f}%)")
    print(f"  JSON -> {OUT_JSON}  SHA256 {hashlib.sha256(OUT_JSON.read_bytes()).hexdigest().upper()}")
    print(f"  ZIP  -> {OUT_ZIP.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
# -*- coding: utf-8 -*-
"""校验 det / det_rb 的结构性预期，并对比三个版本。

关键预期：
1. det 只在「可推算期望编号」的序数条目上覆盖模型；其余回落常规分支
   （即与 qwen 策略完全一致）
2. 模型原答案本就等于期望编号的条目（MATCH），det 下框应逐位不变
   ——因为 candidates[exp-1] 正是模型当初选的那一个
3. det_rb 的可推算范围与 det 完全一致，但全部保留基线
4. 护栏/冻结条目零改动；键集 9555；框全部合法
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

R = Path(r"d:\Desktop\2026AiC")
sys.path.insert(0, str(R))

import _fuse_qwen as F  # noqa: E402


def load(name: str) -> dict:
    return json.loads((R / name).read_text(encoding="utf-8"))


def main() -> int:
    base = load("result_locate_merge_som.json")
    som = load("qwen3vl_som_all.json")
    test = json.loads(F.TEST_JSON.read_text(encoding="utf-8"))
    cores = json.loads(F.CORES_JSON.read_text(encoding="utf-8"))
    ordinal = set(cores)

    qwen = load("result_qwen_qwen.json")
    det = load("result_qwen_det.json")
    rbr = load("result_qwen_det_rb.json")

    print("=== 1) 基本合法性 ===")
    for nm, d in (("qwen", qwen), ("det", det), ("det_rb", rbr)):
        nb, nn = F.validate(d, test, base)
        print(f"  {nm:7s} 条数 {len(d)}  宽框 {nb}->{nn}  合法性 OK")

    print("\n=== 2) det 的覆盖范围 vs 可推算集合 ===")
    # 必须复刻 fuse() 的跳过顺序：FROZEN 与 untrusted 都在 det 分支之前返回，
    # 否则会把 untrusted 里的序数条目也当成可推算（实测差 24 条）。
    untrusted = F.untrusted_ordinals()
    det_qids, undet_qids, invalid_qids = set(), set(), set()
    for qid in ordinal:
        if qid in F.FROZEN or qid in untrusted:
            continue
        rec = som.get(qid, {})
        new, rule = F.det_ordinal("det", rec, test[qid]["query"])
        if rule == "ORD_UNDET":
            undet_qids.add(qid)
        elif rule == "ORD_DET_INVALID":
            invalid_qids.add(qid)
        else:
            det_qids.add(qid)
    print(f"  可推算 ORD_DET {len(det_qids)} + INVALID {len(invalid_qids)}"
          f" = {len(det_qids) + len(invalid_qids)}；ORD_UNDET {len(undet_qids)}")

    # det 与 qwen 的差异必须落在可推算集合内
    diff = {q for q in base if det[q]["bbox"] != qwen[q]["bbox"]}
    outside = diff - det_qids - invalid_qids
    print(f"  det 与 qwen 的实际差异 {len(diff)} 条")
    print(f"    其中落在可推算集合之外: {len(outside)}  "
          f"{'PASS' if not outside else 'FAIL ' + str(sorted(outside)[:5])}")
    assert not outside, "det 改动了不可推算的条目"

    print("\n=== 3) det 的框必须等于该期望编号的候选（模型选框的同位或修正）===")
    bad_box, same_model, fixed = [], 0, 0
    for qid in det_qids:
        rec = som.get(qid, {})
        exp = F.expected_ordinal_num(test[qid]["query"],
                                    rec.get("bbox_all"), rec.get("n_cands"))
        want = F.norm(rec["bbox_all"][exp - 1])
        if det[qid]["bbox"] != want:
            bad_box.append((qid, exp, det[qid]["bbox"], want))
        if rec.get("num") == exp:
            same_model += 1
        else:
            fixed += 1
    print(f"  框与 candidates[期望编号-1] 不符: {len(bad_box)}  "
          f"{'PASS' if not bad_box else 'FAIL ' + str(bad_box[:3])}")
    assert not bad_box
    print(f"  模型答案本就等于期望编号（未修正）: {same_model}")
    print(f"  模型答案被修正为期望编号:          {fixed}")

    print("\n=== 3b) 未被修正的条目在 det/qwen 下应给出同一个框 ===")
    mism = []
    for qid in det_qids:
        rec = som.get(qid, {})
        exp = F.expected_ordinal_num(test[qid]["query"],
                                    rec.get("bbox_all"), rec.get("n_cands"))
        if rec.get("num") != exp:
            continue
        # 模型选框与 det 选框同源；qwen 版本若采纳 SoM 则也应相同
        if qwen[qid]["bbox"] != base[qid]["bbox"] and \
                qwen[qid]["bbox"] != F.norm(rec["bbox"]):
            mism.append(qid)
    print(f"  qwen 版本既非基线又非模型选框的条目: {len(mism)}  "
          f"{'PASS' if not mism else 'FAIL ' + str(mism[:3])}")
    assert not mism

    print("\n=== 4) det 实际改动了多少序数（相对 qwen）===")
    ord_diff = {q for q in diff if q in ordinal}
    print(f"  序数条目: {len(ord_diff)} 条")
    print(f"  其中与基线不同（真改动）: "
          f"{sum(1 for q in ord_diff if det[q]['bbox'] != base[q]['bbox'])}")
    print(f"  与基线相同（= 回退效果）: "
          f"{sum(1 for q in ord_diff if det[q]['bbox'] == base[q]['bbox'])}")

    print("\n=== 5) det_rb 的范围：应只回退「模型答案与期望不符」的可疑条目 ===")
    # 可疑集合必须基于「期望编号可算」（expected_ordinal_num 非 None），
    # 而不是「det 能否采纳」——后者含 ORD_DET_INVALID（如期望候选是边缘退化框），
    # 那类在 det_rb 下同样属于可疑回退（实测 009340_005 即此情形）。
    suspect = set()
    for qid in det_qids | invalid_qids:
        rec = som.get(qid, {})
        exp = F.expected_ordinal_num(test[qid]["query"],
                                    rec.get("bbox_all"), rec.get("n_cands"))
        if exp is not None and rec.get("num") != exp:
            suspect.add(qid)
    rb = {q for q in base if rbr[q]["bbox"] != qwen[q]["bbox"]}
    print(f"  可疑条目（期望可算且模型答案 != 期望）: {len(suspect)}")
    print(f"  det_rb 相对 qwen 的差异:               {len(rb)}")
    extra = rb - suspect
    print(f"    超出可疑集合的部分: {len(extra)}  "
          f"{'PASS' if not extra else 'FAIL ' + str(sorted(extra)[:5])}")
    assert not extra, "det_rb 回退了非可疑条目"
    print(f"  det_rb 差异是否全部等于基线: "
          f"{all(rbr[q]['bbox'] == base[q]['bbox'] for q in rb)}")
    assert all(rbr[q]["bbox"] == base[q]["bbox"] for q in rb)
    missing = suspect - rb
    print(f"  可疑条目中未被回退的: {len(missing)}"
          f"（基线本就等于 qwen 输出时不会有差异，属正常）")
    for q in sorted(missing)[:5]:
        print(f"      {q} 基线={base[q]['bbox'][:2]} qwen={qwen[q]['bbox'][:2]}")

    print("\n=== 6) 冻结/护栏零改动（三版本）===")
    for nm, d in (("qwen", qwen), ("det", det), ("det_rb", rbr)):
        bad = [g for g in F.FROZEN if d[g]["bbox"] != base[g]["bbox"]]
        print(f"  {nm:7s} 被改 {len(bad)} {'PASS' if not bad else 'FAIL ' + str(bad)}")
        assert not bad

    print("\n=== 7) 三种版本差异汇总 ===")
    print(f"  det vs qwen:     {len(diff)} 条不同")
    print(f"  det_rb vs qwen:  {len(rb)} 条不同")
    print(f"  det vs det_rb:   "
          f"{sum(1 for q in base if det[q]['bbox'] != rbr[q]['bbox'])} 条不同")
    print("\nALL CHECKS PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
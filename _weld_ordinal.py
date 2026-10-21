# -*- coding: utf-8 -*-
"""把已人工确认的序数选中框焊回最终提交 JSON。

- 输入：result_gd_3modal_final.json（当前锁版 9555 条）
        + ordinal_select_result.json（已接受的选择，10 条）
- 输出：result_gd_3modal_ord.json（新版本，仅覆盖 10 条 bbox）
- 当前锁版归档到 _deprecated/
校验：总条目仍为 9555，且与 test.json 键集一致。
"""
import hashlib
import json
import os
import shutil

SRC = r"d:\Desktop\2026AiC\result_gd_3modal_final.json"
SEL = r"d:\Desktop\2026AiC\ordinal_select_result.json"
TEST = r"d:\Desktop\2026AiC\初赛数据集-基于大模型的多模态视觉理解与推理\test.json"
OUT = r"d:\Desktop\2026AiC\result_gd_3modal_ord.json"
DEP = r"d:\Desktop\2026AiC\_deprecated"

# 明确被接受的 qid（排除 TRUNC 的 010019_003——其 rank5>3，模型客观只框到 3 人，
# 若强行焊入会把"第5人"错标成第3个远处人，这里默认剔除，交由人复核）
ACCEPTED = [
    "000023_002", "000023_003", "000023_004", "001773_005",
    "010033_004", "012826_001", "014716_003", "016333_002", "016774_001",
]
# TRUNC 案例默认不焊（可人工按需开启）
WRITE_TRUNC = False
TRUNC_SET = {"010019_003"}


def sha256(path: str) -> str:
    """计算文件 SHA256。"""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def main() -> int:
    """执行焊接并校验输出。"""
    os.makedirs(DEP, exist_ok=True)
    data = json.load(open(SRC, encoding="utf-8"))
    sel = json.load(open(SEL, encoding="utf-8"))
    test = json.load(open(TEST, encoding="utf-8"))

    before = len(data)
    assert before == 9555 and set(data) == set(test), \
        "源与 test.json 键不一致"

    qids = list(ACCEPTED)
    if WRITE_TRUNC:
        qids.extend(TRUNC_SET)

    applied = []
    for qid in qids:
        if qid not in sel:
            print(f"[SKIP] {qid} 无选择结果")
            continue
        entry = sel[qid]
        if entry.get("bbox") is None:
            print(f"[SKIP] {qid} 空家底（未选中）")
            continue
        if qid not in data:
            print(f"[ERR] {qid} 不在最终提交中")
            continue
        data[qid]["bbox"] = [round(float(v), 6) for v in entry["bbox"]]
        meta = f"rank={entry['rank']} axis={entry.get('axis')} pool={entry['n_pool']}"
        if entry.get("truncated"):
            meta += " TRUNC"
        applied.append((qid, meta))

    # 保存快照供核对
    snap = {qid: {"bbox": data[qid]["bbox"], "meta": m}
            for qid, m in applied}
    with open(r"d:\Desktop\2026AiC\_ordinal_weld_snapshot.json",
              "w", encoding="utf-8") as f:
        json.dump(snap, f, ensure_ascii=False, indent=1)

    # 校验：键集与条目数不变
    assert len(data) == before and set(data) == set(test), \
        "焊接后键集/条目数异常"
    json.dump(data, open(OUT, "w", encoding="utf-8"), ensure_ascii=False)
    gc = [(a[0], a[1]) for a in applied]

    # 归档旧锁版
    old_name = os.path.basename(SRC).replace(".json", ".prev.json")
    shutil.copy(SRC, os.path.join(DEP, old_name))

    print(f"写入 {OUT}（{len(data)} 条）")
    print("应用条数:", len(gc))
    for q, m in gc:
        print(f"  {q}  {m}")
    print(f"源 SHA256 {sha256(SRC)}")
    print(f"新 SHA256 {sha256(OUT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
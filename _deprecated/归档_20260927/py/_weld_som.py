# -*- coding: utf-8 -*-
"""SoM 序数修复焊接：把 000023_001/005 换成 SoM 选中的窄框。

- 输入：result_locate_merge.json（当前锁定版，9555 条）
        qwen3vl_som.json（SoM 结果，人工已确认 001/005 窄框正确）
- 输出：result_locate_merge_som.json + .zip
- 旧版归档到 _deprecated/
- 校验：9555 键集一致、字段齐全、bbox 合法、护栏 4 样本零改动、只改这 2 条
"""
import hashlib
import json
import os
import shutil
import zipfile
from pathlib import Path

R = Path(r"d:\Desktop\2026AiC")
SRC = R / "result_locate_merge.json"
SOM = R / "qwen3vl_som.json"
TEST = R / "初赛数据集-基于大模型的多模态视觉理解与推理" / "test.json"
OUT_JSON = R / "result_locate_merge_som.json"
OUT_ZIP = R / "result_locate_merge_som.zip"
DEP = R / "_deprecated"
GUARD = ["000282_004", "000282_008", "002004_001", "002004_002"]

APPLY = ["000023_001", "000023_005"]


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest().upper()


def main() -> int:
    DEP.mkdir(exist_ok=True)
    data = json.loads(SRC.read_text(encoding="utf-8"))
    som = json.loads(SOM.read_text(encoding="utf-8"))
    test = json.loads(TEST.read_text(encoding="utf-8"))
    assert len(data) == 9555 and set(data) == set(test), "源键集异常"

    qres = som["000023"]["queries"]
    applied = []
    for qid in APPLY:
        newb = qres[qid]["bbox"]
        oldb = data[qid]["bbox"]
        assert newb, f"{qid} SoM 未选中"
        data[qid]["bbox"] = [round(float(v), 6) for v in newb]
        applied.append((qid, oldb, data[qid]["bbox"], qres[qid]["answer_num"]))

    # 校验
    assert len(data) == 9555 and set(data) == set(test), "键集异常"
    FIELDS = {"visible", "infrared", "depth", "query", "bbox"}
    bad = [k for k, v in data.items()
           if set(v) != FIELDS
           or not (0 <= v["bbox"][0] < v["bbox"][2] <= 1
                   and 0 <= v["bbox"][1] < v["bbox"][3] <= 1)]
    assert not bad, f"非法条目 {bad[:5]}"
    for g in GUARD:
        src = json.loads(SRC.read_text(encoding="utf-8"))
        assert data[g]["bbox"] == src[g]["bbox"], f"护栏被改 {g}"

    OUT_JSON.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    with zipfile.ZipFile(OUT_ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("result_locate_merge_som.json",
                    json.dumps(data, ensure_ascii=False).encode("utf-8"))

    # 归档旧版
    shutil.copy(SRC, DEP / "result_locate_merge.prev20260917.json")
    shutil.copy(R / "result_locate_merge.zip",
                DEP / "result_locate_merge.prev20260917.zip")

    print("== 应用修改 ==")
    for qid, oldb, newb, num in applied:
        print(f"  {qid}: SoM#{num}")
        print(f"    旧 {[round(v,3) for v in oldb]}")
        print(f"    新 {[round(v,3) for v in newb]}")
    print(f"\n条目 {len(data)}，键集一致 True，字段/bbox 异常 0")
    print(f"护栏 4 样本零改动 True")
    print(f"\nJSON SHA256 {sha256(OUT_JSON)}")
    print(f"ZIP  SHA256 {sha256(OUT_ZIP)}")
    print(f"输出 {OUT_JSON.name} / {OUT_ZIP.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
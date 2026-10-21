#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""按技术文档 §3.4 装配 submission.json + zip。

用法：python _make_submission.py <pred.json> <out_dir>
  pred.json : 管线产物 {qid: {"bbox": [x1,y1,x2,y2] | None, ...}}
  out_dir   : 输出目录（写 submission.json 与 submit.zip）

规则（文档 §3.4，逐字对齐）：
  · bbox 归一化到 [0,1]，坐标 round 至 4 位小数
  · 非法 / 缺失输出回退占位框 [0.49, 0.49, 0.51, 0.51]
  · 每查询条目附原始三模态路径与查询文本
  · zip 内部文件名 submission.json
"""
import json
import os
import sys
import zipfile

FALLBACK = [0.49, 0.49, 0.51, 0.51]


def valid(bb):
    """合法 = 4 元数值列表，全部在 [0,1]，且 x1<=x2、y1<=y2。"""
    if not isinstance(bb, (list, tuple)) or len(bb) != 4:
        return False
    try:
        v = [float(x) for x in bb]
    except (TypeError, ValueError):
        return False
    if any(x < 0.0 or x > 1.0 for x in v):
        return False
    return v[0] <= v[2] and v[1] <= v[3]


def main():
    pred_path, out_dir = sys.argv[1], sys.argv[2]
    data = os.environ.get("QWEN_DATA", "/root/rivermind-data/dataset")
    q = json.load(open(os.path.join(data, "queries", "queries.json"),
                       encoding="utf-8"))
    pred = json.load(open(pred_path, encoding="utf-8"))

    sub, n_fb = {}, 0
    for k, item in q.items():
        bb = (pred.get(k) or {}).get("bbox")
        if not valid(bb):
            bb = FALLBACK
            n_fb += 1
        sub[k] = {
            "visible": item["visible"],
            "infrared": item["infrared"],
            "depth": item["depth"],
            "query": item["query"],
            "bbox": [round(float(x), 4) for x in bb],
        }

    os.makedirs(out_dir, exist_ok=True)
    jp = os.path.join(out_dir, "submission.json")
    with open(jp, "w", encoding="utf-8") as f:
        json.dump(sub, f, ensure_ascii=False, indent=1)
    zp = os.path.join(out_dir, "submit.zip")
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(jp, "submission.json")

    # 完整性与合法性自检（文档 §5 质量校验）
    assert len(sub) == len(q) == 5690, f"条数 {len(sub)} != 5690"
    bad = [k for k, v in sub.items() if not valid(v["bbox"])]
    assert not bad, f"非法框 {len(bad)}"
    print(f"OK 条数={len(sub)} 回退占位={n_fb} 非法=0")
    print(f"  {jp} ({os.path.getsize(jp)} bytes)")
    print(f"  {zp} ({os.path.getsize(zp)} bytes)")


if __name__ == "__main__":
    main()

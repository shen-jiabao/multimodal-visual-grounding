# -*- coding: utf-8 -*-
"""Locate 全量推理结果 → 最终提交 JSON/zip。

- 输入：/srv/output/locate_bbox_all.json（{qid: {bbox, ...}}）+ 本地 queries/test.json
- 输出：result_locate_full.json + result_locate_full.zip（内部 result_locate.json）
- 处理：
  1. 缺失 bbox（模型未定位）→ 兜底：整幅图框 [0,0,1,1]（保持字段完整，标记 miss）
  2. 字段按 queries.json 补齐 visible/infrared/depth/query/bbox
  3. 校验 9555 键集一致，bbox 归一化合法
"""
import hashlib
import json
import zipfile
from pathlib import Path

R = Path(__file__).resolve().parent
DATA = R / "初赛数据集-基于大模型的多模态视觉理解与推理"
SRC_JSON = R / "locate_bbox_all.json"   # 从远端拉回后放这里
QUERIES = DATA / "queries" / "queries.json"
TEST = DATA / "test.json"
OUT_JSON = R / "result_locate_full.json"
OUT_ZIP = R / "result_locate_full.zip"


def sha256_b(p: bytes) -> str:
    return hashlib.sha256(p).hexdigest().upper()


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
    loc = json.loads(SRC_JSON.read_text(encoding="utf-8"))
    queries = json.loads(QUERIES.read_text(encoding="utf-8"))
    test = json.loads(TEST.read_text(encoding="utf-8"))
    assert set(queries) == set(test), "queries 与 test 键不一致"

    out = {}
    miss, n_fill = [], 0
    for k, rec in queries.items():
        entry = dict(rec)  # 复制 visible/infrared/depth/query
        lb = loc.get(k, {}).get("bbox")
        if lb is None:
            entry["bbox"] = [0.0, 0.0, 1.0, 1.0]   # 兜底整图
            miss.append(k)
            n_fill += 1
        else:
            entry["bbox"] = [round(float(v), 6) for v in norm(lb)]
        out[k] = entry

    assert len(out) == 9555 and set(out) == set(test), "键集/条目数异常"
    bad_fmt = sum(1 for v in out.values()
                  if not (0.0 <= v["bbox"][0] < v["bbox"][2] <= 1.0
                          and 0.0 <= v["bbox"][1] < v["bbox"][3] <= 1.0))
    assert bad_fmt == 0, "含非法 bbox"

    OUT_JSON.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    with zipfile.ZipFile(OUT_ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("result_locate.json",
                    json.dumps(out, ensure_ascii=False).encode("utf-8"))
    print(f"总共 {len(out)} 条；bbox 命中 {len(out)-len(miss)}，"
          f"兜底 {n_fill}（miss {len(miss)}）")
    print(f"JSON -> {OUT_JSON} SHA256 {sha256_b(OUT_JSON.read_bytes())}")
    print(f"ZIP  -> {OUT_ZIP.name}")
    if miss:
        print("miss 前10:", miss[:10])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
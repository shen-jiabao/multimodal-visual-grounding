# -*- coding: utf-8 -*-
"""20 条冒烟：按图片分组，列出「图上画的框 ↔ 对应问题的完整内容」。

每张目检图里有两种框：
    蓝色细框 = WeDetect top-100 候选（全部）
    红色粗框 = VLX-Seek 选中的那一个（即 result_bbox_list[0]）

同一张图可能对应多条查询（如 000001.png 有 6 条），故同一张图会生成 6 个
目检图副本，文件名即 qid。红框位置各不相同，对应各自的查询。

用法
    python -B _vlx_smoke_read.py
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

R = Path(r"d:\Desktop\2026AiC")
sys.path.insert(0, str(R))

import _fuse_qwen as F  # noqa: E402
import _sub_path as SP  # noqa: E402

SEMI = R / "复赛数据集-基于大模型的多模态视觉理解与推理"
W = 96


def main() -> int:
    res = json.loads((R / "_vlx_smoke20_out.json").read_text(encoding="utf-8"))
    base = json.loads(SP.find("result_semi_n32_orddirect").read_text(
        encoding="utf-8"))
    cores = set(json.loads((R / "_ordinal_cores_r2.json").read_text(
        encoding="utf-8")))
    sup = set(json.loads((R / "_vlx_sup_split.json").read_text(
        encoding="utf-8")))

    by_img: dict[str, list[tuple[str, list[float]]]] = defaultdict(list)
    for k, v in res.items():
        if v.get("bbox"):
            by_img[v["img"]].append((k, v["bbox"]))
    dup: set[str] = set()
    for _, items in by_img.items():
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                if F.iou(items[i][1], items[j][1]) >= 0.90:
                    dup.update({items[i][0], items[j][0]})

    order = list(res.keys())
    groups: dict[str, list[str]] = defaultdict(list)
    for k in order:
        groups[res[k]["img"]].append(k)

    for img, qids in groups.items():
        print("=" * W)
        print(f"【{img}】 共 {len(qids)} 条查询 → {len(qids)} 张目检图")
        print("=" * W)
        for k in qids:
            v = res[k]
            b = v.get("bbox")
            bb = base[k]["bbox"]
            io = F.iou(b, bb) if b else 0.0
            kind = "序数" if k in cores else "非序"
            kinds = [kind]
            if k in sup:
                kinds.append("∈最高级层")
            if k in dup:
                kinds.append("★同图重复框")
            print(f"  目检图 {k}.jpg   [{'+'.join(kinds)}]")
            print(f"    问题（完整）：{v['query']}")
            if b:
                oi = str(v.get("object_index") or "?")
                print(f"    红框 = VLX 选中的候选 {oi}（候选共 {v.get('n_cand')} 个；"
                      f"官方文档称该表为 top-100，故编号疑为置信度降序排名——"
                      f"未经核实）")
                print(f"      框 [{b[0]:.4f} {b[1]:.4f} {b[2]:.4f} {b[3]:.4f}]"
                      f"  面积 {F.area(b):.4f}")
                print(f"    基线框（0.7766 件）"
                      f"[{bb[0]:.4f} {bb[1]:.4f} {bb[2]:.4f} {bb[3]:.4f}]"
                      f"  面积 {F.area(bb):.4f}   IoU {io:.3f}"
                      f"   {'与基线一致(装配时不动)' if io >= 0.5 else '★与基线分歧'}")
            else:
                print(f"    红框：无（result_bbox_list 为空 → 装配时保留基线）")
                print(f"    基线框：[{bb[0]:.4f} {bb[1]:.4f} {bb[2]:.4f} "
                      f"{bb[3]:.4f}]")
            if v.get("err"):
                print(f"    ✗ 异常：{v['err']}")
            print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

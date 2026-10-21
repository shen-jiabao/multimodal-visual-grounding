# -*- coding: utf-8 -*-
"""拉回 Qwen3-VL 探针结果并做三方对比。

对比对象：
  QW = qwen3vl_probe.json            (Qwen3-VL-8B)
  MERGE = result_locate_merge.json    (当前锁定提交)
  GD  = result_gd_3modal_fullseq.json (GD 三模态基线)
无 GT，故只报"与谁一致/分歧"，不断言谁对；分歧样本列坐标供人眼看图。
"""
import json
import os
import statistics
import sys
from collections import Counter

import paramiko

R = r"d:\Desktop\2026AiC"
LOCAL = os.path.join(R, "qwen3vl_probe.json")


def iou(a, b):
    if not a or not b:
        return 0.0
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0., x2 - x1), max(0., y2 - y1)
    it = iw * ih
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - it
    return it/ua if ua > 1e-9 else 0.0


def area(b):
    return (b[2]-b[0])*(b[3]-b[1])


def pull():
    cli = paramiko.SSHClient()
    cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    cli.connect(os.environ["SSH_HOST"], port=int(os.environ["SSH_PORT"]),
                username=os.environ["SSH_USER"],
                password=os.environ["SSH_PW"], timeout=25,
                banner_timeout=30, auth_timeout=30)
    sftp = cli.open_sftp()
    sftp.get("/srv/output/qwen3vl_probe.json", LOCAL)
    sftp.close()
    cli.close()
    print(f"已拉回 {LOCAL}")


def main():
    pull()
    qw = json.load(open(LOCAL, encoding="utf-8"))
    merge = json.load(open(os.path.join(R, "result_locate_merge.json"),
                           encoding="utf-8"))
    gd = json.load(open(os.path.join(R, "result_gd_3modal_fullseq.json"),
                        encoding="utf-8"))
    queries = json.load(open(os.path.join(
        R, "初赛数据集-基于大模型的多模态视觉理解与推理", "queries",
        "queries.json"), encoding="utf-8"))

    n = len(qw)
    hits = sum(1 for v in qw.values() if v.get("bbox"))
    times = [v["time_s"] for v in qw.values() if v.get("time_s")]
    ious_m, ious_g = [], []
    for k, v in qw.items():
        if not v.get("bbox"):
            continue
        ious_m.append(iou(v["bbox"], merge[k]["bbox"]))
        ious_g.append(iou(v["bbox"], gd[k]["bbox"]))

    print(f"\n== Qwen3-VL 探针 {n} 条 ==")
    print(f"命中 {hits} ({100*hits/n:.1f}%)  单条均值 "
          f"{statistics.mean(times):.2f}s  P95 "
          f"{sorted(times)[int(len(times)*0.95)-1]:.2f}s")
    print(f"与当前提交 IoU: 均值 {statistics.mean(ious_m):.3f} "
          f"中位 {statistics.median(ious_m):.3f}")
    print(f"与 GD 基线   IoU: 均值 {statistics.mean(ious_g):.3f} "
          f"中位 {statistics.median(ious_g):.3f}")
    print(f"  QW~提交 一致(IoU>=0.5): {sum(1 for x in ious_m if x>=0.5)}/{len(ious_m)}")
    print(f"  QW~GD   一致(IoU>=0.5): {sum(1 for x in ious_g if x>=0.5)}/{len(ious_g)}")

    # 面积分布（是否出现整图/大框）
    big = [k for k, v in qw.items() if v.get("bbox") and area(v["bbox"]) > 0.35]
    print(f"\nQwen3-VL 大框(area>0.35): {len(big)} {big[:5]}")

    # 分歧最大的样本（与提交 IoU<0.2）
    div = [(k, iou(qw[k]["bbox"], merge[k]["bbox"]))
           for k in qw if qw[k].get("bbox")
           and iou(qw[k]["bbox"], merge[k]["bbox"]) < 0.2]
    div.sort(key=lambda x: x[1])
    print(f"\n与提交分歧(IoU<0.2): {len(div)}/{n}")
    for k, iv in div[:12]:
        q = queries[k]["query"]
        print(f"  {k} IoU={iv:.2f} | {q[:58]}")
        print(f"     QW={[round(x,3) for x in qw[k]['bbox']]} "
              f"提交={[round(x,3) for x in merge[k]['bbox']]}")

    # 类别统计
    import re
    ORD = re.compile(r"\b(first|second|third|fourth|fifth|last)\b", re.I)
    SPAT = re.compile(r"\b(behind|closest|nearest|left|right|above|below)\b", re.I)
    cat = Counter()
    for k in qw:
        q = queries[k]["query"]
        c = "ordinal" if ORD.search(q) else ("spatial" if SPAT.search(q) else "other")
        cat[c] += 1
        if qw[k].get("bbox"):
            cat[c + "_hit"] += 1
    print(f"\n类别分布: {dict(cat)}")

    json.dump({"n": n, "hits": hits,
               "time_mean": statistics.mean(times),
               "iou_vs_merge_mean": statistics.mean(ious_m),
               "iou_vs_gd_mean": statistics.mean(ious_g)},
              open(os.path.join(R, "qwen3vl_probe_summary.json"), "w",
                   encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
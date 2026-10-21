# -*- coding: utf-8 -*-
"""统计"模型实际给出的不同框数 < 该图 query 数"的规模（复赛 5690 条全量）。

回答一个问题：除了 92 条可见占位框，还有多少 query 没有拿到"属于自己的框"？
对每张图：
  n_q      = query 数
  n_uniq   = 非占位框去重后的个数
  n_ph     = 占位框数
  n_real   = 非占位框条数
  gap      = n_q - n_ph - n_uniq   (>0 表示真实框不够分)
"""
import json
import tarfile
from collections import defaultdict

ARCHIVE = "outputs/复赛/archive/run_20260924_090232_qwen_multiimg.tar.gz"
PH = (0.49, 0.49, 0.51, 0.51)


def is_ph(b):
    return all(abs(b[i] - PH[i]) <= 1e-6 for i in range(4))


def main():
    with tarfile.open(ARCHIVE) as t:
        m = [x for x in t.getmembers() if x.name.endswith("/submission.json")][0]
        sub = json.load(t.extractfile(m))

    by_img = defaultdict(list)
    for qid, rec in sub.items():
        by_img[rec["visible"]].append((qid, tuple(rec["bbox"])))

    tot_q = tot_ph = tot_real = tot_uniq = 0
    gap_imgs = 0
    gap_sum = 0
    one_to_one = 0
    hist = defaultdict(int)
    worst = []
    for vis, items in by_img.items():
        n_q = len(items)
        ph = [b for _, b in items if is_ph(b)]
        real = [b for _, b in items if not is_ph(b)]
        uniq = {tuple(round(x, 4) for x in b) for b in real}
        n_ph, n_real, n_uniq = len(ph), len(real), len(uniq)
        gap = n_q - n_ph - n_uniq
        tot_q += n_q; tot_ph += n_ph; tot_real += n_real; tot_uniq += n_uniq
        hist[gap] += 1
        if gap > 0:
            gap_imgs += 1
            gap_sum += gap
            worst.append((gap, n_q, n_ph, n_uniq, vis))
        if n_q == n_uniq + n_ph:
            one_to_one += 1

    print(f"总 query {tot_q}  占位 {tot_ph}  真实框条数 {tot_real}  真实框去重 {tot_uniq}")
    print(f"图片 {len(by_img)}")
    print(f"gap>0 (真实框不够分) 的图: {gap_imgs}  gap 合计 {gap_sum} 条")
    print(f"query 数 == 不同框数 + 占位数 (一一对应) 的图: {one_to_one}")
    print(f"gap 直方图 (gap: 图片数): {dict(sorted(hist.items()))}")
    print("\ngap 最大的 15 张图 (gap, n_q, n_ph, n_uniq, visible):")
    for row in sorted(worst, reverse=True)[:15]:
        print(f"  gap={row[0]:2d} q={row[1]:2d} ph={row[2]} uniq={row[3]:2d}  {row[4].split('/')[-1]}")

    # 每个 query 的"独立框比例"：真实框条数中落在多重复框上的比例
    multi = sum(1 for _, items in by_img.items()
                for b in [x[1] for x in items]
                if not is_ph(b) and sum(1 for _, bb in items if tuple(round(v, 4) for v in bb) == tuple(round(v, 4) for v in b)) > 1)
    print(f"\n落在'多 query 共用同一框'上的 query 条数: {multi} ({multi/tot_q*100:.2f}%)")


if __name__ == "__main__":
    main()

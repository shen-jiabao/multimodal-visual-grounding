# -*- coding: utf-8 -*-
"""按 visible 文件名（后缀匹配）打印该图全部 query 的匹配结果，用于人工核对异常图。

用法: python scripts/复赛/dump_image.py 001128.png 016269.png 001436.png
"""
import json
import sys
import tarfile

ARCHIVE = "outputs/复赛/archive/run_20260924_090232_qwen_multiimg.tar.gz"
PH = (0.49, 0.49, 0.51, 0.51)


def is_ph(b):
    return all(abs(b[i] - PH[i]) <= 1e-6 for i in range(4))


def main():
    targets = sys.argv[1:]
    with tarfile.open(ARCHIVE) as t:
        m = [x for x in t.getmembers() if x.name.endswith("/submission.json")][0]
        sub = json.load(t.extractfile(m))
    for target in targets:
        print("=" * 78)
        print(target)
        rows = [(k, v) for k, v in sub.items() if v["visible"].endswith(target)]
        rows.sort(key=lambda x: x[0])
        for k, v in rows:
            b = v["bbox"]
            tag = "PLACEHOLDER" if is_ph(b) else "real       "
            print(f"  {k}  {tag} [{b[0]:.4f},{b[1]:.4f},{b[2]:.4f},{b[3]:.4f}] "
                  f"len={len(v['query']):3d}  {v['query'][:92]}")
        boxes = [v["bbox"] for _, v in rows]
        uniq = {tuple(round(x, 4) for x in b) for b in boxes}
        print(f"  -> n={len(rows)}  distinct_boxes={len(uniq)}  placeholders={sum(1 for b in boxes if is_ph(b))}")


if __name__ == "__main__":
    main()

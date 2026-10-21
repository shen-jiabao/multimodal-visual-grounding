# -*- coding: utf-8 -*-
"""按 qid 清单从证据文件里筛子集 / 取补集（分层读数用）。

为什么需要它
    VLX 的改动面要按「含空间最高级 646 / 其余 3941」分两层读，
    但**单次总分只给一个标量**，拆不出两层。做法是**用同一份装配器再跑一次**：
        A 喂全部证据            → −156 条（已实测）
        B 只喂 3941（排除 646） → 得 3941 层的净翻转
        ⇒ 646 层 = A − B
    本脚本只负责造 B 的输入，**不含任何判据**（判据只在 `_assembly_32b.py` 里，
    这样不会漂移——本项目已有"渲染器与融合脚本必须共用判据"的教训）。

用法
    python -B _vlx_sel.py <源.json> <qid清单.json> <输出.json> [--exclude]
        --exclude：取补集（输出清单之外的条目）
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("keys")
    ap.add_argument("dst")
    ap.add_argument("--exclude", action="store_true",
                    help="取补集：输出清单**之外**的条目")
    a = ap.parse_args()

    src, keys, dst = Path(a.src), Path(a.keys), Path(a.dst)
    d = json.loads(src.read_text(encoding="utf-8"))
    ks = json.loads(keys.read_text(encoding="utf-8"))

    if a.exclude:
        drop = set(ks)
        sub = {k: v for k, v in d.items() if k not in drop}
        print(f"  源 {len(d)} 条 − 排除清单 {len(drop)} 条 → 子集 {len(sub)} 条")
    else:
        sub = {k: d[k] for k in ks if k in d}
        print(f"  源 {len(d)} 条 → 清单 {len(ks)} 条 → 子集 {len(sub)} 条")
        miss = [k for k in ks if k not in d]
        if miss:
            print(f"  ★清单里有 {len(miss)} 条不在源里（前 5：{miss[:5]}）")

    dst.write_text(json.dumps(sub, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    print(f"  已写 {dst}（{dst.stat().st_size/1e6:.2f} MB）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

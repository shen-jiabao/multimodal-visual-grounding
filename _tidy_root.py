# -*- coding: utf-8 -*-
"""根目录整理：把"无任何源码引用"的文件归档到 `_deprecated/归档_20260727/`。

两条硬约束（不可违反）
    1. **不删除任何文件**——只移动，并留下可一键回退的 MANIFEST。
    2. **不移动任何被引用或被实录指纹锚定的文件**——否则 `跑后实录` 的
       "本地可复现"路径会断，且活跃链路（复赛规则层/装配/校验）会失效。

判据（可审计、可复现）
    对根级每个文件 f，若满足任一即判为"动不得"：
      a) `f.name` 出现在任一 .py/.sh/.bat 源码文本中；或
      b) `f` 是 .py 且 `f.stem` 出现在源码中（覆盖 `import _fuse_qwen` 这类
         只出现模块名的引用——按文件名判会漏掉）；或
      c) `f` 在 PROTECT 豁免集内（其内容哈希被写进 `_frozen_criteria__跑后实录.md`，
         或本身是预登记件/清单/复算脚本/交付依赖）。
    其余一律归档。

用法
    python -B _tidy_root.py            # 预演（只打印，不动文件）
    python -B _tidy_root.py --apply    # 真执行
"""
from __future__ import annotations

import argparse
import shutil
import sys
from collections import Counter
from pathlib import Path

R = Path(r"d:\Desktop\2026AiC")
DEST = R / "_deprecated" / "归档_20260927"
SELF = Path(__file__).resolve().name

# 豁免：移动它们会破坏"跑后实录"的可核查性，或属交付依赖
PROTECT = {
    "_frozen_criteria.md",              # 预登记件（其指纹必须继续有效）
    "_frozen_criteria__跑后实录.md",
    "_root_manifest.txt",               # 关服清单（补丁前）
    "_root_manifest2.txt",              # 关服清单（补丁后）
    "_ord_layer_levers.py",             # 实录的复算脚本
    "requirements.txt",
    SELF,
}

CAT = {".py": "py", ".sh": "sh", ".bat": "sh",
       ".json": "json", ".log": "log",
       ".md": "doc", ".txt": "doc", ".pdf": "doc",
       ".zip": "zip", ".tgz": "zip"}

# 无源码引用的目录也一并归档
DIR_CANDIDATES = ["viz_probe_8b_vs_32b", "viz_thermal_random",
                  "viz_thermal_typical", "__pycache__"]


def classify():
    """返回 (动不得, 可归档)；判据见模块 docstring。"""
    srcs = [p for p in R.iterdir() if p.suffix in (".py", ".sh", ".bat")]
    text = "\n".join(p.read_text(encoding="utf-8", errors="replace")
                     for p in srcs)
    keep, move = [], []
    for p in sorted(R.iterdir()):
        if not p.is_file():
            continue
        hit = (p.name in text
               or (p.suffix == ".py" and p.stem in text)
               or p.name in PROTECT)
        (keep if hit else move).append(p)
    return keep, move, len(srcs)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="真执行（缺省只预演）")
    a = ap.parse_args()

    keep, move, n_src = classify()
    print(f"根级文件 {len(keep) + len(move)} | 动不得 {len(keep)} | "
          f"可归档 {len(move)}（源码 {n_src} 个参与判据）")
    by = Counter(CAT.get(p.suffix.lower(), "other") for p in move)
    print("  归档去向：", dict(sorted(by.items())))
    print(f"  合计 {sum(p.stat().st_size for p in move) / 1e6:.1f} MB")

    if not a.apply:
        print("\n[预演] 未移动任何文件。加 --apply 执行。")
        print("  动不得（前 20）:", [p.name for p in keep[:20]])
        return 0

    DEST.mkdir(parents=True, exist_ok=True)
    rows = []
    for p in move:
        sub = CAT.get(p.suffix.lower(), "other")
        d = DEST / sub
        d.mkdir(parents=True, exist_ok=True)
        tgt = d / p.name
        if tgt.exists():
            tgt = d / (p.stem + "__2" + p.suffix)
        shutil.move(str(p), str(tgt))
        rows.append((p.name, str(tgt.relative_to(DEST)),
                     p.stat().st_size if p.exists() else 0))
    # 目录
    moved_dirs = []
    for nm in DIR_CANDIDATES:
        src = R / nm
        if src.is_dir():
            tgt = DEST / "dirs" / nm
            tgt.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(tgt))
            moved_dirs.append(nm)

    # MANIFEST（可回退）
    man = DEST / "MANIFEST.txt"
    with man.open("w", encoding="utf-8") as f:
        f.write("# 归档清单：原始名<TAB>归档内相对路径<TAB>字节数\n")
        f.write("# 由 _tidy_root.py 于根目录整理时生成；判据见该脚本 docstring。\n")
        f.write("# 回退：python -B _deprecated/归档_20260927/取回.py\n")
        f.write("# 注意：归档区脚本内的路径常量仍指向项目根，故**不可直接在归档区运行**，\n")
        f.write("#       需要运行请先按本清单取回原位。\n")
        for nm, rel, sz in rows:
            f.write(f"{nm}\t{rel}\t{sz}\n")
        for nm in moved_dirs:
            f.write(f"[目录] {nm}\tdirs/{nm}\t0\n")
    print(f"\n已归档 {len(rows)} 个文件 + {len(moved_dirs)} 个目录 → {DEST}")
    print(f"清单 {man.relative_to(R)}（{man.stat().st_size} 字节）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
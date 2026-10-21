from __future__ import annotations

import argparse
import json
import os
import zipfile
import zlib
from collections import Counter


def disk_files(root: str, ignore: list[str] | None = None):
    result = {}
    for dirpath, _, names in os.walk(root):
        for n in names:
            p = os.path.join(dirpath, n)
            rel = os.path.relpath(p, root).replace("\\", "/")
            if ignore and any(rel == pre or rel.startswith(pre + "/") for pre in ignore):
                continue
            result[rel] = os.path.getsize(p)
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True, help="官方数据集 zip 路径")
    ap.add_argument("--root", required=True, help="磁盘数据集根目录（如 dataset）")
    ap.add_argument("--json", default=None, help="结果 JSON 输出路径")
    ap.add_argument("--ignore", nargs="*", default=[], help="扫描时忽略的相对路径前缀（如 初赛）")
    args = ap.parse_args()

    with zipfile.ZipFile(args.zip) as z:
        zmap = {}
        for i in z.infolist():
            if i.is_dir():
                continue
            rel = i.filename.split("/", 1)[1] if "/" in i.filename else i.filename
            zmap[rel] = (i.file_size, i.CRC)

    dmap = disk_files(args.root, args.ignore)

    missing = sorted(set(zmap) - set(dmap))
    extra = sorted(set(dmap) - set(zmap))
    size_mismatch = sorted(k for k in zmap if k in dmap and zmap[k][0] != dmap[k])
    crc_mismatch = []
    crc_checked = 0
    for k in sorted(set(zmap) & set(dmap)):
        p = os.path.join(args.root, k.replace("/", os.sep))
        crc = 0
        with open(p, "rb") as f:
            while True:
                chunk = f.read(1 << 20)
                if not chunk:
                    break
                crc = zlib.crc32(chunk, crc)
        crc_checked += 1
        if crc & 0xFFFFFFFF != zmap[k][1]:
            crc_mismatch.append(k)

    res = {
        "zip_files": len(zmap),
        "disk_files": len(dmap),
        "missing": missing,
        "extra": extra,
        "size_mismatch": size_mismatch,
        "crc_mismatch": crc_mismatch,
        "crc_checked": crc_checked,
        "summary": {
            "missing_n": len(missing),
            "extra_n": len(extra),
            "size_mismatch_n": len(size_mismatch),
            "crc_mismatch_n": len(crc_mismatch),
            "ok": not missing and not extra and not size_mismatch and not crc_mismatch,
        },
    }
    if args.json:
        os.makedirs(os.path.dirname(args.json), exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=1)
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()

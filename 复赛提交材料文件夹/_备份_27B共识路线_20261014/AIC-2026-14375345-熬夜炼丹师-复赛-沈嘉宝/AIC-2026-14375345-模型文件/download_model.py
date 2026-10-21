# -*- coding: utf-8 -*-
"""下载模型权重（Qwen3.8-27B，ModelScope 通道，实测 16.7 MB/s）。

用法
    pip install modelscope
    python download_model.py            # 下载到 ./models/Qwen3.8-27B
    python download_model.py --verify   # 下载后逐分片比对 model_sha256.txt
"""
import argparse
import hashlib
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()
    out = Path("./models/Qwen3.8-27B")
    if not a.verify:
        from modelscope import snapshot_download
        snapshot_download("Qwen/Qwen3.8-27B", cache_dir="./models")
    want = {}
    for ln in Path("model_sha256.txt").read_text("utf-8").splitlines():
        h, name = ln.split()[0], ln.split()[-1].split("/")[-1]
        want[name] = h.lower()
    ok = True
    for name, h in sorted(want.items()):
        p = out / name
        if not p.exists():
            print(f"MISSING {name}")
            ok = False
            continue
        d = hashlib.sha256()
        with open(p, "rb") as f:
            for blk in iter(lambda: f.read(1 << 22), b""):
                d.update(blk)
        got = d.hexdigest()
        print(("OK   " if got == h else "FAIL ") + name)
        ok = ok and got == h
    print("ALL OK" if ok else "HAS FAILURE")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

# -*- coding: utf-8 -*-
"""下载本方案所用开源模型权重（不入包，按独立链接获取）。

用法
    pip install modelscope huggingface_hub
    python download_models.py            # 下载全部（按需注释掉不需要的）
    python download_models.py --verify   # 逐文件比对 model_sha256.txt

说明
    权重实体体积远超网盘单文件限制，故本包只提供下载脚本与逐文件 SHA256。
    交付件（0.7963）实际使用的模型是 Qwen/Qwen3.8-27B。
"""
import argparse
import hashlib
import sys
from pathlib import Path

MODELS = [
    # (本地目录, 仓库 id, 通道)
    ("Qwen3-VL-8B-Instruct", "Qwen/Qwen3-VL-8B-Instruct", "ms"),
    ("Qwen3-VL-32B-Instruct-AWQ", "QuantTrio/Qwen3-VL-32B-Instruct-AWQ", "ms"),
    ("Qwen3-VL-32B-Instruct", "Qwen/Qwen3-VL-32B-Instruct", "hf"),
    ("Qwen3.8-27B", "Qwen/Qwen3.8-27B", "hf"),
    ("LocateAnything-3B", "nv-community/LocateAnything-3B", "ms"),
]


def download(root: Path) -> None:
    for local, repo, chan in MODELS:
        dst = root / local
        print(f"[下载] {repo} -> {dst}")
        if chan == "ms":
            from modelscope import snapshot_download
            snapshot_download(repo, local_dir=str(dst))
        else:
            from huggingface_hub import snapshot_download
            snapshot_download(repo, local_dir=str(dst))


def verify(root: Path) -> int:
    want = {}
    for ln in Path("model_sha256.txt").read_text("utf-8").splitlines():
        if ln.startswith("#") or not ln.strip():
            continue
        h, rel = ln.split()[0], ln.split()[-1]
        want[rel] = h.lower()
    ok = True
    for rel, h in sorted(want.items()):
        p = root / rel
        if not p.exists():
            print(f"MISSING {rel}")
            ok = False
            continue
        d = hashlib.sha256()
        with open(p, "rb") as f:
            for blk in iter(lambda: f.read(1 << 22), b""):
                d.update(blk)
        got = d.hexdigest()
        print(("OK   " if got == h else "FAIL ") + rel)
        ok = ok and got == h
    print("ALL OK" if ok else "HAS FAILURE")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="./models")
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()
    root = Path(a.root)
    if a.verify:
        return verify(root)
    download(root)
    return 0


if __name__ == "__main__":
    sys.exit(main())

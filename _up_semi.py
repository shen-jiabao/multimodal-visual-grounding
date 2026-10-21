# -*- coding: utf-8 -*-
"""复赛数据集并行上传（断点续传）。

为什么单独写一个
    `_rc.py --put` 是单流 SFTP，1295 张 × ~3 MB 按 1 MB/s 要 65 分钟。本脚本用多
    条独立 SFTP 通道并行，并支持续传（远端大小一致即跳过），中断后可原样重跑。

传什么
    `queries/queries.json` + 各模态**被复赛 queries.json 实际引用**的图
    （visible 1295 张 = 3.93 GB；目录里另有 705 张/模态未被引用，不必传）。

用法
    $env:MATPOOL_PW='...'; $env:MATPOOL_HOST='...'; $env:MATPOOL_PORT='8000'
    python -B _up_semi.py --local <本地数据集根> --remote <远端数据集根> --mods visible
    python -B _up_semi.py ... --workers 8          # 并发通道数
    python -B _up_semi.py ... --limit 20           # 只传 20 张（测速）
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path

import paramiko

HOST = os.environ.get("MATPOOL_HOST", "")
PORT = int(os.environ.get("MATPOOL_PORT", "22"))
USER = os.environ.get("MATPOOL_USER", "root")

_lock = threading.Lock()
_done = {"n": 0, "bytes": 0, "skip": 0, "fail": 0}


def connect() -> paramiko.SSHClient:
    """建一条独立 SSH 连接（每条工作线程各持一条）。"""
    pw = os.environ.get("MATPOOL_PW")
    if not pw:
        sys.exit("错误：未设置 MATPOOL_PW")
    cli = paramiko.SSHClient()
    cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    cli.connect(HOST, port=PORT, username=USER, password=pw,
                timeout=30, banner_timeout=40, auth_timeout=40)
    return cli


def collect(local_root: Path, q: dict, mods: list) -> list:
    """展平成 [(本地绝对路径, 远端相对路径, 字节数)]。"""
    out = [(local_root / "queries" / "queries.json",
            "queries/queries.json",
            (local_root / "queries" / "queries.json").stat().st_size)]
    for mod in mods:
        rels = sorted({v[mod] for v in q.values()})
        miss = [r for r in rels if not (local_root / r).exists()]
        if miss:
            print(f"  ★ {mod}: {len(miss)} 张本地缺失，将跳过")
        for r in rels:
            lp = local_root / r
            if lp.exists():
                out.append((lp, r, lp.stat().st_size))
    return out


def makedirs(sftp: paramiko.SFTPClient, remote_root: str, rel: str,
             created: set) -> None:
    """在远端逐级建目录。★必须基于 `remote_root` 拼**绝对路径**——
    SFTP 的 CWD 是 `/root`，用相对路径会把目录建到别处（曾因此 20/20 全失败）。"""
    d = os.path.dirname(rel)
    if not d:
        return
    cur = remote_root.rstrip("/")
    for p in d.split("/"):
        cur = cur + "/" + p
        if cur not in created:
            try:
                sftp.mkdir(cur)
            except OSError:
                pass          # 已存在或并发建过，忽略
            created.add(cur)


def worker(batch: list, remote_root: str, total: int, t0: float,
           report_every: int) -> None:
    """一条通道：顺序上传自己那批文件。"""
    cli = connect()
    created: set = set()
    try:
        sftp = cli.open_sftp()
        sftp.get_channel().settimeout(120)
        for lp, rp, size in batch:
            rpath = remote_root.rstrip("/") + "/" + rp
            try:
                try:
                    if sftp.stat(rpath).st_size == size:
                        with _lock:
                            _done["skip"] += 1
                            _done["n"] += 1
                        continue
                except OSError:
                    pass
                makedirs(sftp, remote_root, rp, created)
                sftp.put(str(lp), rpath)
                with _lock:
                    _done["n"] += 1
                    _done["bytes"] += size
                    n = _done["n"]
                    if n % report_every == 0 or n == total:
                        el = time.time() - t0
                        sp = _done["bytes"] / el / 1024 / 1024 if el else 0
                        left = (total - n) / max(n / el, 1e-9) / 60 if el else 0
                        print(f"    {n}/{total}  {_done['bytes']/1024**2:.0f} MB  "
                              f"{sp:.2f} MB/s  跳过 {_done['skip']}  "
                              f"失败 {_done['fail']}  剩余约 {left:.0f} min",
                              flush=True)
            except Exception as exc:                       # noqa: BLE001
                with _lock:
                    _done["fail"] += 1
                    _done["n"] += 1
                print(f"    ★失败 {rp}: {type(exc).__name__}: {exc}", flush=True)
        sftp.close()
    finally:
        cli.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--local", required=True)
    ap.add_argument("--remote", required=True)
    ap.add_argument("--mods", default="visible")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--report-every", type=int, default=25)
    a = ap.parse_args()

    local_root = Path(a.local)
    q = json.loads((local_root / "queries" / "queries.json")
                   .read_text(encoding="utf-8"))
    mods = [m.strip() for m in a.mods.split(",") if m.strip()]
    files = collect(local_root, q, mods)
    if a.limit:
        files = files[:a.limit]
    total = len(files)
    tot_bytes = sum(f[2] for f in files)
    print(f"待上传 {total} 个文件，合计 {tot_bytes/1024**3:.2f} GB "
          f"→ {USER}@{HOST}:{a.remote}  ({a.workers} 通道)", flush=True)

    # 轮转分派，使各通道体量均衡（大图与小 json 混合）
    batches = [[] for _ in range(a.workers)]
    for i, f in enumerate(files):
        batches[i % a.workers].append(f)

    # 远端根目录必须先存在：`makedirs` 是从 remote_root 往下拼的，根不存在则整批失败
    cli0 = connect()
    try:
        sftp0 = cli0.open_sftp()
        cur = ""
        for p in a.remote.strip("/").split("/"):
            cur = (cur + "/" + p) if cur else p
            try:
                sftp0.mkdir(cur)
            except OSError:
                pass
        try:
            print(f"  远端根已就绪: {a.remote} "
                  f"（已存在条目 {len(sftp0.listdir(a.remote))} 个）", flush=True)
        except OSError as exc:
            sys.exit(f"★ 远端根不可访问: {exc}")
        sftp0.close()
    finally:
        cli0.close()

    t0 = time.time()
    ths = [threading.Thread(target=worker, args=(b, a.remote, total, t0,
                                                 a.report_every), daemon=True)
           for b in batches if b]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    el = time.time() - t0
    print(f"\n完成：{_done['n']}/{total}（跳过 {_done['skip']} 失败 {_done['fail']}）"
          f" 用时 {el/60:.1f} min  平均 {_done['bytes']/el/1024/1024:.2f} MB/s")
    return 0 if _done["fail"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
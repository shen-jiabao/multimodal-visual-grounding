# -*- coding: utf-8 -*-
"""上传 dataset.tar (12.2G) 到 px2 /srv/dataset.tar，带进度回调与重试。"""
import os
import socket
import sys
import time

import paramiko

SRC = r"d:\Desktop\2026AiC\dataset_rgb.tar"
DST = "/srv/dataset_rgb.tar"


def upload() -> None:
    host = os.environ["SSH_HOST"]
    port = int(os.environ["SSH_PORT"])
    user = os.environ["SSH_USER"]
    pw = os.environ["SSH_PW"]
    cli = paramiko.SSHClient()
    cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    cli.connect(host, port=port, username=user, password=pw,
                timeout=30, banner_timeout=40, auth_timeout=40)
    # 加大传输窗口/包大小：默认 2MB window 在跨地域链路吞吐极低
    tr = cli.get_transport()
    tr.default_window_size = 256 * 1024 * 1024
    tr.packet_size = 64 * 1024
    tr.window_size = 256 * 1024 * 1024
    size = os.path.getsize(SRC)
    print(f"[up] {SRC} {size/1e9:.2f} GB -> {DST}", flush=True)
    st = time.time()
    last_pct = -1
    last_t = st

    def cb(sent, total):
        nonlocal last_pct, last_t
        now = time.time()
        if now - last_t >= 30 or sent == total:
            pct = 100.0 * sent / total if total else 0
            speed = sent / 1e6 / max(now - st, 1e-6)
            print(f"[up] {sent/1e9:.2f}/{total/1e9:.2f} GB  "
                  f"{pct:.1f}%  {speed:.1f} MB/s  "
                  f"ETA {(total-sent)/1e6/max(speed,1e-6)/60:.1f} min", flush=True)
            last_t = now
            last_pct = pct

    try:
        sftp = cli.open_sftp()
        sftp.put(SRC, DST, callback=cb)
        sftp.close()
        print(f"[up] 完成 {size/1e9:.2f} GB 耗时 {time.time()-st:.0f}s", flush=True)
    except (OSError, socket.timeout, paramiko.SSHException) as ex:
        print(f"[up] FAILED: {ex!r}", flush=True)
        sys.exit(3)
    finally:
        cli.close()


if __name__ == "__main__":
    upload()
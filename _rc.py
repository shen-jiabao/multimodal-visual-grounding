# -*- coding: utf-8 -*-
"""远端执行助手（matpool 实例）。

安全约定
- **密码只从环境变量 `MATPOOL_PW` 读取，绝不写入任何文件**（项目硬约束：
  含明文凭据的临时脚本必须即时删除；本脚本因此设计为"零凭据"）。
- 每次调用建立独立连接，用完即关；远端命令的退出码原样返回。

用法
    $env:MATPOOL_PW='...'                     # 先注入密码（不落盘）
    python -B _rc.py "nvidia-smi"             # 执行命令
    python -B _rc.py --put 本地 远端          # 上传
    python -B _rc.py --get 远端 本地          # 下载
    python -B _rc.py --timeout 3600 "长命令"  # 长任务（默认 600s）

注意：必须用 `-B`（或设 PYTHONDONTWRITEBYTECODE=1），否则 paramiko 会尝试在
site-packages 写 .pyc，可能被沙箱拦截。
"""
from __future__ import annotations

import argparse
import os
import queue
import sys
import threading
import time

import paramiko

# 端点取自环境变量，缺省为当前实例。matpool 实例重建后 IP/端口会变，
# 换机器时只改环境变量，不改代码（含凭据的配置一律不落盘）。
HOST = os.environ.get("MATPOOL_HOST", "hz-t3.matpool.com")
PORT = int(os.environ.get("MATPOOL_PORT", "27852"))
USER = os.environ.get("MATPOOL_USER", "root")


def connect() -> paramiko.SSHClient:
    """建立 SSH 连接；密码取自环境变量 MATPOOL_PW。"""
    pw = os.environ.get("MATPOOL_PW")
    if not pw:
        sys.exit("错误：未设置环境变量 MATPOOL_PW")
    cli = paramiko.SSHClient()
    cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    cli.connect(HOST, port=PORT, username=USER, password=pw,
                timeout=25, banner_timeout=30, auth_timeout=30)
    return cli


def run(cmd: str, timeout: int = 600) -> int:
    """执行远端命令，回显 stdout/stderr，返回远端退出码。"""
    cli = connect()
    try:
        _, so, se = cli.exec_command(cmd, timeout=timeout)
        out = so.read().decode("utf-8", "replace")
        err = se.read().decode("utf-8", "replace")
        rc = so.channel.recv_exit_status()
    finally:
        cli.close()
    if out:
        print(out, end="" if out.endswith("\n") else "\n")
    if err:
        print(err, end="" if err.endswith("\n") else "\n", file=sys.stderr)
    return rc


def run_script(path: str, timeout: int = 600) -> int:
    """把本地脚本内容经 stdin 交给远端 `bash -s` 执行。

    这样可**彻底避开本地 shell 的引号解析**——本项目曾因内联引号畸形导致
    `_check_remote.py` 报 0/9555。远端命令一律走本函数。
    """
    body = open(path, "rb").read()
    cli = connect()
    try:
        si, so, se = cli.exec_command("bash -s", timeout=timeout)
        si.write(body)
        si.flush()
        si.channel.shutdown_write()
        out = so.read().decode("utf-8", "replace")
        err = se.read().decode("utf-8", "replace")
        rc = so.channel.recv_exit_status()
    finally:
        cli.close()
    if out:
        print(out, end="" if out.endswith("\n") else "\n")
    if err:
        print(err, end="" if err.endswith("\n") else "\n", file=sys.stderr)
    return rc


def put(local: str, remote: str) -> int:
    """上传本地文件/目录到远端。"""
    cli = connect()
    try:
        sftp = cli.open_sftp()
        if os.path.isdir(local):
            _put_dir(sftp, local, remote)
        else:
            sftp.put(local, remote)
        sftp.close()
    finally:
        cli.close()
    return 0


def _put_dir(sftp: paramiko.SFTPClient, local: str, remote: str) -> None:
    """递归上传目录（远端目录不存在则创建）。"""
    try:
        sftp.mkdir(remote)
    except OSError:
        pass
    for name in os.listdir(local):
        lp = os.path.join(local, name)
        rp = remote.rstrip("/") + "/" + name
        if os.path.isdir(lp):
            _put_dir(sftp, lp, rp)
        else:
            sftp.put(lp, rp)


def _rstat(sftp: paramiko.SFTPClient, path: str):
    """取远端文件大小；不存在返回 None。"""
    try:
        return sftp.stat(path).st_size
    except OSError:
        return None


def _collect(local: str, remote: str, only: set | None = None) -> list:
    """展平目录，返回 [(本地文件, 远端文件)]；`only` 给定时只保留其中文件名。"""
    out = []
    for name in sorted(os.listdir(local)):
        lp = os.path.join(local, name)
        rp = remote.rstrip("/") + "/" + name
        if os.path.isdir(lp):
            out.extend(_collect(lp, rp, only))
        elif only is None or name in only:
            out.append((lp, rp))
    return out


def sync(local: str, remote: str, jobs: int = 1,
         only: str | None = None) -> int:
    """断点续传式上传目录；可反复执行直至补齐。

    传输策略是**实测**出来的，三次试错：
    ① 单连接串行 0.15 MB/s；
    ② 16 条独立 SSH 连接被 sshd 拒（`Error reading SSH protocol
       banner`，MaxStartups 限流）；
    ③ 单连接开 8 条 SFTP 通道反而降到 0.14 MB/s——多通道共享同一 TCP
       拥塞窗口，不如独立连接。
    故采用**少量独立连接 + 工作队列**：某条连接建不起来只是少一个工人，
    其份额由其余工人从队列里领走，不会丢文件。`only` 给定时只传子集。
    """
    if not os.path.isdir(local):
        sys.exit("--sync 的本地参数必须是目录")
    allow = None
    if only:
        allow = {ln.strip() for ln in open(only, encoding="utf-8")
                 if ln.strip()}
        print(f"白名单 {len(allow)} 个文件名", flush=True)
    pairs = _collect(local, remote, allow)
    total = len(pairs)
    print(f"待同步 {total} 个文件，并发连接 {jobs}", flush=True)
    st = {"i": 0, "skip": 0, "sent": 0, "bytes": 0, "fail": 0,
          "t0": time.time(), "lock": threading.Lock()}
    work = queue.Queue()
    for p in pairs:
        work.put(p)

    # 先建好远端目录，避免每个工人各自 mkdir 竞争
    cli = connect()
    try:
        s0 = cli.open_sftp()
        _mkdirs(s0, remote)
        s0.close()
    finally:
        cli.close()

    def open_conn():
        """建连接，退避重试 5 次；全失败返回 None（不影响其他工人）。"""
        for k in range(5):
            try:
                c = connect()
                return c, c.open_sftp()
            except Exception as ex:
                if k == 4:
                    print(f"  [警告] 建连失败，放弃：{type(ex).__name__}",
                          flush=True)
                    return None, None
                time.sleep(3 * (k + 1))
        return None, None

    def worker() -> None:
        conn, sftp = open_conn()
        if conn is None:
            return
        try:
            while True:
                try:
                    lp, rp = work.get_nowait()
                except queue.Empty:
                    return
                ok = False
                for k in range(4):
                    try:
                        lsz = os.path.getsize(lp)
                        if _rstat(sftp, rp) == lsz:
                            with st["lock"]:
                                st["i"] += 1
                                st["skip"] += 1
                            ok = True
                            break
                        sftp.put(lp, rp)
                        with st["lock"]:
                            st["i"] += 1
                            st["sent"] += 1
                            st["bytes"] += lsz
                        ok = True
                        break
                    except Exception:
                        sftp.close()
                        conn.close()
                        time.sleep(2 ** k)
                        conn, sftp = open_conn()
                        if conn is None:
                            break
                if not ok:
                    with st["lock"]:
                        st["i"] += 1
                        st["fail"] += 1
                        st["bad"] = st.get("bad", []) + [os.path.basename(lp)]
                    print(f"  [失败] {os.path.basename(lp)}", flush=True)
                with st["lock"]:
                    if st["i"] % 50 == 0 or st["i"] == total:
                        el = time.time() - st["t0"]
                        print(f"  [{st['i']}/{total}] 传 {st['sent']} "
                              f"跳过 {st['skip']} 失败 {st['fail']} "
                              f"{st['bytes']/1e9:.2f}GB {el/60:.1f}min "
                              f"{st['bytes']/1e6/max(el,1e-9):.2f}MB/s",
                              flush=True)
        finally:
            try:
                sftp.close()
                conn.close()
            except Exception:
                pass

    ths = [threading.Thread(target=worker, daemon=True)
           for _ in range(min(jobs, total))]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    el = time.time() - st["t0"]
    print(f"同步完成：共 {total} 个，新传 {st['sent']}，跳过 {st['skip']}，"
          f"失败 {st['fail']}，{st['bytes']/1e9:.2f}GB / {el/60:.1f}min "
          f"({st['bytes']/1e6/max(el,1e-9):.2f}MB/s)")
    if st["fail"]:
        print("失败清单前 10:", st.get("bad", [])[:10])
    return 0


def _mkdirs(sftp: paramiko.SFTPClient, remote: str) -> None:
    """逐级创建远端目录。"""
    parts = remote.strip("/").split("/")
    cur = ""
    for p in parts:
        cur += "/" + p
        try:
            sftp.mkdir(cur)
        except OSError:
            pass


def get(remote: str, local: str) -> int:
    """下载远端文件到本地。"""
    cli = connect()
    try:
        sftp = cli.open_sftp()
        sftp.get(remote, local)
        sftp.close()
    finally:
        cli.close()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", help="要执行的远端命令")
    ap.add_argument("--script", help="本地 shell 脚本（经 stdin 交 bash -s 执行，推荐）")
    ap.add_argument("--put", nargs=2, metavar=("本地", "远端"))
    ap.add_argument("--get", nargs=2, metavar=("远端", "本地"))
    ap.add_argument("--sync", nargs=2, metavar=("本地目录", "远端目录"),
                    help="断点续传式上传：远端存在且大小相同则跳过")
    ap.add_argument("--jobs", type=int, default=6,
                    help="--sync 的并发 SSH 连接数（实测 16 条会被 sshd 拒）")
    ap.add_argument("--only", help="--sync 的文件名白名单（每行一个）")
    ap.add_argument("--timeout", type=int, default=600)
    a = ap.parse_args()
    if a.sync:
        return sync(*a.sync, jobs=a.jobs, only=a.only)
    if a.put:
        return put(*a.put)
    if a.get:
        return get(*a.get)
    if a.script:
        return run_script(a.script, a.timeout)
    if not a.cmd:
        ap.error("需要给出命令，或用 --script/--put/--get")
    return run(a.cmd, a.timeout)


if __name__ == "__main__":
    sys.exit(main())

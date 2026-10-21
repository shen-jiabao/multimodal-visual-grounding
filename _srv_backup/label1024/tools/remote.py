"""远程主机操作助手（paramiko）。

用途：在非交互环境里完成 SSH 登录、执行命令、传文件。
本机 ``ssh.exe`` 的密码认证需要交互式 TTY，一次性进程喂不进去；``sshpass``/``plink`` 未安装，
故统一走 paramiko（同一条通道既能 exec 也能 SFTP）。

凭据默认从仓库根 ``.remote.json`` 读取（已 gitignore）：:

    {"host": "...", "port": 22, "user": "root", "password": "..."}

若已装公钥，可用 ``--key <私钥路径>`` 免密登录，或让 ``password`` 留空。

用法::

    python tools/remote.py exec  --cmd "nvidia-smi -L"
    python tools/remote.py exec  --script <本地脚本>        # 以 bash -s 执行，免转义
    python tools/remote.py put   --local a.tar.gz --remote /root/a.tar.gz
    python tools/remote.py get   --remote /root/b.tar.gz --local b.tar.gz
    python tools/remote.py putdir  --local dir --remote /root/dir
    python tools/remote.py getdir  --remote /root/dir --local dir
    python tools/remote.py setup-key --pubkey ~/.ssh/id_ed25519.pub

退出码：远端命令的退出码即本进程退出码（连接/认证失败为 2）。
"""
from __future__ import annotations

import argparse
import json
import os
import posixpath
import stat
import sys
from typing import Iterable

import paramiko

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONF = os.path.join(ROOT, ".remote.json")


# ------------------------------------------------------------------ 连接
def load_conf(args) -> dict:
    conf: dict = {}
    if os.path.exists(CONF):
        with open(CONF, encoding="utf-8") as fh:
            conf = json.load(fh)
    for k in ("host", "user", "password", "key"):
        v = getattr(args, k, None)
        if v:
            conf[k] = v
    if getattr(args, "port", None):
        conf["port"] = args.port
    conf.setdefault("port", 22)
    conf.setdefault("user", "root")
    if not conf.get("host"):
        sys.exit("缺少 host：请在 .remote.json 或 --host 指定")
    return conf


def connect(conf: dict) -> paramiko.SSHClient:
    cli = paramiko.SSHClient()
    cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    kw = {
        "hostname": conf["host"],
        "port": int(conf["port"]),
        "username": conf["user"],
        "timeout": 30,
        "banner_timeout": 30,
        "auth_timeout": 30,
        "allow_agent": False,
        "look_for_keys": False,
    }
    key = conf.get("key")
    if key:
        key = os.path.expanduser(key)
        kw["key_filename"] = key
    elif conf.get("password"):
        kw["password"] = conf["password"]
    cli.connect(**kw)
    return cli


# ------------------------------------------------------------------ exec
def run(cli: paramiko.SSHClient, cmd: str, pty: bool = False,
        timeout: float | None = None) -> int:
    stdin, stdout, stderr = cli.exec_command(cmd, get_pty=pty, timeout=timeout)
    stdin.close()
    out = stdout.read().decode("utf-8", "replace")
    err = stderr.read().decode("utf-8", "replace")
    code = stdout.channel.recv_exit_status()
    if out:
        sys.stdout.write(out)
        if not out.endswith("\n"):
            sys.stdout.write("\n")
    if err.strip():
        sys.stdout.write("[stderr]\n" + err)
        if not err.endswith("\n"):
            sys.stdout.write("\n")
    if code != 0:
        sys.stdout.write(f"[exit code: {code}]\n")
    return code


def run_script(cli: paramiko.SSHClient, path: str, timeout: float | None) -> int:
    with open(path, encoding="utf-8") as fh:
        body = fh.read()
    # 以 bash -s 读 stdin，彻底回避引号/换行转义问题
    transport = cli.get_transport()
    chan = transport.open_session()
    chan.settimeout(timeout)
    chan.exec_command("bash -s")
    chan.sendall(body.encode("utf-8"))
    chan.shutdown_write()
    out = b""
    err = b""
    while True:
        if chan.recv_ready():
            out += chan.recv(65536)
        elif chan.recv_stderr_ready():
            err += chan.recv_stderr(65536)
        elif chan.exit_status_ready():
            while chan.recv_ready():
                out += chan.recv(65536)
            while chan.recv_stderr_ready():
                err += chan.recv_stderr(65536)
            break
        else:
            import time
            time.sleep(0.05)
    code = chan.recv_exit_status()
    chan.close()
    sys.stdout.write(out.decode("utf-8", "replace"))
    if err.strip():
        sys.stdout.write("[stderr]\n" + err.decode("utf-8", "replace"))
    if code != 0:
        sys.stdout.write(f"[exit code: {code}]\n")
    return code


# ------------------------------------------------------------------ 传输
def _sftp(cli):
    return cli.open_sftp()


def put_file(cli, local: str, remote: str) -> None:
    sf = _sftp(cli)
    d = posixpath.dirname(remote)
    _mkdirs(sf, d)
    sf.put(local, remote)
    size = sf.stat(remote).st_size
    print(f"put  {local}  ->  {remote}  ({size} B)")
    sf.close()


def get_file(cli, remote: str, local: str) -> None:
    sf = _sftp(cli)
    os.makedirs(os.path.dirname(os.path.abspath(local)) or ".", exist_ok=True)
    sf.get(remote, local)
    print(f"get  {remote}  ->  {local}  ({os.path.getsize(local)} B)")
    sf.close()


def _mkdirs(sf, path: str) -> None:
    if not path or path == "/":
        return
    parts = path.strip("/").split("/")
    cur = "/" if path.startswith("/") else ""
    for p in parts:
        cur = posixpath.join(cur, p) if cur else p
        try:
            sf.stat(cur)
        except IOError:
            sf.mkdir(cur)


def _walk_local(base: str) -> Iterable[tuple[str, str]]:
    for dp, dns, fns in os.walk(base):
        rel = os.path.relpath(dp, base)
        yield dp, rel
        for fn in fns:
            yield dp, os.path.join(rel, fn) if rel != "." else fn


def put_dir(cli, local: str, remote: str) -> None:
    sf = _sftp(cli)
    _mkdirs(sf, remote)
    n = 0
    for dp, rel in _walk_local(local):
        rdir = remote if rel == "." else posixpath.join(remote, rel.replace(os.sep, "/"))
        _mkdirs(sf, rdir)
        for fn in os.listdir(dp):
            lp = os.path.join(dp, fn)
            if os.path.isfile(lp):
                sf.put(lp, posixpath.join(rdir, fn))
                n += 1
    print(f"putdir  {local}  ->  {remote}  ({n} files)")
    sf.close()


def get_dir(cli, remote: str, local: str) -> None:
    sf = _sftp(cli)
    n = 0

    def rec(rp: str, lp: str) -> None:
        nonlocal n
        os.makedirs(lp, exist_ok=True)
        for e in sf.listdir_attr(rp):
            rchild = posixpath.join(rp, e.filename)
            lchild = os.path.join(lp, e.filename)
            if stat.S_ISDIR(e.st_mode):
                rec(rchild, lchild)
            else:
                sf.get(rchild, lchild)
                n += 1

    rec(remote, local)
    print(f"getdir  {remote}  ->  {local}  ({n} files)")
    sf.close()


# ------------------------------------------------------------------ 公钥
def setup_key(cli, pubkey: str) -> int:
    pubkey = os.path.expanduser(pubkey)
    with open(pubkey, encoding="utf-8") as fh:
        key = fh.read().strip()
    cmd = (
        "mkdir -p ~/.ssh && chmod 700 ~/.ssh && touch ~/.ssh/authorized_keys && "
        "chmod 600 ~/.ssh/authorized_keys && "
        f"grep -qxF '{key}' ~/.ssh/authorized_keys || echo '{key}' >> ~/.ssh/authorized_keys"
    )
    rc = run(cli, cmd)
    if rc == 0:
        run(cli, "grep -c . ~/.ssh/authorized_keys")
    return rc


# ------------------------------------------------------------------ main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host")
    ap.add_argument("--port", type=int)
    ap.add_argument("--user")
    ap.add_argument("--password")
    ap.add_argument("--key", help="私钥路径（免密登录）")
    sub = ap.add_subparsers(dest="action", required=True)

    p = sub.add_parser("exec", help="执行远端命令")
    p.add_argument("--cmd")
    p.add_argument("--script", help="本地脚本文件，以 bash -s 执行")
    p.add_argument("--pty", action="store_true")
    p.add_argument("--timeout", type=float)

    p = sub.add_parser("put", help="上传单个文件")
    p.add_argument("--local", required=True)
    p.add_argument("--remote", required=True)

    p = sub.add_parser("get", help="下载单个文件")
    p.add_argument("--remote", required=True)
    p.add_argument("--local", required=True)

    p = sub.add_parser("putdir", help="递归上传目录")
    p.add_argument("--local", required=True)
    p.add_argument("--remote", required=True)

    p = sub.add_parser("getdir", help="递归下载目录")
    p.add_argument("--remote", required=True)
    p.add_argument("--local", required=True)

    p = sub.add_parser("setup-key", help="把本地公钥写入远端 authorized_keys")
    p.add_argument("--pubkey", default=os.path.expanduser("~/.ssh/id_ed25519.pub"))

    args = ap.parse_args()
    conf = load_conf(args)

    try:
        cli = connect(conf)
    except Exception as exc:
        print(f"连接/认证失败：{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    try:
        if args.action == "exec":
            if args.script:
                return run_script(cli, args.script, args.timeout)
            if not args.cmd:
                sys.exit("exec 需要 --cmd 或 --script")
            return run(cli, args.cmd, pty=args.pty, timeout=args.timeout)
        if args.action == "put":
            put_file(cli, args.local, args.remote)
        elif args.action == "get":
            get_file(cli, args.remote, args.local)
        elif args.action == "putdir":
            put_dir(cli, args.local, args.remote)
        elif args.action == "getdir":
            get_dir(cli, args.remote, args.local)
        elif args.action == "setup-key":
            return setup_key(cli, args.pubkey)
        return 0
    finally:
        cli.close()


if __name__ == "__main__":
    sys.exit(main())

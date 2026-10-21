"""云端部署 / 数据搬运 CLI（基于 paramiko），供 SSH 到手后驱动云端操作。

凭据从命令行或环境变量读取，**绝不写入仓库**：
    --host/--port/--user/--password   或   SSH_HOST / SSH_PORT / SSH_USER / SSH_PASSWORD

子命令：
    check                                   采集远端环境（hostname/uname/nvidia-smi/df/python/数据目录）
    run   --cmd "..."                       在远端执行一条命令并打印输出
    put   --local F --remote F              sftp 上传单文件
    get   --remote F --local F              sftp 下载单文件
    upload-code [--local-root .] [--remote-dir /data/coding/aic-grounding]
                                            打包本地代码（排除大目录）→ 上传 → 远端解包
    upload-data --local-dir dataset [--remote-dir /data/dataset] [--chunk-mb 1500]
                                            打包数据 → 分块上传 → 远端合并 + MD5 校验 → 解包

示例：
    python tools/deploy.py --host H --port 22 --user root --password PW check
    python tools/deploy.py ... upload-data --local-dir dataset --remote-dir /data/dataset
"""
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import io
import json
import os
import sys
import tarfile
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor

# 打包代码时排除的大目录 / 产物
EXCLUDE_CODE = {".git", ".venv", "checkpoints", "dataset", "data", "outputs",
                "__pycache__", ".idea", ".vscode", ".pytest_cache", ".mypy_cache",
                "*.zip", "*.tar", "*.tar.gz"}
# 打包数据时的排除
EXCLUDE_DATA = {".git", "__pycache__", ".DS_Store"}


# ---------------- 连接 ----------------
def _make_client(args):
    """建立一条 SSH 连接（不打印），供并行上传的多连接复用。"""
    try:
        import paramiko
    except ImportError:
        raise SystemExit("缺少 paramiko：pip install paramiko")
    host = args.host or os.environ.get("SSH_HOST")
    port = int(args.port or os.environ.get("SSH_PORT") or 22)
    user = args.user or os.environ.get("SSH_USER") or "root"
    pw = args.password or os.environ.get("SSH_PASSWORD")
    if not host or not pw:
        raise SystemExit("缺少连接信息：请用 --host/--password 或环境变量 SSH_HOST/SSH_PASSWORD")
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(host, port=port, username=user, password=pw, timeout=30)
    return client


def _connect(args):
    client = _make_client(args)
    print(f"# 已连接 {args.user or os.environ.get('SSH_USER') or 'root'}@"
          f"{args.host or os.environ.get('SSH_HOST')}:"
          f"{int(args.port or os.environ.get('SSH_PORT') or 22)}")
    return client


def run_cmd(client, cmd: str, timeout: int = 600):
    _stdin, stdout, stderr = client.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode("utf-8", "replace")
    err = stderr.read().decode("utf-8", "replace")
    code = stdout.channel.recv_exit_status()
    return code, out, err


def _md5_file(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _make_tar(src_root: str, tar_path: str, excludes: set, include_dir_name: bool) -> str:
    """打包目录。

    ``excludes`` 语义：
      - 不含通配符的条目 = **顶层**目录名（仅第一层剪枝，不影响嵌套同名目录，如 ``src/data``）；
      - 含通配符的条目 = 文件 glob（任意层级，如 ``*.zip``）；
      - ``__pycache__`` 任意层级剪枝。
    """
    top_dirs = {e for e in excludes if not any(c in e for c in "*?[")}
    file_globs = [e for e in excludes if any(c in e for c in "*?[")]
    src_abs = os.path.abspath(src_root)
    base = os.path.basename(src_abs.rstrip("/\\"))
    with tarfile.open(tar_path, "w") as tf:
        for root, dirs, files in os.walk(src_root):
            at_top = os.path.abspath(root) == src_abs
            dirs[:] = [
                d for d in dirs
                if d != "__pycache__" and not (at_top and d in top_dirs)
            ]
            for f in files:
                if f.endswith((".pyc", ".pyo")):
                    continue
                if any(fnmatch.fnmatch(f, g) for g in file_globs):
                    continue
                full = os.path.join(root, f)
                rel = os.path.relpath(full, src_root)
                arc = os.path.join(base, rel) if include_dir_name else rel
                tf.add(full, arcname=arc)
    return tar_path


def _split_file(path: str, chunk_bytes: int):
    parts = []
    i = 0
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk_bytes)
            if not block:
                break
            part = f"{path}.part{i:03d}"
            with open(part, "wb") as g:
                g.write(block)
            parts.append(part)
            i += 1
    return parts


def _git_info(cwd: str) -> dict:
    """采集本地 git 来源（上传代码时附带 _code_origin.json，供云端快照回填）。"""
    import subprocess

    def g(*a):
        try:
            return subprocess.run(["git", *a], cwd=cwd, capture_output=True,
                                  text=True, timeout=30).stdout.strip()
        except Exception:
            return ""

    return {
        "git_head": g("rev-parse", "HEAD"),
        "git_branch": g("rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty": [l for l in g("status", "--porcelain").splitlines() if l.strip()],
    }


# ---------------- 子命令 ----------------
def cmd_check(client, args):
    cmds = [
        "hostname && uname -a",
        "nvidia-smi",
        "df -h / /data 2>/dev/null || df -h",
        "which python python3; python3 --version",
        "ls -d /data/* 2>/dev/null | head -20",
    ]
    for c in cmds:
        code, out, err = run_cmd(client, c, timeout=60)
        print(f"=== {c} (exit {code}) ===")
        print(out.strip())
        if err.strip():
            print("STDERR:", err.strip()[:400])


def cmd_run(client, args):
    code, out, err = run_cmd(client, args.cmd, timeout=args.timeout)
    if out:
        print(out, end="")
    if err.strip():
        print("STDERR:", err.strip(), file=sys.stderr)
    sys.exit(code)


def cmd_put(client, args):
    sftp = client.open_sftp()
    sftp.put(args.local, args.remote)
    print(f"# 已上传 {args.local} → {args.remote}")


def cmd_get(client, args):
    sftp = client.open_sftp()
    os.makedirs(os.path.dirname(os.path.abspath(args.local)), exist_ok=True)
    sftp.get(args.remote, args.local)
    print(f"# 已下载 {args.remote} → {args.local}")


def cmd_upload_code(client, args):
    work = getattr(args, "work_dir", None) or tempfile.gettempdir()
    os.makedirs(work, exist_ok=True)
    tar_path = os.path.join(work, f"code_{int(time.time())}.tar")
    _make_tar(args.local_root, tar_path, EXCLUDE_CODE | set(args.exclude or []), include_dir_name=False)
    # 附带本地代码来源（云端无 .git 时，code_snapshot 可回退读取）
    origin = _git_info(args.local_root)
    with tarfile.open(tar_path, "a") as tf:
        payload = json.dumps(origin, ensure_ascii=False, indent=1).encode("utf-8")
        ti = tarfile.TarInfo("_code_origin.json")
        ti.size = len(payload)
        tf.addfile(ti, io.BytesIO(payload))
    print(f"# 代码包：{tar_path}（{os.path.getsize(tar_path) // 1024} KB，"
          f"origin={origin.get('git_head', '?')[:12]} dirty={len(origin.get('git_dirty', []))}）")
    remote_tar = f"/tmp/{os.path.basename(tar_path)}"
    client.open_sftp().put(tar_path, remote_tar)
    code, out, err = run_cmd(
        client,
        f"mkdir -p '{args.remote_dir}' && tar -xf '{remote_tar}' -C '{args.remote_dir}' "
        f"&& rm -f '{remote_tar}' && echo EXTRACT_OK",
        timeout=300,
    )
    print(out.strip(), err.strip())
    if "EXTRACT_OK" not in out:
        raise SystemExit("远端解包失败")
    print(f"# 代码已就位：{args.remote_dir}")
    try:
        os.remove(tar_path)
    except OSError:
        pass


def cmd_upload_data(client, args):
    work = getattr(args, "work_dir", None) or tempfile.gettempdir()
    os.makedirs(work, exist_ok=True)
    tar_path = os.path.join(work, f"data_{int(time.time())}.tar")
    _make_tar(args.local_dir, tar_path, EXCLUDE_DATA | set(args.exclude or []), include_dir_name=False)
    local_md5 = _md5_file(tar_path)
    size_gb = os.path.getsize(tar_path) / (1024 ** 3)
    print(f"# 数据包：{tar_path}（{size_gb:.2f} GB，md5={local_md5}）")

    tmp = args.remote_tmp.rstrip("/")
    run_cmd(client, f"mkdir -p '{tmp}' '{args.remote_dir}'", timeout=60)

    parts = _split_file(tar_path, args.chunk_mb * 1024 * 1024)
    remote_parts = [f"{tmp}/{os.path.basename(p)}" for p in parts]
    print(f"# 分块 {len(parts)} 个，并行 {args.parallel} 路上传 ...")
    # 分块完成即删本地 tar，释放本地空间
    try:
        os.remove(tar_path)
        print("# 已删本地 tar（分块完成）")
    except OSError:
        pass

    def _upload(item):
        i, local_p = item
        wc = _make_client(args)          # 每路独立连接
        try:
            wc.open_sftp().put(local_p, remote_parts[i - 1])
        finally:
            wc.close()
        try:
            os.remove(local_p)           # 该分块上传完成，删本地副本
        except OSError:
            pass
        print(f"# 分块 {i}/{len(parts)} 完成：{os.path.basename(local_p)}")

    with ThreadPoolExecutor(max_workers=max(1, args.parallel)) as ex:
        list(ex.map(_upload, enumerate(parts, 1)))

    remote_all = f"{tmp}/{os.path.basename(tar_path)}"
    run_cmd(client, "cat " + " ".join(remote_parts) + f" > {remote_all}", timeout=600)
    # 合并后立即删分块，压低峰值占用
    run_cmd(client, "rm -f " + " ".join(remote_parts), timeout=120)
    _code, out, _err = run_cmd(client, f"md5sum {remote_all}", timeout=120)
    remote_md5 = out.split()[0] if out.strip() else ""
    print(f"# md5 本地={local_md5} 远端={remote_md5} 一致={local_md5 == remote_md5}")
    if local_md5 != remote_md5:
        raise SystemExit(f"MD5 不一致，中止（tar 保留在 {remote_all} 供排查）")

    code, out, err = run_cmd(
        client,
        f"tar -xf {remote_all} -C '{args.remote_dir}' && echo EXTRACT_OK",
        timeout=1800,
    )
    print(out.strip(), err.strip())
    if "EXTRACT_OK" not in out:
        raise SystemExit("远端解包失败")

    # 解包后立即删 tar
    run_cmd(client, f"rm -f {remote_all}", timeout=120)
    print(f"# 数据已解包：{args.remote_dir}（远端临时 {tmp} 已清理）")
    for p in list(parts) + [tar_path]:
        try:
            os.remove(p)
        except OSError:
            pass


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="云端部署 / 数据搬运 CLI")
    p.add_argument("--host", default=None)
    p.add_argument("--port", default=None)
    p.add_argument("--user", default=None)
    p.add_argument("--password", default=None)
    sub = p.add_subparsers(dest="action", required=True)

    sub.add_parser("check")

    r = sub.add_parser("run")
    r.add_argument("--cmd", required=True)
    r.add_argument("--timeout", type=int, default=600)

    pu = sub.add_parser("put")
    pu.add_argument("--local", required=True)
    pu.add_argument("--remote", required=True)

    ge = sub.add_parser("get")
    ge.add_argument("--remote", required=True)
    ge.add_argument("--local", required=True)

    uc = sub.add_parser("upload-code")
    uc.add_argument("--local-root", default=".")
    uc.add_argument("--remote-dir", default="/data/coding/aic-grounding")
    uc.add_argument("--exclude", nargs="*", default=[], help="额外排除的目录名（如 初赛）")
    uc.add_argument("--work-dir", default=None, help="本地临时目录（默认系统 temp；系统盘满时改用大磁盘）")

    ud = sub.add_parser("upload-data")
    ud.add_argument("--local-dir", default="dataset")
    ud.add_argument("--remote-dir", default="/data/dataset")
    ud.add_argument("--chunk-mb", type=int, default=1500)
    ud.add_argument("--exclude", nargs="*", default=[], help="额外排除的目录名（如 初赛）")
    ud.add_argument("--remote-tmp", default="/tmp", help="远端临时目录（大文件放大数据盘，如 /data/_tmp）")
    ud.add_argument("--parallel", type=int, default=4, help="并行上传连接数（默认 4）")
    ud.add_argument("--work-dir", default=None, help="本地临时目录（默认系统 temp；系统盘满时改用大磁盘）")
    return p


def main() -> None:
    args = build_parser().parse_args()
    client = _connect(args)
    try:
        {
            "check": cmd_check,
            "run": cmd_run,
            "put": cmd_put,
            "get": cmd_get,
            "upload-code": cmd_upload_code,
            "upload-data": cmd_upload_data,
        }[args.action](client, args)
    finally:
        client.close()


if __name__ == "__main__":
    main()

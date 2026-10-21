# -*- coding: utf-8 -*-
"""上传本地脚本到远端并执行（凭据从环境变量读取）。

用法（PowerShell）：
    $env:SSH_PW='<password>'; python -B _remote_run_file.py <local_script> [remote_path]
"""
import os
import sys

import paramiko


def main() -> int:
    host = os.environ.get("SSH_HOST", "px-cloud1.matpool.com")
    port = int(os.environ.get("SSH_PORT", "27675"))
    user = os.environ.get("SSH_USER", "root")
    pw = os.environ.get("SSH_PW")
    if not pw or len(sys.argv) < 2:
        print("用法: SSH_PW=... python _remote_run_file.py <local_script> [remote_path]",
              file=sys.stderr)
        return 2
    local = sys.argv[1]
    remote = sys.argv[2] if len(sys.argv) > 2 else "/root/_inv.sh"

    cli = paramiko.SSHClient()
    cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    cli.connect(host, port=port, username=user, password=pw,
                timeout=20, banner_timeout=30, auth_timeout=30)
    sftp = cli.open_sftp()
    sftp.put(local, remote)
    sftp.close()
    _, out, err = cli.exec_command(f"bash {remote}", timeout=600)
    o = out.read().decode("utf-8", "replace")
    e = err.read().decode("utf-8", "replace")
    print(o, end="")
    if e.strip():
        print("--- stderr ---")
        print(e, end="")
    cli.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

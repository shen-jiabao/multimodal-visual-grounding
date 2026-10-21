# -*- coding: utf-8 -*-
"""远端命令执行器（凭据从环境变量读取，不落盘）。

用法（PowerShell）：
    $env:SSH_PW='<password>'; python _remote_exec.py "<remote shell command>"

环境变量：
    SSH_HOST  默认 px-cloud2.matpool.com
    SSH_PORT  默认 29833
    SSH_USER  默认 root
    SSH_PW    必填（口令）
"""
import os
import sys

import paramiko


def main() -> int:
    host = os.environ.get("SSH_HOST", "px-cloud2.matpool.com")
    port = int(os.environ.get("SSH_PORT", "29833"))
    user = os.environ.get("SSH_USER", "root")
    pw = os.environ.get("SSH_PW")
    if not pw:
        print("缺少环境变量 SSH_PW", file=sys.stderr)
        return 2
    cmd = sys.argv[1] if len(sys.argv) > 1 else "echo no-command"

    cli = paramiko.SSHClient()
    cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    cli.connect(host, port=port, username=user, password=pw,
                timeout=20, banner_timeout=30, auth_timeout=30)
    _, out, err = cli.exec_command(cmd, timeout=300)
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

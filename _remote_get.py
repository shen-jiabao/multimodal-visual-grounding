# -*- coding: utf-8 -*-
"""从远端下载文件到本地（通用小工具）。

用法: SSH_PW=... python _remote_get.py <remote_path> <local_path>
"""
import os
import sys

import paramiko


def main() -> int:
    """下载单个远端文件。"""
    if len(sys.argv) != 3:
        print("用法: python _remote_get.py <remote> <local>", file=sys.stderr)
        return 2
    remote, local = sys.argv[1], sys.argv[2]
    cli = paramiko.SSHClient()
    cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    cli.connect(os.environ.get("SSH_HOST", "px-cloud1.matpool.com"),
                port=int(os.environ.get("SSH_PORT", "27675")),
                username=os.environ.get("SSH_USER", "root"),
                password=os.environ["SSH_PW"], timeout=20)
    sftp = cli.open_sftp()
    sftp.get(remote, local)
    sftp.close()
    cli.close()
    print(f"下载 {remote} -> {local}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
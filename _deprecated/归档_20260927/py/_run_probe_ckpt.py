# -*- coding: utf-8 -*-
"""上传探针脚本到远端并执行。"""
import os

import paramiko


def main() -> int:
    host = os.environ.get("SSH_HOST", "px-cloud1.matpool.com")
    port = int(os.environ.get("SSH_PORT", "27675"))
    user = os.environ.get("SSH_USER", "root")
    pw = os.environ.get("SSH_PW")
    cli = paramiko.SSHClient()
    cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    cli.connect(host, port=port, username=user, password=pw,
                timeout=25, banner_timeout=30, auth_timeout=30)
    sftp = cli.open_sftp()
    sftp.put(r"d:\Desktop\2026AiC\_probe_ckpt.py", "/root/infer/_probe_ckpt.py")
    sftp.close()
    _, out, err = cli.exec_command("python3 /root/infer/_probe_ckpt.py", timeout=60)
    print(out.read().decode("utf-8", "replace"))
    e = err.read().decode("utf-8", "replace")
    if e.strip():
        print("-- stderr --\n" + e)
    cli.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
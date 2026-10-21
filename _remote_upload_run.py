# -*- coding: utf-8 -*-
"""上传多个本地文件到远端并执行远端 bash 脚本（凭据走环境变量）。

用法（PowerShell）：
    $env:SSH_PW='<password>'; python -B _remote_upload_run.py <local_bash_script> <local_file1> [<local_file2> ...]
上传文件到 /root/ 同名，随后 bash <bash_script>。
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
        print("用法: SSH_PW=... python _remote_upload_run.py <bash_script> <file1> [file2 ...]",
              file=sys.stderr)
        return 2
    script = sys.argv[1]
    files = sys.argv[2:]

    cli = paramiko.SSHClient()
    cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    cli.connect(host, port=port, username=user, password=pw,
                timeout=20, banner_timeout=30, auth_timeout=30)
    sftp = cli.open_sftp()
    for f in files:
        remote = "/root/" + os.path.basename(f)
        sftp.put(f, remote)
        print(f"上传 {f} -> {remote}", file=sys.stderr)
    sftp.close()
    _, out, err = cli.exec_command(f"bash /root/{os.path.basename(script)}", timeout=3000)
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
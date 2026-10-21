# -*- coding: utf-8 -*-
"""上传并运行 SoM 探针。"""
import os
import paramiko


def main() -> int:
    cli = paramiko.SSHClient()
    cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    cli.connect(os.environ["SSH_HOST"], port=int(os.environ["SSH_PORT"]),
                username=os.environ["SSH_USER"],
                password=os.environ["SSH_PW"], timeout=25,
                banner_timeout=30, auth_timeout=30)
    sftp = cli.open_sftp()
    sftp.put(r"d:\Desktop\2026AiC\_probe_som.py", "/root/_probe_som.py")
    sftp.close()
    print("已上传 _probe_som.py")
    _, out, err = cli.exec_command(
        "cd /root && /srv/venv-locate/bin/python -u _probe_som.py 2>&1 | tail -40",
        timeout=900)
    print(out.read().decode("utf-8", "replace"))
    cli.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
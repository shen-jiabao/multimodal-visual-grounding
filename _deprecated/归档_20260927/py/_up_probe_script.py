import os
import paramiko


def main() -> int:
    cli = paramiko.SSHClient()
    cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    cli.connect(os.environ["SSH_HOST"], port=int(os.environ["SSH_PORT"]),
                username=os.environ["SSH_USER"],
                password=os.environ["SSH_PW"], timeout=30,
                banner_timeout=40, auth_timeout=40)
    sftp = cli.open_sftp()
    sftp.put(r"d:\Desktop\2026AiC\_probe_locale.py", "/root/_probe_locale.py")
    sftp.close()
    print("已上传 _probe_locale.py")
    cli.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
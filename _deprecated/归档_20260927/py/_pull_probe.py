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
    sftp.get("/srv/output/probe_result.json",
             r"d:\Desktop\2026AiC\probe_result.json")
    sftp.close()
    print("已拉回 probe_result.json")
    cli.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
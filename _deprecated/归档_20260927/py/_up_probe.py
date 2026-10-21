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
    sftp.put(r"d:\Desktop\2026AiC\probe.tar", "/srv/probe.tar")
    sftp.close()
    _, out, err = cli.exec_command("rm -rf /srv/dataset_probe && mkdir -p /srv/dataset_probe && "
                                   "tar -xf /srv/probe.tar -C /srv/dataset_probe && "
                                   "du -sh /srv/dataset_probe && ls /srv/dataset_probe", timeout=60)
    print(out.read().decode("utf-8", "replace"))
    e = err.read().decode("utf-8", "replace")
    if e.strip():
        print("-- stderr --\n" + e)
    cli.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
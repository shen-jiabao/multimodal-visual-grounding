import os
import paramiko


def main():
    cli = paramiko.SSHClient()
    cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    cli.connect(os.environ["SSH_HOST"], port=int(os.environ["SSH_PORT"]),
                username=os.environ["SSH_USER"],
                password=os.environ["SSH_PW"], timeout=25,
                banner_timeout=30, auth_timeout=30)

    def run(cmd, t=60):
        _, out, err = cli.exec_command(cmd, timeout=t)
        return out.read().decode("utf-8", "replace"), \
            err.read().decode("utf-8", "replace")

    print("== setup2.log tail ==")
    so, _ = run("tail -n 15 /root/setup2.log 2>/dev/null")
    print(so.strip() or "(无)")

    print("== 模型目录 ==")
    so, _ = run("du -sh /srv/models/LocateAnything-3B 2>/dev/null; "
                "ls /srv/models/LocateAnything-3B 2>/dev/null | head")
    print(so.strip() or "(无)")

    print("== 上传中的 dataset.tar ==")
    so, _ = run("ls -la /srv/dataset.tar 2>/dev/null")
    print(so.strip() or "(无)")

    print("== 进程 ==")
    so, _ = run("pgrep -af 'snapshot_download|modelscope|pip install' | head -5")
    print(so.strip() or "(无)")
    cli.close()


if __name__ == "__main__":
    main()
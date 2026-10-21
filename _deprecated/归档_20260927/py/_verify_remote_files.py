import hashlib
import os
import paramiko


def main() -> int:
    cli = paramiko.SSHClient()
    cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    cli.connect(os.environ["SSH_HOST"], port=int(os.environ["SSH_PORT"]),
                username=os.environ["SSH_USER"],
                password=os.environ["SSH_PW"], timeout=30,
                banner_timeout=40, auth_timeout=40)

    def run(cmd, t=60):
        _, out, err = cli.exec_command(cmd, timeout=t)
        return out.read().decode("utf-8", "replace"), \
            err.read().decode("utf-8", "replace")

    print("== 远端启动器 ==")
    so, _ = run("ls -la /root/_run_locate_full.sh; test -x /root/_run_locate_full.sh && echo EXEC_OK || echo NOT_EXEC; wc -l /root/_run_locate_full.sh")
    print(so.strip())
    print("== 远端脚本 ==")
    so, _ = run("ls -la /root/_locate_full.py; wc -l /root/_locate_full.py")
    print(so.strip())
    print("== 远端校验行(关键逻辑) ==")
    so, _ = run("grep -n 'NIMG=\\|maxdepth\\|wc -l\\|png.*jpg' /root/_run_locate_full.sh | head")
    print(so.strip() or "(无匹配)")
    print("== 远端 _locate_full.py 路径来源 ==")
    so, _ = run("grep -n 'rec\\[\"visible\"\\]\\|Image.open\\|glob\\|\\\\.png\\|\\\\.jpg' /root/_locate_full.py | head")
    print(so.strip() or "(无匹配)")
    print("== 数据等待器状态 ==")
    so, _ = run("tail -n 4 /root/data_wait.log 2>/dev/null")
    print(so.strip() or "(启动器未运行?)")
    cli.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
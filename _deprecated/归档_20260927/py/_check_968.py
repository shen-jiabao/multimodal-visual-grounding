import os
import paramiko


def run(cmd, t=60):
    _, out, err = cli.exec_command(cmd, timeout=t)
    return out.read().decode("utf-8", "replace"), \
        err.read().decode("utf-8", "replace")


cli = paramiko.SSHClient()
cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
cli.connect(os.environ["SSH_HOST"], port=int(os.environ["SSH_PORT"]),
            username=os.environ["SSH_USER"],
            password=os.environ["SSH_PW"], timeout=25,
            banner_timeout=30, auth_timeout=30)

print("== 000968 文件 ==")
so, _ = run("ls -la /srv/dataset/Images/visible/000968* 2>/dev/null || echo '(不存在)'")
print(so.strip())

print("== 远端 visible 计数 ==")
# 用 find 数（注意避免引号地狱，用单层单引号包裹 -name）
so, _ = run(r"""find /srv/dataset/Images/visible -maxdepth 1 -type f | wc -l""")
print("远端文件总数:", so.strip())

print("== 解包日志 ==")
so, _ = run("tail -n 5 /root/data_wait.log 2>/dev/null")
print(so.strip() or "(无)")
cli.close()
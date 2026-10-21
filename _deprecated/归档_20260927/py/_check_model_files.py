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
print("== fix_model.log tail 30 ==")
so, _ = run("tail -n 30 /root/fix_model.log 2>/dev/null")
print(so)
print("== 模型目录全列 ==")
so, _ = run("ls -la /srv/models/LocateAnything-3B/ 2>/dev/null")
print(so)
print("== 有无 temp/partial ==")
so, _ = run("find /srv/models -name '*.safetensors*' -o -name '*.tmp*' 2>/dev/null | head")
print(so.strip() or "(无)")
cli.close()
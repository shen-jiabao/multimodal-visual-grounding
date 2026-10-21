import os
import paramiko


def run(cmd):
    _, out, err = cli.exec_command(cmd, timeout=120)
    return out.read().decode("utf-8", "replace"), \
        err.read().decode("utf-8", "replace")


cli = paramiko.SSHClient()
cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
cli.connect(os.environ["SSH_HOST"], port=int(os.environ["SSH_PORT"]),
            username=os.environ["SSH_USER"],
            password=os.environ["SSH_PW"], timeout=25,
            banner_timeout=30, auth_timeout=30)

for f in ["/tmp/pip_err.log", "/tmp/venv_err.log"]:
    print(f"===== {f} =====")
    so, _ = run(f"tail -20 {f} 2>/dev/null")
    print(so.strip() or "(无)")

print("===== pip config / index =====")
so, _ = run("/srv/venv-locate/bin/pip config list 2>/dev/null; "
            "cat /root/.pip/pip.conf 2>/dev/null; "
            "cat ~/.config/pip/pip.conf 2>/dev/null")
print(so.strip() or "(无)")

print("===== venv pip 指向 =====")
so, _ = run("head -1 /srv/venv-locate/bin/pip; "
            "/srv/venv-locate/bin/python -m pip --version 2>&1 | head -2")
print(so.strip() or "(无)")
cli.close()
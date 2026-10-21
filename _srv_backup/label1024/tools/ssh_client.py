import os
import paramiko


def get_client():
    host = os.environ.get("SSH_HOST")
    port = int(os.environ.get("SSH_PORT", "22"))
    user = os.environ.get("SSH_USER") or "root"
    password = os.environ.get("SSH_PASSWORD")
    if not host or not password:
        raise SystemExit(
            "缺少连接信息：请设置环境变量 SSH_HOST / SSH_PASSWORD（可选 SSH_PORT / SSH_USER）"
        )
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(host, port=port, username=user, password=password, timeout=30)
    return client


def run_cmd(client, command, timeout=120):
    stdin, stdout, stderr = client.exec_command(command, timeout=timeout)
    out = stdout.read().decode("utf-8", errors="replace")
    err = stderr.read().decode("utf-8", errors="replace")
    code = stdout.channel.recv_exit_status()
    return code, out, err


def main():
    client = get_client()
    commands = [
        "hostname && uname -a && nvidia-smi",
        "df -h / /data 2>/dev/null",
        "ls /data 2>/dev/null; ls /data/miniconda/envs 2>/dev/null; ls /root 2>/dev/null",
        "which python3; python3 --version",
    ]
    for cmd in commands:
        code, out, err = run_cmd(client, cmd, timeout=60)
        print(f"=== {cmd} (exit {code}) ===")
        print(out)
        if err:
            print("STDERR:", err[:500])
    client.close()


if __name__ == "__main__":
    main()

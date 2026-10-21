# -*- coding: utf-8 -*-
"""替换为健壮版等待器并重启（先杀旧进程）。"""
import os
import paramiko


def main() -> int:
    cli = paramiko.SSHClient()
    cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    cli.connect(os.environ["SSH_HOST"], port=int(os.environ["SSH_PORT"]),
                username=os.environ["SSH_USER"],
                password=os.environ["SSH_PW"], timeout=25,
                banner_timeout=30, auth_timeout=30)
    # 杀旧等待器（不动下载进程）
    _, out, _ = cli.exec_command(
        "pkill -f _wait_probe_qwen.sh; sleep 1; "
        "pgrep -af _wait_probe_qwen.sh | grep -v pgrep | wc -l", timeout=30)
    print("旧等待器残留:", out.read().decode().strip())

    sftp = cli.open_sftp()
    sftp.put(r"d:\Desktop\2026AiC\_wait_probe_qwen.sh", "/root/_wait_probe_qwen.sh")
    sftp.close()
    _, out, err = cli.exec_command(
        "chmod +x /root/_wait_probe_qwen.sh && (setsid nohup bash "
        "/root/_wait_probe_qwen.sh > /root/qwen_probe_run.log 2>&1 < /dev/null &) "
        "; sleep 8; echo ==NEW_LOG==; tail -n 5 /root/qwen_probe_run.log; "
        "echo ==PROC==; pgrep -af _wait_probe_qwen.sh | grep -v pgrep | head -2",
        timeout=60)
    print(out.read().decode("utf-8", "replace"))
    cli.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
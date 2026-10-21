# -*- coding: utf-8 -*-
"""上传并后台启动 Qwen3-VL 探针等待器。"""
import os
import paramiko


def main() -> int:
    cli = paramiko.SSHClient()
    cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    cli.connect(os.environ["SSH_HOST"], port=int(os.environ["SSH_PORT"]),
                username=os.environ["SSH_USER"],
                password=os.environ["SSH_PW"], timeout=25,
                banner_timeout=30, auth_timeout=30)
    sftp = cli.open_sftp()
    sftp.put(r"d:\Desktop\2026AiC\_wait_probe_qwen.sh", "/root/_wait_probe_qwen.sh")
    sftp.close()
    _, out, err = cli.exec_command(
        "chmod +x /root/_wait_probe_qwen.sh && (setsid nohup bash "
        "/root/_wait_probe_qwen.sh > /root/qwen_probe_run.log 2>&1 < /dev/null &) "
        "; sleep 2; pgrep -af wait_probe_qwen | head -2", timeout=60)
    print(out.read().decode("utf-8", "replace"))
    cli.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
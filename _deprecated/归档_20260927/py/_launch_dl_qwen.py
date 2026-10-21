# -*- coding: utf-8 -*-
"""上传并后台启动 Qwen3-VL 下载。"""
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
    sftp.put(r"d:\Desktop\2026AiC\_dl_qwen3vl.sh", "/root/_dl_qwen3vl.sh")
    sftp.close()
    _, out, err = cli.exec_command(
        "chmod +x /root/_dl_qwen3vl.sh && (setsid nohup bash /root/_dl_qwen3vl.sh "
        "> /root/dl_qwen.log 2>&1 < /dev/null &) ; sleep 3; echo STARTED; "
        "tail -n 5 /root/dl_qwen.log 2>/dev/null", timeout=60)
    print(out.read().decode("utf-8", "replace"))
    cli.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
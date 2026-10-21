# -*- coding: utf-8 -*-
"""上传 Qwen3-VL 探针脚本 + 查下载进度。"""
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
    sftp.put(r"d:\Desktop\2026AiC\_probe_qwen3vl.py", "/root/_probe_qwen3vl.py")
    sftp.close()
    print("探针脚本已上传")

    cmd = r'''
echo ==MODEL_SIZE==
du -sh /srv/models/Qwen3-VL-8B-Instruct 2>/dev/null || echo "(下载中/未完成)"
echo ==SHARDS==
ls -la /srv/models/Qwen3-VL-8B-Instruct/*.safetensors* 2>/dev/null | awk '{print $5, $9}'
echo ==DL_LOG==
tail -n 3 /root/dl_qwen.log 2>/dev/null
echo ==PROBE_DATA==
ls /srv/dataset_probe/ 2>/dev/null
echo ==DISK==
df -h / | tail -1
'''
    _, out, err = cli.exec_command(cmd, timeout=90)
    print(out.read().decode("utf-8", "replace"))
    cli.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
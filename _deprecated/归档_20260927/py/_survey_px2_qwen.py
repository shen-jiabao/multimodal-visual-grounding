# -*- coding: utf-8 -*-
"""盘点 px2 新会话环境（Qwen3-VL 准备）。"""
import os
import paramiko

CMD = r'''
hostname
echo ==GPU==
nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader 2>/dev/null
echo ==DISK==
df -h / /srv 2>/dev/null | tail -3
echo ==SRV==
ls /srv 2>/dev/null
echo ==MODELS==
du -sh /srv/models/* 2>/dev/null
echo ==VENV==
ls -la /srv/venv-locate/bin/python 2>/dev/null || echo NO_VENV
echo ==DATA==
ls /srv/dataset 2>/dev/null || echo NO_DATA
echo ==DATAPROBE==
ls /srv/dataset_probe 2>/dev/null || echo NO_PROBE
echo ==PROC==
pgrep -af python | head -5
echo ==CACHE==
ls ~/.cache/modelscope/hub 2>/dev/null | head
ls ~/.cache/huggingface/hub 2>/dev/null | head
echo ==NVIDIA_PY==
python3 --version
'''


def main() -> int:
    cli = paramiko.SSHClient()
    cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    cli.connect(os.environ["SSH_HOST"], port=int(os.environ["SSH_PORT"]),
                username=os.environ["SSH_USER"],
                password=os.environ["SSH_PW"], timeout=25,
                banner_timeout=30, auth_timeout=30)
    _, out, err = cli.exec_command(CMD, timeout=120)
    print(out.read().decode("utf-8", "replace"))
    e = err.read().decode("utf-8", "replace")
    if e.strip():
        print("--- stderr ---")
        print(e)
    cli.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
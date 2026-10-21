# -*- coding: utf-8 -*-
"""查 Qwen3-VL 下载进度 + 探针运行状态。"""
import os
import paramiko

CMD = r'''
echo ==MODEL_SIZE==
du -sh /srv/models/Qwen3-VL-8B-Instruct 2>/dev/null
echo ==SHARDS==
for n in 1 2 3 4; do
  stat -c "%n %s" /srv/models/Qwen3-VL-8B-Instruct/model-0000${n}-of-00004.safetensors 2>/dev/null
done
ls /srv/models/Qwen3-VL-8B-Instruct/ 2>/dev/null | grep -c incomplete
echo ==WAITER_LOG==
tail -n 8 /root/qwen_probe_run.log 2>/dev/null
echo ==PROBE_STDOUT==
tail -n 15 /srv/output/qwen3vl_probe_stdout.log 2>/dev/null || echo "(探针未开始)"
echo ==PROBE_JSON==
ls -la /srv/output/qwen3vl_probe.json 2>/dev/null || echo "(未生成)"
echo ==PROC==
pgrep -af "wait_probe_qwen|_probe_qwen3vl|snapshot_download" | grep -v pgrep | head -5
echo ==GPU==
nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader
echo ==DISK==
df -h / | tail -1
'''


def main() -> int:
    cli = paramiko.SSHClient()
    cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    cli.connect(os.environ["SSH_HOST"], port=int(os.environ["SSH_PORT"]),
                username=os.environ["SSH_USER"],
                password=os.environ["SSH_PW"], timeout=25,
                banner_timeout=30, auth_timeout=30)
    _, out, err = cli.exec_command(CMD, timeout=90)
    print(out.read().decode("utf-8", "replace"))
    cli.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
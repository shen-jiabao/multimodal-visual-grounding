# -*- coding: utf-8 -*-
"""修补分片跳过后重启 Qwen3-VL 全量管线。

背景：原 `todo_imgs` 只按 direct 是否齐备跳过图片，导致旧脚本已跑完的
18 张图整体跳过、其序数问句永不进入 SoM 阶段。已修补为 direct/som 双条件。

本脚本流程：
1. 备份远端两份 JSON 与旧日志（防 kill 时写入中断丢数据）
2. 统计 SoM 缺口（前缀图中属序数但无 SoM 记录的 qid）
3. 杀旧进程 → 校验残留
4. 上传修补版管线 → 重启
5. 打印新日志头
"""
import os

import paramiko

PIPE = r"d:\Desktop\2026AiC\_qwen_full_pipeline.py"

PRE = r'''
set -e
cd /srv/output
cp -f qwen3vl_all.json qwen3vl_all.bak.json
cp -f qwen3vl_som_all.json qwen3vl_som_all.bak.json
cp -f qwen_full.log qwen_full.prev.log
echo "备份完成"
/srv/venv-locate/bin/python - <<'PY'
import json
d = json.load(open('/srv/output/qwen3vl_all.json'))
s = json.load(open('/srv/output/qwen3vl_som_all.json'))
cores = json.load(open('/srv/output/_ordinal_cores_v3.json'))
prefix = [k for k, v in d.items() if 'n' not in v]
gap = [k for k in prefix if k in cores and k not in s]
print('已跑(旧脚本前缀) =', len(prefix))
print('其中序数问句     =', sum(1 for k in prefix if k in cores))
print('SoM 缺口         =', len(gap))
print('缺口样例         =', sorted(gap)[:8])
PY
pkill -f _qwen_full_pipeline || true
sleep 2
echo -n "残留进程数: "
pgrep -f _qwen_full_pipeline | wc -l
'''

POST = (
    "cd /root && (setsid nohup /srv/venv-locate/bin/python -u "
    "_qwen_full_pipeline.py > /srv/output/qwen_full.log 2>&1 < /dev/null &) ; "
    "sleep 6; echo ==NEW_PROC==; pgrep -af _qwen_full_pipeline | head -2; "
    "echo ==NEW_LOG_TAIL==; tail -n 3 /srv/output/qwen_full.log"
)


def main() -> int:
    cli = paramiko.SSHClient()
    cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    cli.connect(os.environ["SSH_HOST"], port=int(os.environ["SSH_PORT"]),
                username=os.environ["SSH_USER"],
                password=os.environ["SSH_PW"], timeout=25,
                banner_timeout=30, auth_timeout=30)
    print("== 备份 + 缺口统计 ==")
    _, o, _ = cli.exec_command(PRE, timeout=120)
    print(o.read().decode("utf-8", "replace"))

    sftp = cli.open_sftp()
    sftp.put(PIPE, "/root/_qwen_full_pipeline.py")
    sftp.close()
    print("== 已上传修补版管线，重启 ==")
    _, o, _ = cli.exec_command(POST, timeout=90)
    print(o.read().decode("utf-8", "replace"))
    cli.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

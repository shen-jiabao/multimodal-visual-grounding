# -*- coding: utf-8 -*-
"""核心词清理 → 上传 → 启动 Qwen3-VL 全量管线。"""
import json
import os
import sys

import paramiko

R = r"d:\Desktop\2026AiC"
CORES = os.path.join(R, "_ordinal_cores_v3.json")
PIPE = os.path.join(R, "_qwen_full_pipeline.py")

# 判据转发：核心词清理的**唯一实现处**在 `_analyze_ordinal3.py`。
# 历史教训：本函数曾在此处**另有一份实现**，与 `_analyze_ordinal3.py` 漂移，
# 导致单独跑抽取器得到的核心词与线上实际使用的 38 条不同。禁止再复制实现。
sys.path.insert(0, R)
from _analyze_ordinal3 import cleanup  # noqa: E402


def main() -> int:
    cores = json.load(open(CORES, encoding="utf-8"))
    cleaned = {k: cleanup(v) for k, v in cores.items()}
    bad = [k for k, v in cleaned.items() if len(v) < 3]
    print(f"核心词 {len(cleaned)} 条；清理后空/过短 {len(bad)}")
    for k in bad[:6]:
        print("  ", k, repr(cleaned[k]))
    out = os.path.join(R, "_ordinal_cores_v3.json")
    json.dump(cleaned, open(out, "w", encoding="utf-8"), ensure_ascii=False,
              indent=1)
    print("样例:", [cleaned[k] for k in list(cleaned)[:10]])

    cli = paramiko.SSHClient()
    cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    cli.connect(os.environ["SSH_HOST"], port=int(os.environ["SSH_PORT"]),
                username=os.environ["SSH_USER"],
                password=os.environ["SSH_PW"], timeout=25,
                banner_timeout=30, auth_timeout=30)
    sftp = cli.open_sftp()
    sftp.put(out, "/srv/output/_ordinal_cores_v3.json")
    sftp.put(PIPE, "/root/_qwen_full_pipeline.py")
    sftp.close()
    print("已上传 cores + pipeline")

    cmd = ("mkdir -p /srv/output && cd /root && (setsid nohup "
           "/srv/venv-locate/bin/python -u _qwen_full_pipeline.py "
           "> /srv/output/qwen_full.log 2>&1 < /dev/null &) ; sleep 8; "
           "echo ==LOG==; tail -n 12 /srv/output/qwen_full.log; "
           "echo ==PROC==; pgrep -af _qwen_full_pipeline | grep -v pgrep | head -2")
    _, o, _ = cli.exec_command(cmd, timeout=90)
    print(o.read().decode("utf-8", "replace"))
    cli.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
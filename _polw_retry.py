# -*- coding: utf-8 -*-
"""轮询二次序数枚举批次的远端状态（无凭据落盘，密码走 SSH_PW 环境变量）。

输出：二次批进程、日志尾部、GPU 状态、二次/首轮 checkpoint 各图候选数。
用法（PowerShell）：
    $env:SSH_PW='<password>'; python -B _polw_retry.py
"""
import os
import sys

import paramiko


def _checkpoint_summary(path: str) -> str:
    """读取远端 checkpoint json 并汇总各字段候选数。"""
    code = (
        "import json,os\n"
        f'p="{path}"\n'
        "d=json.load(open(p)) if os.path.exists(p) else {}\n"
        "print({k:(v.get('n_raw_boxes'), v.get('error')) "
        "for k,v in d.items()})\n"
    )
    return code


def main() -> int:
    """连接远端执行批状态检查。"""
    host = os.environ.get("SSH_HOST", "px-cloud1.matpool.com")
    port = int(os.environ.get("SSH_PORT", "27675"))
    user = os.environ.get("SSH_USER", "root")
    pw = os.environ.get("SSH_PW")
    if not pw:
        print("未设置 SSH_PW", file=sys.stderr)
        return 2

    rty_ckpt = _checkpoint_summary("/root/output/locate_ordinal_retry.json")
    first_ckpt = _checkpoint_summary("/root/output/locate_ordinal_batch.json")
    # 用 base64 传递 python 代码，避免嵌套引号被 shell 破坏
    import base64
    b64 = lambda code: base64.b64encode(code.encode()).decode()  # noqa: E731

    cmd = (
        "echo ==RETRY_PROC==; pgrep -af _locate_ordinal_batch.py; "
        "echo ==RETRY_LOG_TAIL==; tail -n 40 /root/output/ordinal_retry.log "
        "2>/dev/null; "
        "echo ==RETRY_CHECKPOINT==; python3 -c "
        f"\"import base64;exec(base64.b64decode('{b64(rty_ckpt)}'))\"; "
        "echo ==FIRST_CHECKPOINT==; python3 -c "
        f"\"import base64;exec(base64.b64decode('{b64(first_ckpt)}'))\"; "
        "echo ==GPU==; nvidia-smi --query-gpu=memory.used,memory.free,"
        "utilization.gpu --format=csv,noheader"
    )
    cli = paramiko.SSHClient()
    cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    cli.connect(host, port=port, username=user, password=pw,
                timeout=25, banner_timeout=30, auth_timeout=30)
    _, out, err = cli.exec_command(cmd, timeout=60)
    print(out.read().decode("utf-8", "replace"))
    e = err.read().decode("utf-8", "replace")
    if e.strip():
        print("---stderr---\n" + e)
    cli.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
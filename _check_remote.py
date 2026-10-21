# -*- coding: utf-8 -*-
"""只读核查远端全量推理真实进度（避免 PowerShell 内联转义问题）。"""
import os
import json

import paramiko


def main() -> int:
    host = os.environ.get("SSH_HOST", "px-cloud1.matpool.com")
    port = int(os.environ.get("SSH_PORT", "27675"))
    user = os.environ.get("SSH_USER", "root")
    pw = os.environ.get("SSH_PW")
    if not pw:
        print("缺少 SSH_PW")
        return 2
    cli = paramiko.SSHClient()
    cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    cli.connect(host, port=port, username=user, password=pw,
                timeout=25, banner_timeout=30, auth_timeout=30)

    def run(command):
        _, out, err = cli.exec_command(command, timeout=60)
        return out.read().decode("utf-8", "replace"), err.read().decode("utf-8", "replace")

    code = (
        "import json,os;"
        "p='/root/infer/result_gd_3modal_full.json';"
        "d=json.load(open(p)) if os.path.exists(p) else {};"
        "print(len(d))"
    )
    so, _ = run(f'python3 -c "{code}"')
    n = int(so.strip() or 0)
    po, _ = run("pgrep -af predict_gd_3modal.py || true")
    tl, _ = run("tail -n 4 /root/infer/run_full_new.log 2>/dev/null || true")
    gp, _ = run("nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader")
    print(f"[remote full] {n}/9555")
    print("[proc]", po.strip() or "(none)")
    print("[tail]")
    for ln in tl.strip().splitlines():
        print("  ", ln)
    print("[gpu]", gp.strip())
    cli.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
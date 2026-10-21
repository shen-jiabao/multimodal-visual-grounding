# -*- coding: utf-8 -*-
"""后台监控远端全量 GD 推理，直到 checkpoint 完成（9555 条且进程结束）。

用法（PowerShell，后台运行）：
    $env:SSH_PW='<password>'; python -B _monitor_full.py
"""
import os
import time

import paramiko

HOST = os.environ.get("SSH_HOST", "px-cloud1.matpool.com")
PORT = int(os.environ.get("SSH_PORT", "27675"))
USER = os.environ.get("SSH_USER", "root")
PW = os.environ.get("SSH_PW")
OUT = "/root/infer/result_gd_3modal_full.json"
POLL_S = 120


def run_remote(cli, cmd: str) -> str:
    """执行远端命令并返回 stdout。"""
    _, out, err = cli.exec_command(cmd, timeout=60)
    o = out.read().decode("utf-8", "replace")
    e = err.read().decode("utf-8", "replace")
    return o, e


def poll_once(cli):
    """返回 (完成的条数, 错误条数, 进程是否仍在跑)。"""
    code = (
        f"import json,os;"
        f'p="{OUT}";'
        f"d=json.load(open(p)) if os.path.exists(p) else {{}};"
        f"print(len(d),len([k for k,v in d.items() if 'error' in v]))"
    )
    o, _ = run_remote(cli, f"python3 -c \"{code}\"")
    parts = o.strip().split()
    done = int(parts[0]) if parts and parts[0].isdigit() else 0
    nerr = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
    po, _ = run_remote(cli, "pgrep -f predict_gd_3modal.py")
    running = bool(po.strip())
    return done, nerr, running


def main() -> int:
    """循环血压直至完成并打印结果。"""
    if not PW:
        print("未设置 SSH_PW", file=os.sys.stderr)
        return 2
    cli = paramiko.SSHClient()
    cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    cli.connect(HOST, port=PORT, username=USER, password=PW,
                timeout=25, banner_timeout=30, auth_timeout=30)
    print("[monitor] 开始监控", flush=True)
    last = 0
    while True:
        try:
            done, nerr, running = poll_once(cli)
        except Exception as ex:  # noqa: BLE001
            print(f"[monitor] poll 异常: {ex}", flush=True)
            time.sleep(POLL_S)
            continue
        if done != last:
            print(f"[monitor] {done}/9555 条 err={nerr} running={running}",
                  flush=True)
            last = done
        if done >= 9555 or not running:
            print(f"[DONE] {done}/9555 条 err={nerr} running={running}",
                  flush=True)
            break
        time.sleep(POLL_S)
    cli.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
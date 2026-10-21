# -*- coding: utf-8 -*-
"""测量杀掉竞争进程后的真实吞吐（两次快照相隔 WINDOW 秒）。"""
import os
import re
import time

import paramiko

WINDOW = 200
CMD = ("tail -n 1 /srv/output/qwen_full.log; date +%s; "
       "/srv/venv-locate/bin/python -c \"import json;"
       "d=json.load(open('/srv/output/qwen3vl_all.json'));"
       "s=json.load(open('/srv/output/qwen3vl_som_all.json'));"
       "print(len(d),len(s))\"")


def snap(cli) -> tuple:
    _, o, _ = cli.exec_command(CMD, timeout=120)
    txt = o.read().decode("utf-8", "replace").strip().splitlines()
    line, epoch = txt[0], int(txt[1])
    nd, ns = (int(x) for x in txt[2].split())
    # 进度行有两种形态：逐图 "[86/1966] xxx.png" 与批检查点 "[A] 95/1966 图; ..."，
    # 前者带方括号、后者不带，故不锚定括号，直接取第一个 "数/数"。
    m = re.search(r"(\d+)/(\d+)", line)
    img = int(m.group(1)) if m else -1
    return {"line": line, "epoch": epoch, "direct": nd, "som": ns,
            "img": img}


def main() -> int:
    cli = paramiko.SSHClient()
    cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    cli.connect(os.environ["SSH_HOST"], port=int(os.environ["SSH_PORT"]),
                username=os.environ["SSH_USER"],
                password=os.environ["SSH_PW"], timeout=25,
                banner_timeout=30, auth_timeout=30)
    a = snap(cli)
    print(f"T0  {a['line']}")
    print(f"    direct={a['direct']} som={a['som']} 图={a['img']}")
    time.sleep(WINDOW)
    b = snap(cli)
    print(f"\nT1  {b['line']}")
    print(f"    direct={b['direct']} som={b['som']} 图={b['img']}")

    dt = b["epoch"] - a["epoch"]
    dd, di = b["direct"] - a["direct"], b["img"] - a["img"]
    print(f"\n窗口 {dt}s：direct +{dd}（{dd / dt:.2f} q/s）、图 +{di}"
          f"（{di / dt * 3600:.0f} 图/h）")
    if di > 0:
        remain_img = 1966 - b["img"]
        print(f"按图外推剩余 {remain_img} 图 → {remain_img / (di / dt) / 3600:.1f}h")
    for label, rem in (("direct", 9555 - b["direct"]),
                       ("som选择", 2014 - b["som"])):
        if dd > 0 and label == "direct":
            print(f"按 direct 速率外推剩余 {rem} 条 → "
                  f"{rem / (dd / dt) / 3600:.1f}h（仅 direct 部分）")
    cli.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

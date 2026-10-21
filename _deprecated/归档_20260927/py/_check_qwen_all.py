# -*- coding: utf-8 -*-
"""核实 qwen3vl_all.json 的 100 条已有条目来源与格式一致性。"""
import os
import paramiko

CMD = r'''
echo ==FILES==
ls -la /srv/output/qwen3vl_all.json /srv/output/qwen3vl_som_all.json 2>/dev/null
echo ==COUNT==
/srv/venv-locate/bin/python -c "
import json
d=json.load(open('/srv/output/qwen3vl_all.json'))
print('entries', len(d))
ks=list(d)[:3]
for k in ks: print(k, d[k])
"
echo ==LOG==
tail -n 6 /srv/output/qwen_full.log
'''


def main() -> int:
    cli = paramiko.SSHClient()
    cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    cli.connect(os.environ["SSH_HOST"], port=int(os.environ["SSH_PORT"]),
                username=os.environ["SSH_USER"],
                password=os.environ["SSH_PW"], timeout=25,
                banner_timeout=30, auth_timeout=30)
    _, o, _ = cli.exec_command(CMD, timeout=90)
    print(o.read().decode("utf-8", "replace"))
    cli.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
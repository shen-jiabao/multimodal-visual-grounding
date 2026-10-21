import os
import json
import paramiko
import hashlib


def run(cmd, t=120):
    _, out, err = cli.exec_command(cmd, timeout=t)
    return out.read().decode("utf-8", "replace"), \
        err.read().decode("utf-8", "replace")


cli = paramiko.SSHClient()
cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
cli.connect(os.environ["SSH_HOST"], port=int(os.environ["SSH_PORT"]),
            username=os.environ["SSH_USER"],
            password=os.environ["SSH_PW"], timeout=25,
            banner_timeout=30, auth_timeout=30)

# 取样 5 个 ERR id 比对远端 vs 本地文件 md5
sample_ids = ["000380", "000968", "001192", "002123", "003901"]
for iid in sample_ids:
    remote_md5, _ = run(f"md5sum /srv/dataset/Images/visible/{iid}.png 2>/dev/null")
    local_p = rf"d:\Desktop\2026AiC\初赛数据集-基于大模型的多模态视觉理解与推理\Images\visible\{iid}.png"
    h = hashlib.md5()
    if os.path.exists(local_p):
        h.update(open(local_p, "rb").read())
        print(f"{iid}.png  远端[{remote_md5.split()[0] if remote_md5.split() else 'MISSING'}] "
              f"本地[{h.hexdigest()}]  {'OK' if remote_md5.startswith(h.hexdigest()) else 'DIFF!'}")
    else:
        print(f"{iid}.png  远端[{remote_md5.split()[0] if remote_md5.split() else 'MISSING'}] 本地[文件不存在]")

# 远端 png 大小异常检查（62 id 的所有 png 大小）
print("\n== 62 id 远端文件大小(取样) ==")
so, _ = run("""for i in 000380 000674 000968 001192 001416 001640 001871; do
  stat -c "%n %s" /srv/dataset/Images/visible/$i.png 2>/dev/null
done""")
print(so.strip())
cli.close()
import os
import json
import paramiko


def run(cmd, t=60):
    _, out, err = cli.exec_command(cmd, timeout=t)
    return out.read().decode("utf-8", "replace"), \
        err.read().decode("utf-8", "replace")


cli = paramiko.SSHClient()
cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
cli.connect(os.environ["SSH_HOST"], port=int(os.environ["SSH_PORT"]),
            username=os.environ["SSH_USER"],
            password=os.environ["SSH_PW"], timeout=25,
            banner_timeout=30, auth_timeout=30)

# 提取 62 个 ERR id（本地从远端日志拿）
so, _ = run(r"""grep '\[ERR\]' /srv/output/locate_full_stdout.log | sed -E 's/.* ([0-9]{6})_001.*/\1/' | sort -u""")
err_ids = [x.strip() for x in so.splitlines() if x.strip()]
print("ERR 唯一 id 数:", len(err_ids))
print("前15:", err_ids[:15])

# 远端检查这些文件是否存在
from collections import Counter
png_ok, jpg_ok, missing = [], [], []
for iid in err_ids:
    for ext in ("png", "jpg"):
        hit, _ = run(f"test -f /srv/dataset/Images/visible/{iid}.{ext} && echo yes || echo no")
        if hit.strip() == "yes":
            (png_ok if ext == "png" else jpg_ok).append(iid)
            break
    else:
        missing.append(iid)
print(f"\n远端检查 62 id:")
print(f"  存在为 .png 的: {len(png_ok)}")
print(f"  存在为 .jpg 的: {len(jpg_ok)}")
print(f"  两者都不存在: {len(missing)}")
if missing:
    print("  缺失:", missing[:10])
cli.close()
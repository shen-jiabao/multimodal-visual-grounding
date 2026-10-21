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

print("== 进度行 ==")
so, _ = run("grep -E '^\\[' /srv/output/locate_full_stdout.log | tail -n 5")
print(so.strip() or "(无)")

print("== 完成标记 ==")
so, _ = run("grep -E '全部完成|完成|总共' /srv/output/locate_full_stdout.log | tail -n 3")
print(so.strip() or "(未完成)")

print("== ERR 计数 ==")
so, _ = run("grep -c '\\[ERR\\]' /srv/output/locate_full_stdout.log")
print("ERR 总数:", so.strip())
so, _ = run("grep '\\[ERR\\]' /srv/output/locate_full_stdout.log | tail -n 8")
print(so.strip() or "(无 ERR)")

print("== checkpoint 条目数 ==")
sftp = cli.open_sftp()
try:
    sftp.get("/srv/output/locate_bbox_all.json", r"d:\Desktop\2026AiC\_prog_ckpt.json")
    d = json.load(open(r"d:\Desktop\2026AiC\_prog_ckpt.json", encoding="utf-8"))
    print("entries =", len(d))
    import os as _os
    _os.remove(r"d:\Desktop\2026AiC\_prog_ckpt.json")
except Exception as ex:
    print("ckpt read ERR:", repr(ex))
sftp.close()

print("== 进程/GPU ==")
so, _ = run("pgrep -f _locate_full.py | wc -l; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader")
print(so.strip())
cli.close()
import os
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
print("== 模型分片大小 ==")
so, _ = run("stat -c %n=%s /srv/models/LocateAnything-3B/model-0000*.safetensors 2>/dev/null")
print(so.strip() or "(无)")
print("== 模型总量 ==")
so, _ = run("du -sh /srv/models/LocateAnything-3B 2>/dev/null")
print(so.strip())
print("== 模型下载进程 ==")
so, _ = run("pgrep -af 'snapshot_download|modelscope' | head -3")
print(so.strip() or "(无)")
print("== 探针等待进程 ==")
so, _ = run("pgrep -af probe_wait | head -3")
print(so.strip() or "(无)")
print("== probe_wait.log ==")
so, _ = run("cat /root/probe_wait.log 2>/dev/null")
print(so.strip() or "(无)")
print("== probe_stdout.log ==")
so, _ = run("tail -n 30 /srv/output/probe_stdout.log 2>/dev/null")
print(so.strip() or "(无)")
print("== probe_result.json 是否存在 ==")
so, _ = run("ls -la /srv/output/probe_result.json 2>/dev/null")
print(so.strip() or "(未生成)")
print("== 数据上传进度 ==")
so, _ = run("ls -la /srv/dataset_rgb.tar 2>/dev/null")
print(so.strip() or "(无)")
cli.close()
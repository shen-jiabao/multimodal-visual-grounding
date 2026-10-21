import os
import json
import paramiko


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

# 拉回完整全量结果
sftp = cli.open_sftp()
sftp.get("/srv/output/locate_bbox_all.json", r"d:\Desktop\2026AiC\locate_bbox_all.json")
sftp.close()
print("已拉回 locate_bbox_all.json")

# 统计分析
d = json.load(open(r"d:\Desktop\2026AiC\locate_bbox_all.json", encoding="utf-8"))
print("总条目:", len(d))
noframe = [k for k, v in d.items() if v.get("bbox") is None]
print("无框:", len(noframe))
# 62 ERR 的 qid 形态
err_qids = [k for k in noframe if "-" not in k]
from collections import Counter
suffix = Counter(k.split("_")[-1] for k in noframe)
print("无框的后缀分布:", dict(suffix.most_common(6)))
# 无框里随机看10条(非ERR)
with open(r"d:\Desktop\2026AiC\初赛数据集-基于大模型的多模态视觉理解与推理\queries\queries.json", encoding="utf-8") as f:
    q = json.load(f)
print("\n无框样本(前12):")
for k in noframe[:12]:
    print("  ", k, "|", q[k]["query"][:60])
cli.close()
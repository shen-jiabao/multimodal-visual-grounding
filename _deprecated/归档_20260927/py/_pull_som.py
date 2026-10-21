# -*- coding: utf-8 -*-
"""拉回 SoM 结果 + 标注图，与提交对比量化。"""
import json
import os

import paramiko

R = r"d:\Desktop\2026AiC"


def iou(a, b):
    if not a or not b:
        return 0.0
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0., x2-x1), max(0., y2-y1)
    it = iw*ih
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - it
    return it/ua if ua > 1e-9 else 0.0


def area(b):
    return (b[2]-b[0])*(b[3]-b[1])


cli = paramiko.SSHClient()
cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
cli.connect(os.environ["SSH_HOST"], port=int(os.environ["SSH_PORT"]),
            username=os.environ["SSH_USER"], password=os.environ["SSH_PW"],
            timeout=25, banner_timeout=30, auth_timeout=30)
sftp = cli.open_sftp()
sftp.get("/srv/output/qwen3vl_som.json", os.path.join(R, "qwen3vl_som.json"))
sftp.get("/srv/output/som_000023.png", os.path.join(R, "som_000023.png"))
sftp.close()
cli.close()
print("已拉回 qwen3vl_som.json + som_000023.png\n")

som = json.load(open(os.path.join(R, "qwen3vl_som.json"), encoding="utf-8"))
merge = json.load(open(os.path.join(R, "result_locate_merge.json"),
                       encoding="utf-8"))
print("== 候选池（Qwen 枚举） ==")
for i, b in enumerate(som["000023"]["candidates"], 1):
    print(f"  #{i} {[round(v,3) for v in b]}  area={area(b):.4f}")

print("\n== SoM 选择 vs 当前提交 ==")
for qid, r in som["000023"]["queries"].items():
    sb = merge[qid]["bbox"]
    ob = r["bbox"]
    print(f"{qid}  选编号={r['answer_num']}  耗时={r['time_s']}s")
    print(f"   SoM  ={None if not ob else [round(v,3) for v in ob]} "
          f"area={area(ob) if ob else 0:.4f}")
    print(f"   提交 ={[round(v,3) for v in sb]} area={area(sb):.4f}")
    print(f"   IoU={iou(ob, sb):.3f}")
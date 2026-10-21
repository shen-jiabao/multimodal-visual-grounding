#!/bin/bash
# 验证红外补轮脚本可正常导入与被调用（不加载模型、不推理）。
# 目的：在 8 小时等待之前，把语法/导入/清单问题全部暴露掉。
VPY=/srv/venv-la2/bin/python
cd /root
$VPY - <<'PY'
import ast, json, sys

src = open("/root/_retry_ir.py", encoding="utf-8").read()
ast.parse(src)
print("1) 语法 OK")

sys.path.insert(0, "/root")
import _locate_full as L          # noqa: E402
print("2) _locate_full 可导入 | MAX_IMAGE_SIZE =", L.MAX_IMAGE_SIZE)
print("   load_model 可调用:", callable(L.load_model),
      "| infer_qid 可调用:", callable(L.infer_qid))

qids = json.load(open("/srv/output/nobox_qids.json", encoding="utf-8"))
print("3) 无框清单", len(qids), "条 | 前 3:", qids[:3])

q = json.load(open("/srv/dataset/queries/queries.json", encoding="utf-8"))
import os
ir = sorted({q[k]["infrared"] for k in qids})
miss = [p for p in ir if not os.path.exists(os.path.join("/srv/dataset", p))]
print(f"4) 所需红外图 {len(ir)} 张 | 缺失 {len(miss)}")
print("   样例路径:", q[qids[0]]["infrared"], "存在:",
      os.path.exists(os.path.join("/srv/dataset", q[qids[0]]["infrared"])))

from PIL import Image
im = Image.open(os.path.join("/srv/dataset", q[qids[0]]["infrared"]))
print("5) 红外图可读:", im.size, im.mode)
PY

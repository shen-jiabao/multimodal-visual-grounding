#!/bin/bash
# 把**运行中**的 checkpoint 转成一份自洽快照（原子落盘），供本地装配器读取。
#
# 为什么需要它：管线用 `json.dump(direct, open(path,"w"))` **就地重写**，非原子。
# 直接 `scp/sftp get` 会拿到"写一半"的内容——实测连续 6 次下载全部解析失败
# （报 Extra data），不是偶发。这里在远端用 raw_decode 只取**首个完整 JSON 值**，
# 再写 .tmp 后 `os.replace` 成快照；这样下载到的一定是完整文件。
#
# 若整体可解析则直接用（说明运气好，抓到了写间隙）；否则取首个完整值并报告尾随字节。
set -u
PY=/root/rivermind-data/venv/bin/python

$PY - <<'PYEOF'
import json
import os
import time

SRC = "/root/rivermind-data/output/qwen32b_direct.json"
DST = "/root/rivermind-data/output/_snap_health.json"

d = None
note = ""
for attempt in range(6):
    txt = open(SRC, encoding="utf-8").read()
    try:
        d = json.loads(txt)
        note = "整体可解析"
        break
    except Exception as ex:
        try:
            d, end = json.JSONDecoder().raw_decode(txt)
            tail = txt[end:]
            note = ("就地重写中，已取首个完整 JSON 值（%d 字节）；尾随 %d 字节"
                    "（非空白 %d）: %r"
                    % (end, len(tail),
                       sum(1 for c in tail if not c.isspace()),
                       tail[:40]))
            break
        except Exception:
            note = "第 %d 次读取无完整值（%s），重试" % (attempt + 1,
                                                type(ex).__name__)
            print(note, flush=True)
            time.sleep(1.0)

if d is None:
    raise SystemExit("连续读取均未取得完整 JSON 值")

print("源 %d 字节 | %s | 条目 %d" % (len(txt), note, len(d)))
nb = sum(1 for v in d.values() if v.get("bbox"))
print("其中 有框 %d ｜ 输出[] %d" % (nb, len(d) - nb))

tmp = DST + ".tmp"
json.dump(d, open(tmp, "w", encoding="utf-8"), ensure_ascii=False)
os.replace(tmp, DST)
print("快照已原子落盘 %s（%d 字节）" % (DST, os.path.getsize(DST)))
PYEOF
# -*- coding: utf-8 -*-
"""重做附图 6：只取关键节点（便于投屏阅读），并把完整 36 条另存为文本证据。"""
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

EV = Path(r"d:\Desktop\2026AiC\答辩材料\AIC-2026-14375345-熬夜炼丹师-复赛-沈嘉宝"
          r"\AIC-2026-14375345-代码与数据\远端现场证据\服务器取证")
OUT = EV / "附图"
FONT = r"C:\Windows\Fonts\simsun.ttc"
BG, FG, DIM, ACC, WARN = (11, 32, 39), (214, 236, 240), (122, 150, 158), (0, 168, 150), (242, 169, 59)

MEM = Path(r"c:\Users\86189\.trae-cn\memory\projects\-d-Desktop-2026AiC--p2-a81c289e5a57af7b7d50")
pat = re.compile(r"\[session_id:\s*(\w+)\s*\|\s*topic_summary_time:\s*([0-9\-: ]+)\]\s*(.*)")


def clean(t: str) -> str:
    t = re.sub(r"([Pp]assword)\s+\S+", r"\1 ***", t)
    t = re.sub(r"root@\S+|:\d{4,5}\b", "***", t)
    t = re.sub(r"9AsgCRIGspNd|VvH1VXW94BFK", "***", t)
    return t.strip()


rows = []
for f in sorted(MEM.rglob("topics.md")):
    for ln in f.read_text("utf-8", "replace").splitlines():
        m = pat.search(ln)
        if m:
            rows.append((m.group(2).strip(), clean(m.group(3))))
rows.sort()

# 完整 36 条 → 文本证据
lines = ["# 本队开发会话时间线（AI 协作开发留痕）",
         "# 来源：本机开发会话记录 topic_summary_time（+0800）；已脱敏（移除凭据 / 主机串）",
         f"# 共 {len(rows)} 条，覆盖 {len({r[0][:10] for r in rows})} 个日期",
         ""]
for ts, txt in rows:
    lines.append(f"{ts}  {txt}")
(EV / "开发会话时间线_完整36条.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("已写出 开发会话时间线_完整36条.txt")

# 关键节点（人工挑选 15 条，逐字取自上述记录）
KEY = ["2026-08-30 21:42", "2026-08-31 15:11", "2026-09-08 19:44", "2026-09-11 22:01",
       "2026-09-12 00:18", "2026-09-15 21:24", "2026-09-17 15:39", "2026-09-27 16:33",
       "2026-09-27 18:20", "2026-09-28 12:03", "2026-10-03 12:41", "2026-10-03 15:24",
       "2026-10-05 12:03", "2026-10-05 14:21", "2026-10-06 17:58"]
picked = [r for r in rows if r[0][:16] in KEY]

im = Image.new("RGB", (1600, 62 + 24 * len(picked) + 58), BG)
d = ImageDraw.Draw(im)
f = ImageFont.truetype(FONT, 16)
ft = ImageFont.truetype(FONT, 25)
fn = ImageFont.truetype(FONT, 14)
d.rectangle([0, 0, 1600, 44], fill=(7, 22, 27)); d.rectangle([0, 0, 5, 44], fill=ACC)
d.text((26, 12), "附图 6 ｜ 开发会话时间戳（关键节点 15 / 完整 36 条随包）", font=ft, fill=(255, 255, 255))
y = 62
for ts, txt in picked:
    hot = "2026-10-03 15:24" in ts
    d.text((26, y), ts, font=f, fill=WARN if hot else ACC)
    d.text((186, y), txt[:92], font=f, fill=(255, 255, 255) if hot else FG)
    y += 24
y += 8
d.line([26, y, 1574, y], fill=(38, 70, 78), width=1)
d.text((26, y + 8), "来源：本机开发会话记录 topic_summary_time（+0800），已脱敏；完整 36 条见 开发会话时间线_完整36条.txt",
       font=fn, fill=DIM)
p = OUT / "e6_开发会话时间线_1600x472.png"
im.save(p)
print("生成", p.name, im.size)

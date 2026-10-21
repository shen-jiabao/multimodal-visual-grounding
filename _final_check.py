# -*- coding: utf-8 -*-
"""交付终检：包内文件、文档一致性、合规用词、PDF/Word 内容、zip 完整性。"""
import hashlib
import re
import zipfile
from pathlib import Path

from pypdf import PdfReader

ROOT = Path(r"d:\Desktop\2026AiC\复赛提交材料文件夹")
PKG = ROOT / "AIC-2026-14375345-熬夜炼丹师-复赛-沈嘉宝"
ZIP = ROOT / "AIC-2026-14375345-熬夜炼丹师-复赛-沈嘉宝.zip"
PDF = PKG / "AIC-2026-14375345-技术方案.PDF"
DOCX = PKG / "AIC-2026-14375345-技术方案.docx"
GEN = PKG / "AIC-2026-14375345-代码与数据/版本生成脚本"
EXPECT = ("51CD093556C43745FAA0586F4D3097FBE0EBF77E03914D0FF"
          "B7FA2A7065D4D6C")

files = [p for p in PKG.rglob("*") if p.is_file()]
total_mb = sum(p.stat().st_size for p in files) / 1048576
print(f"[1] 包内文件 {len(files)} 个，合计 {total_mb:.2f} MB")
print(f"[2] 版本生成脚本 {len(list(GEN.glob('*.py')))} 个")

for rel in ["AIC-2026-14375345-代码与数据/代码与数据说明.txt",
            "AIC-2026-14375345-代码与数据/投票池来源表.txt",
            "AIC-2026-14375345-代码与数据/版本生成脚本/README.txt"]:
    t = (PKG / rel).read_text(encoding="utf-8")
    print("    " + Path(rel).name,
          "数量声明", sorted(set(re.findall(r"（(\d+) 个）", t))))

BAD = ("人工|复核|手工|人眼|肉眼|亲判|校准|诱饵|归零法|预登记|挑选|筛选")
hits = []
for p in files:
    if p.suffix in (".py", ".txt", ".md", ".yaml"):
        try:
            t = p.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for m in re.finditer(BAD, t):
            hits.append(f"{p.relative_to(PKG)}:{t[:m.start()].count(chr(10)) + 1}")
print(f"[3] 敏感词命中 {len(hits)} 处 {hits[:5]}")

reader = PdfReader(str(PDF))
text = "".join(p.extract_text() or "" for p in reader.pages)
raw = reader.pages[0].get_contents().get_data()
print(f"[4] PDF {len(reader.pages)} 页；框图圆角曲线 {raw.count(b' c\n')} 段"
      f"（28 = 7 框 x 4 角）；含「提交件校验与结果」"
      f"{'提交件校验与结果' in text}；含得分 {'0.8116' in text}")

xml = None
if DOCX.exists():
    xml = zipfile.ZipFile(str(DOCX)).read("word/document.xml").decode(
        "utf-8", "ignore")
    flat = re.sub(r"<[^>]+>", "", xml)
    print(f"[5] Word 框图单格表 "
          f"{xml.count('w:val=' + chr(34) + 'TableGrid' + chr(34)) - 5} 个；"
          f"箭头 {xml.count('▼')} 个；含得分 {'0.8116' in flat}")
else:
    print("[5] 包内无 Word 版（已按要求移除）")

z = zipfile.ZipFile(str(ZIP))
names = z.namelist()
inner = [x for x in names if x.endswith("result.zip")]
bad = [x for x in names
       if re.search(r"_pick_rec|_consensus_v3|annotate_pose|"
                    r"train_competition|_weld_ordinal", x)]
sha = hashlib.sha256(z.read(inner[0])).hexdigest().upper() if inner else "-"
print(f"[6] zip {len(names)} 条目；反斜杠条目 "
      f"{sum(1 for x in names if chr(92) in x)}；zip 数 {len(inner)}；"
      f"违规脚本 {bad}")
print(f"    result.zip SHA256 {sha}")
print(f"    与期望一致 {sha == EXPECT}")

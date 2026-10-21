# -*- coding: utf-8 -*-
"""核对两个包里的技术方案 PDF 各自是哪条路线的版本。"""
from pathlib import Path

from pypdf import PdfReader

CASES = {
    "活动包(复赛提交材料文件夹/…-复赛-沈嘉宝)":
        Path(r"d:\Desktop\2026AiC\复赛提交材料文件夹\AIC-2026-14375345-熬夜炼丹师-复赛-沈嘉宝"
             r"\AIC-2026-14375345-技术方案.PDF"),
    "备份包(_备份_27B共识路线_20261014/…)":
        Path(r"d:\Desktop\2026AiC\复赛提交材料文件夹\_备份_27B共识路线_20261014"
             r"\AIC-2026-14375345-熬夜炼丹师-复赛-沈嘉宝\AIC-2026-14375345-技术方案.PDF"),
}
for name, p in CASES.items():
    try:
        r = PdfReader(str(p))
        t = "".join((pg.extract_text() or "") for pg in r.pages)
        marks = {k: (k in t) for k in
                 ("共识投票", "0.8116", "0.7963", "Qwen3.8-27B", "序数")}
        print(f"{name}\n    页数 {len(r.pages)}｜大小 {p.stat().st_size/1024:.0f} KB｜命中 {marks}")
    except Exception as exc:  # noqa: BLE001
        print(f"{name}\n    读取失败: {exc}")

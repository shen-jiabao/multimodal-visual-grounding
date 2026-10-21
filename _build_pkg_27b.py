# -*- coding: utf-8 -*-
"""组装「复赛提交材料文件夹」（以 result_semi_27b_annotate.zip 为准）。

结构按《复赛作品提交要求.pdf》：
    参赛团队编号-团队名称-复赛-队长姓名/        （根，占位符，拿到真实信息后整体改名）
    ├─ 参赛团队编号-技术方案.pdf
    ├─ 参赛团队编号-模型文件/                   （下载脚本 + 逐分片 SHA256 + 说明）
    ├─ 参赛团队编号-代码与数据/                 （code27b 全量代码 + 产物 + 日志 + 环境）
    └─ 参赛团队编号-赛题结果文档/               （提交 zip + SHA256.txt）

可重跑：每次先清空根目录再全量重建，输出逐项清单。
"""
from __future__ import annotations

import hashlib
import shutil
import zipfile
from pathlib import Path

R = Path(r"d:\Desktop\2026AiC")
PKG = R / "复赛提交材料文件夹"
TEAM = "参赛团队编号-团队名称-复赛-队长姓名"
P = "参赛团队编号"          # 文件名前缀（占位符）

SRC_ZIP = R / "submission_zips" / "semifinal" / "未提交" / "result_semi_27b_annotate.zip"
SRC_CODE = R / "_pkg27b_code_x"
SRC_PRED = R / "_pkg27b_predictions.json"
SRC_SHA = R / "_pkg27b_model_sha256.txt"
SRC_ENV = R / "_pkg27b_env.txt"
SRC_LOG = R / "_pkg27b_run.log"
SRC_DOC = R / "技术文档.md"
SRC_QUERIES = (R / "复赛数据集-基于大模型的多模态视觉理解与推理"
               / "queries" / "queries.json")


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest().upper()


def main() -> int:
    root = PKG / TEAM
    if root.exists():
        shutil.rmtree(root)
    (root / f"{P}-模型文件").mkdir(parents=True)
    (root / f"{P}-代码与数据").mkdir()
    (root / f"{P}-赛题结果文档").mkdir()

    # ── 1) 技术方案 PDF ──
    pdf = root / f"{P}-技术方案.pdf"
    import subprocess
    subprocess.run(["python", "-B", str(R / "_md2pdf.py"),
                    str(SRC_DOC), str(pdf)], check=True)

    # ── 2) 模型文件（下载说明路线，权重实体不入包）──
    md = root / f"{P}-模型文件"
    (md / "model_sha256.txt").write_text(
        SRC_SHA.read_text("utf-8").replace(
            "/root/autodl-tmp/models/Qwen3.8-27B/", ""), "utf-8")
    (md / "download_model.py").write_text('''# -*- coding: utf-8 -*-
"""下载模型权重（Qwen3.8-27B，ModelScope 通道，实测 16.7 MB/s）。

用法
    pip install modelscope
    python download_model.py            # 下载到 ./models/Qwen3.8-27B
    python download_model.py --verify   # 下载后逐分片比对 model_sha256.txt
"""
import argparse
import hashlib
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()
    out = Path("./models/Qwen3.8-27B")
    if not a.verify:
        from modelscope import snapshot_download
        snapshot_download("Qwen/Qwen3.8-27B", cache_dir="./models")
    want = {}
    for ln in Path("model_sha256.txt").read_text("utf-8").splitlines():
        h, name = ln.split()[0], ln.split()[-1].split("/")[-1]
        want[name] = h.lower()
    ok = True
    for name, h in sorted(want.items()):
        p = out / name
        if not p.exists():
            print(f"MISSING {name}")
            ok = False
            continue
        d = hashlib.sha256()
        with open(p, "rb") as f:
            for blk in iter(lambda: f.read(1 << 22), b""):
                d.update(blk)
        got = d.hexdigest()
        print(("OK   " if got == h else "FAIL ") + name)
        ok = ok and got == h
    print("ALL OK" if ok else "HAS FAILURE")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
''', "utf-8")
    (md / "模型文件说明.txt").write_text(
        "模型：Qwen3.8-27B-Instruct（ModelScope: Qwen/Qwen3.8-27B）\n"
        "参数量 27.78B，bfloat16，Safetensors 18 分片约 54GB。\n"
        "权重实体体积超出网盘限制，按赛方「模型文件」要求提供：\n"
        "  1) download_model.py —— 一键下载（snapshot_download）；\n"
        "  2) model_sha256.txt —— 实测推理所用 18 分片逐位 SHA256，\n"
        "     下载后 `python download_model.py --verify` 可验证与\n"
        "     跑分所用权重逐位一致。\n", "utf-8")

    # ── 3) 代码与数据 ──
    cd = root / f"{P}-代码与数据"
    shutil.copytree(SRC_CODE, cd / "code27b")
    d2 = cd / "data_outputs"
    d2.mkdir()
    shutil.copy2(SRC_PRED, d2 / "predictions_27b_annotate.json")
    shutil.copy2(SRC_LOG, d2 / "run_annotate.log")
    shutil.copy2(SRC_ENV, d2 / "env_record.txt")
    shutil.copy2(SRC_QUERIES, d2 / "queries.json")
    (cd / "代码与数据说明.txt").write_text(
        "code27b/        复赛推理代码（数据预处理/推理/后处理全链，\n"
        "                即技术方案 PDF 所述 src/ 与 configs/ 的完整实现）。\n"
        "data_outputs/   本次提交对应的实测产物：\n"
        "  predictions_27b_annotate.json  5690 条全量预测（2026-10-05 跑）\n"
        "  run_annotate.log               全量运行日志（02:01-07:23）\n"
        "  env_record.txt                 实测环境（GPU/驱动/torch/transformers）\n"
        "  queries.json                   赛题查询索引（用于复跑与校验）\n"
        "运行方式见 code27b/README.md 与技术方案 PDF §6。\n", "utf-8")

    # ── 4) 赛题结果文档 ──
    rd = root / f"{P}-赛题结果文档"
    z = rd / SRC_ZIP.name
    shutil.copy2(SRC_ZIP, z)
    with zipfile.ZipFile(z) as zf:
        names = zf.namelist()
    (rd / "SHA256.txt").write_text(
        f"{sha256(z)}  {z.name}\n"
        f"内部文件: {', '.join(names)}\n"
        "线上得分: 待提交后回填（排行榜按历史最高分记录）\n", "utf-8")

    # ── 5) 根 README（占用 PDF 未单列，但评委核对必需）──
    (root / "README.txt").write_text(
        "本文件夹 = 复赛作品材料（对应《复赛作品提交要求》四类材料）\n"
        "根目录名含占位符「参赛团队编号/团队名称/队长姓名」，\n"
        "拿到真实信息后请整体重命名根目录与各文件名前缀。\n"
        "\n"
        "四类材料对照：\n"
        f"  {P}-技术方案.pdf          技术方案（Qwen3.8-27B 推理，2026-10-04 版）\n"
        f"  {P}-模型文件/             权重下载脚本 + 18 分片实测 SHA256 + 说明\n"
        f"  {P}-代码与数据/           全量代码 code27b + 本次提交产物/日志/环境\n"
        f"  {P}-赛题结果文档/         result_semi_27b_annotate.zip（5690 条，\n"
        "                            校验器 4 项全过；SHA256 见内附清单）\n"
        "\n"
        "结果文件与代码对应关系：\n"
        "  zip 内 submission.json 由 data_outputs/predictions_27b_annotate.json\n"
        "  经 make_sub 逻辑（round4 + 回退框）装配，可逐条复算。\n", "utf-8")

    # ── 清单输出 ──
    print("=== 包结构 ===")
    for p in sorted(root.rglob("*")):
        rel = p.relative_to(root)
        tag = "d" if p.is_dir() else f"{p.stat().st_size/1024:.0f}KB"
        print(f"  {rel}  [{tag}]")
    print(f"\n根目录: {root}")
    print(f"zip SHA256: {sha256(z)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

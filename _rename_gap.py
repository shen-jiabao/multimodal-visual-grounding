"""将 _n32_leak_*.py 更名为 _n32_gap_*（"能力缺口"），并同步内部自引用。

leak 在本工程中指的是"序数词漏检 / 空间最高级漏检"这类能力缺口，
为避免与数据泄漏混淆，统一改用 gap。
"""
from __future__ import annotations

from pathlib import Path

DST = Path(
    r"d:\Desktop\2026AiC\复赛提交材料文件夹"
    r"\AIC-2026-14375345-熬夜炼丹师-复赛-沈嘉宝"
    r"\AIC-2026-14375345-代码与数据\版本生成脚本"
)

REPLACEMENTS = [
    ("_n32_leak_detail", "_n32_gap_detail"),
    ("_n32_spatial_leak", "_n32_spatial_gap"),
]

RENAMES = {
    "_n32_leak_detail.py": "_n32_gap_detail.py",
    "_n32_spatial_leak.py": "_n32_spatial_gap.py",
}


def main() -> None:
    """先替换文件内自引用，再重命名文件本身。"""
    for path in sorted(DST.glob("*.py")):
        text = path.read_text(encoding="utf-8")
        new_text = text
        for old, rep in REPLACEMENTS:
            new_text = new_text.replace(old, rep)
        if new_text != text:
            path.write_text(new_text, encoding="utf-8")
            print(f"内容更新: {path.name}")

    for old, new in RENAMES.items():
        src = DST / old
        if src.exists():
            src.rename(DST / new)
            print(f"改名: {old} -> {new}")


if __name__ == "__main__":
    main()

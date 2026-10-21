"""第五批用词优化：把"复核"统一为"校验"（仅改注释与提示文本）。"""
from __future__ import annotations

from pathlib import Path

DST = Path(
    r"d:\Desktop\2026AiC\复赛提交材料文件夹"
    r"\AIC-2026-14375345-熬夜炼丹师-复赛-沈嘉宝"
    r"\AIC-2026-14375345-代码与数据\版本生成脚本"
)


def main() -> None:
    """遍历目录下全部 .py，替换"复核"为"校验"。"""
    changed = 0
    for path in sorted(DST.glob("*.py")):
        text = path.read_text(encoding="utf-8")
        new_text = text.replace("复核", "校验")
        if new_text != text:
            path.write_text(new_text, encoding="utf-8")
            changed += 1
            print(f"{path.name}: 复核->校验 x{text.count('复核')}")
    print(f"\n完成：{changed} 个文件被修改")


if __name__ == "__main__":
    main()

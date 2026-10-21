"""第四批用词优化（仅改注释与文档字符串、提示文本）。

替换对照：
    人工目检        -> 线上评分
    便于人工核对排序 -> 便于核对排序
    人工看质量      -> 检查质量
    深度轴护栏      -> 深度轴约束
"""
from __future__ import annotations

from pathlib import Path

DST = Path(
    r"d:\Desktop\2026AiC\复赛提交材料文件夹"
    r"\AIC-2026-14375345-熬夜炼丹师-复赛-沈嘉宝"
    r"\AIC-2026-14375345-代码与数据\版本生成脚本"
)

RULES: list[tuple[str, str]] = [
    ("只能是线上分数或人工目检", "只能依据线上评分"),
    ("便于人工核对排序", "便于核对排序"),
    ("前 30，人工看质量", "前 30，供检查"),
    ("人工看质量", "检查质量"),
    ("深度轴护栏", "深度轴约束"),
]


def main() -> None:
    """遍历目录下全部 .py，执行第四批用词替换并打印命中统计。"""
    changed = 0
    for path in sorted(DST.glob("*.py")):
        text = path.read_text(encoding="utf-8")
        new_text = text
        hits: list[str] = []
        for old, rep in RULES:
            count = new_text.count(old)
            if count:
                hits.append(f"{old}->{rep} x{count}")
                new_text = new_text.replace(old, rep)
        if new_text != text:
            path.write_text(new_text, encoding="utf-8")
            changed += 1
            print(f"{path.name}: " + "; ".join(hits))
    print(f"\n完成：{changed} 个文件被修改")


if __name__ == "__main__":
    main()

"""批量优化版本生成脚本中的表述用词。

仅修改注释与文档字符串，不改动任何逻辑。
替换对照：
    必然错的角落小框   -> 角落处与目标不重叠的小框
    诱饵框             -> 对照框
    必然错             -> 与目标不重叠
    真实正确条数        -> 一致条数
    归零法             -> 分组测量法
"""
from __future__ import annotations

from pathlib import Path

DST = Path(
    r"d:\Desktop\2026AiC\复赛提交材料文件夹"
    r"\AIC-2026-14375345-熬夜炼丹师-复赛-沈嘉宝"
    r"\AIC-2026-14375345-代码与数据\版本生成脚本"
)

# 长词优先，避免被短词提前打断
RULES: list[tuple[str, str]] = [
    ("必然错的角落小框", "角落处与目标不重叠的小框"),
    ("诱饵框", "对照框"),
    ("必然错", "与目标不重叠"),
    ("真实正确条数", "一致条数"),
    ("归零法", "分组测量法"),
]


def main() -> None:
    """遍历目录下全部 .py，执行用词替换并打印命中统计。"""
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

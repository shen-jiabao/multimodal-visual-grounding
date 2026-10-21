"""第二批用词优化（仅改注释与文档字符串）。

替换对照：
    平台按历史最高分记录 ⇒ 本件不拉低成绩，而得分直接读出：  -> 由得分变化可估计：
    把线上得分当**测量仪器**用                                -> 由线上得分变化估计子集一致情况
    6σ 负面                                                   -> 显著负向
    预登记                                                    -> 预先设定
    归零                                                      -> 置空
"""
from __future__ import annotations

from pathlib import Path

DST = Path(
    r"d:\Desktop\2026AiC\复赛提交材料文件夹"
    r"\AIC-2026-14375345-熬夜炼丹师-复赛-沈嘉宝"
    r"\AIC-2026-14375345-代码与数据\版本生成脚本"
)

# 长句优先
RULES: list[tuple[str, str]] = [
    ("平台按历史最高分记录 ⇒ 本件不拉低成绩，而得分直接读出：", "由得分变化可估计："),
    ("把线上得分当**测量仪器**用", "由线上得分变化估计子集一致情况"),
    ("这正是两次 6σ 负面的根源", "这正是此前两次显著负向结果的根源"),
    ("6σ 负面", "显著负向"),
    ("预登记", "预先设定"),
    ("归零", "置空"),
]


def main() -> None:
    """遍历目录下全部 .py，执行第二批用词替换并打印命中统计。"""
    changed = 0
    for path in sorted(DST.glob("*.py")):
        text = path.read_text(encoding="utf-8")
        new_text = text
        hits: list[str] = []
        for old, rep in RULES:
            count = new_text.count(old)
            if count:
                hits.append(f"{old[:12]}->{rep[:12]} x{count}")
                new_text = new_text.replace(old, rep)
        if new_text != text:
            path.write_text(new_text, encoding="utf-8")
            changed += 1
            print(f"{path.name}: " + "; ".join(hits))
    print(f"\n完成：{changed} 个文件被修改")


if __name__ == "__main__":
    main()

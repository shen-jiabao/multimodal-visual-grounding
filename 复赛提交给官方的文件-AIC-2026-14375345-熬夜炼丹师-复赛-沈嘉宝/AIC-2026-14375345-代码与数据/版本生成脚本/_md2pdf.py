#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""技术文档.md → PDF（提交材料用）。

版式：封面题头 / 分级标题 + 色条 / 矢量表格（内容居中）/ 底纹代码块 / 引用块 / 页脚。
覆盖本文档实际用到的 Markdown 元素：
  · 标题 # ~ ####（首个 # 渲染为题头）
  · 表格（GFM 管道语法，含表头分隔行）
  · 围栏代码块 ```
  · 引用块 >
  · 列表项（· / - / * / 数字.）
  · 行内 **加粗**（按 markdown 渲染）、`code`（去反引号）

排版要点：Markdown 的**软换行**（同段内手动折行）按语义合并为一段——
中文之间不加空格、拉丁文之间补一个空格，避免出现断行残段与加粗失效。
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

from fpdf import FPDF

# 字体：正文等线（DengXian），标题/代码黑体（SimHei）
FONT_BODY = Path("C:/Windows/Fonts/Deng.ttf")
FONT_BODY_BOLD = Path("C:/Windows/Fonts/Dengb.ttf")
FONT_HEI = Path("C:/Windows/Fonts/simhei.ttf")

TEXT = (0, 0, 0)              # 文字一律纯黑
NAVY = (28, 52, 102)          # 装饰色：标题色条 / 框图描边 / 引用条
RULE = (205, 210, 220)
CODE_BG = (244, 246, 249)
CELL_ALT = (247, 249, 252)
TABLE_HEAD = (222, 231, 243)  # 表头浅底（保证黑字可读）
CELL_LH = 4.2                 # 表格行内行高
CELL_PAD = 1.8                # 表格单元格上下内边距

_ITEM_RE = re.compile(r"^([-·*]|\d+\.)\s")
# 中日韩文字与全角标点（用于判断拼接时是否需要空格）
_CJK_RE = re.compile(r"[\u3000-\u303f\u4e00-\u9fff\uff00-\uffef]")


class Doc(FPDF):
    """带页脚的技术文档 PDF 文档对象。"""

    def footer(self) -> None:
        self.set_y(-14)
        self.set_draw_color(*RULE)
        self.set_line_width(0.2)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin,
                  self.get_y())
        self.set_y(-12)
        self.set_font("hei", size=7.5)
        self.set_text_color(*TEXT)
        self.cell(0, 8, "AIC-2026-14375345 · 熬夜炼丹师 · 技术方案",
                  align="L")
        self.set_y(-12)
        self.cell(0, 8, f"- {self.page_no()} -", align="R")


def unescape(s: str) -> str:
    """还原 Markdown 转义字符。"""
    for a, b in (("\\[", "["), ("\\]", "]"), ("\\~", "~"), ("\\|", "|"),
                 ("\\*", "*"), ("\\_", "_"), ("\\@", "@")):
        s = s.replace(a, b)
    return s


def clean_plain(s: str) -> str:
    """去掉行内标记，仅保留文字（表格/标题用）。"""
    s = re.sub(r"\*\*(.+?)\*\*", r"\1", s)
    s = re.sub(r"`([^`]*)`", r"\1", s)
    return unescape(s).strip()


def clean_md(s: str) -> str:
    """保留 **加粗** 标记供 markdown 渲染，去反引号与转义。"""
    s = re.sub(r"`([^`]*)`", r"\1", s)
    return unescape(s)


def join_soft(parts: list[str]) -> str:
    """合并软换行的各行为一段（中文间不插空格，拉丁文间补一个空格）。"""
    out = parts[0].strip() if parts else ""
    for nxt in parts[1:]:
        nxt = nxt.strip()
        if not nxt:
            continue
        sep = "" if (_CJK_RE.search(out[-1:]) or _CJK_RE.search(nxt[:1])) \
            else " "
        out = out + sep + nxt
    return out


def read_blocks(lines: list[str]) -> list[tuple[str, object]]:
    """把 Markdown 行切成逻辑块：code / table / head / quote / item / para。"""
    blocks: list[tuple[str, object]] = []
    i, n = 0, len(lines)
    stop = ("#", "|", ">", "```")

    while i < n:
        s = lines[i].strip()
        if s.startswith("```"):
            info = s[3:].strip().lower()
            buf = []
            i += 1
            while i < n and not lines[i].strip().startswith("```"):
                buf.append(lines[i])
                i += 1
            i += 1
            blocks.append(("flow" if info == "flow" else "code", buf))
        elif s.startswith("|"):
            buf = []
            while i < n and lines[i].strip().startswith("|"):
                buf.append(lines[i])
                i += 1
            blocks.append(("table", buf))
        elif not s:
            blocks.append(("blank", ""))
            i += 1
        elif s.startswith("#"):
            blocks.append(("head", s))
            i += 1
        elif s.startswith(">"):
            buf = []
            while i < n and lines[i].strip().startswith(">"):
                buf.append(lines[i].strip().lstrip(">"))
                i += 1
            blocks.append(("quote", join_soft(buf)))
        elif _ITEM_RE.match(s):
            buf = [s]
            i += 1
            while (i < n and lines[i].strip()
                   and not lines[i].strip().startswith(stop)
                   and not _ITEM_RE.match(lines[i].strip())):
                buf.append(lines[i])
                i += 1
            blocks.append(("item", join_soft(buf)))
        else:
            buf = [s]
            i += 1
            while (i < n and lines[i].strip()
                   and not lines[i].strip().startswith(stop)
                   and not _ITEM_RE.match(lines[i].strip())):
                buf.append(lines[i])
                i += 1
            blocks.append(("para", join_soft(buf)))
    return blocks


def lines_of(pdf: FPDF, w: float, h: float, txt: str, md: bool) -> int:
    """估算文本在宽度 w 下的行数（优先用 fpdf 干跑）。"""
    if not txt:
        return 1
    try:
        return max(1, len(pdf.multi_cell(w, h, txt, dry_run=True,
                                         output="LINES", markdown=md,
                                         wrapmode="CHAR")))
    except Exception:
        return 1 + len(txt) // max(1, int(w / 3.4))


def rule(pdf: FPDF, y: float, color=RULE, width: float = 0.3) -> None:
    """在当前页 y 处画一条水平线。"""
    pdf.set_draw_color(*color)
    pdf.set_line_width(width)
    pdf.line(pdf.l_margin, y, pdf.w - pdf.r_margin, y)


def need_space(pdf: FPDF, h: float) -> None:
    """空间不足则换页。"""
    if pdf.get_y() + h > pdf.h - 22:
        pdf.add_page()


def table_layout(pdf: Doc, rows: list[list[str]]):
    """量算表格版式：返回 (各列宽, [(单元格文本, 各格行数, 行高), ...])。"""
    ncol = max(len(r) for r in rows)
    w = pdf.epw
    lens = [max(len(r[i]) if i < len(r) else 0 for r in rows)
            for i in range(ncol)]
    tot = sum(lens) or 1
    colw = [max(18.0, w * n / tot) for n in lens]
    scale = w / sum(colw)
    colw = [c * scale for c in colw]
    layout = []
    for ri, row in enumerate(rows):
        cells = [clean_plain(row[i]) if i < len(row) else ""
                 for i in range(ncol)]
        pdf.set_font("zh", "B" if ri == 0 else "", size=8.4)
        nl = [lines_of(pdf, colw[i] - 2.4, CELL_LH, cells[i], md=False)
              for i in range(ncol)]
        layout.append((cells, nl, CELL_LH * max(nl) + CELL_PAD))
    return colw, layout


def parse_table_rows(payload: list[str]) -> list[list[str]]:
    """把 GFM 表格的原始行解析为单元格矩阵（丢掉表头分隔行）。"""
    rows = []
    for raw in payload:
        cells = re.split(r"(?<!\\)\|", raw.strip().strip("|"))
        if not all(re.fullmatch(r":?-{3,}:?", c.strip()) for c in cells):
            rows.append(cells)
    return rows


def measure_block(pdf: Doc, kind: str, payload) -> float:
    """估算一个块占用的高度（mm），供「标题与后续内容同页」判断使用。"""
    if kind == "blank":
        return 1.2
    if kind == "head":
        level = len(payload) - len(payload.lstrip("#"))
        return {2: 13, 3: 11.5, 4: 10.5}.get(level, 11.5) * 0.62 + 8.5
    if kind == "para":
        pdf.set_font("zh", size=9.5)
        return lines_of(pdf, pdf.epw - 6, 5.4, clean_md(payload), True) * 5.4
    if kind == "item":
        pdf.set_font("zh", size=9.5)
        body = _ITEM_RE.sub("", payload, count=1)
        return lines_of(pdf, pdf.epw - 15, 5.4, clean_md(body), True) * 5.4
    if kind == "quote":
        pdf.set_font("zh", size=9)
        return lines_of(pdf, pdf.epw - 5, 5.0, payload, True) * 5.0
    if kind == "code":
        pdf.set_font("hei", size=8.2)
        n = sum(len(wrap_code(pdf, ln, pdf.epw - 6)) for ln in payload)
        return n * 4.2 + 2
    if kind == "flow":
        steps = [ln.strip() for ln in payload
                 if ln.strip() and ln.strip() not in ("▼", "↓", "|", "v")]
        return len(steps) * 7.6 + (len(steps) - 1) * 5.2 + 2
    if kind == "table":
        rows = parse_table_rows(payload)
        if not rows:
            return 0.0
        _, layout = table_layout(pdf, rows)
        return sum(h for _, _, h in layout) + 1.5
    return 0.0


def add_table(pdf: Doc, rows: list[list[str]]) -> None:
    """GFM 表格：整行等高、内容居中，并尽量让整张表不跨页。"""
    if not rows:
        return
    colw, layout = table_layout(pdf, rows)

    # 整表尽量落在同一页：本页放不下、但整页放得下时，先换页再画
    total = sum(r[2] for r in layout)
    if (total > pdf.h - 22 - pdf.get_y()
            and total <= pdf.h - pdf.t_margin - 22):
        pdf.add_page()

    pdf.set_text_color(*TEXT)
    pdf.set_draw_color(*RULE)
    pdf.set_line_width(0.2)
    for ri, (cells, nl, h) in enumerate(layout):
        if pdf.get_y() + h > pdf.h - 22:
            pdf.add_page()
        x0, y0 = pdf.get_x(), pdf.get_y()
        for i, c in enumerate(cells):
            x = x0 + sum(colw[:i])
            pdf.set_fill_color(*(TABLE_HEAD if ri == 0
                                 else (CELL_ALT if ri % 2
                                       else (255, 255, 255))))
            # 单元格撑满整行高度，文字在框内垂直居中
            pdf.rect(x, y0, colw[i], h, style="DF")
            pdf.set_xy(x + 1.2, y0 + (h - CELL_LH * nl[i]) / 2)
            pdf.multi_cell(colw[i] - 2.4, CELL_LH, c, border=0, align="C",
                           wrapmode="CHAR")
        pdf.set_xy(x0, y0 + h)
    pdf.set_text_color(0, 0, 0)
    pdf.ln(1.5)


def add_flow(pdf: Doc, buf: list[str]) -> None:
    """流程图框图：圆角方框 + 向下箭头（首尾框用深一档底色区分输入/输出）。"""
    steps = [ln.strip() for ln in buf
             if ln.strip() and ln.strip() not in ("▼", "↓", "|", "v")]
    if not steps:
        return
    max_w = pdf.epw - 6
    size = 9.5
    for cand in (9.5, 9, 8.5, 8, 7.5):
        pdf.set_font("hei", size=cand)
        if max(pdf.get_string_width(s) for s in steps) <= max_w - 12:
            size = cand
            break
    pdf.set_font("hei", size=size)
    box_w = min(max(pdf.get_string_width(s) for s in steps) + 12, max_w)
    box_h, gap = 7.6, 5.2
    total = len(steps) * box_h + (len(steps) - 1) * gap
    if pdf.get_y() + total > pdf.h - 22:
        pdf.add_page()
    x0 = pdf.l_margin + (pdf.epw - box_w) / 2
    y_top = pdf.get_y() + 0.5
    y = y_top
    for i, txt in enumerate(steps):
        end = i in (0, len(steps) - 1)
        pdf.set_fill_color(*(222, 231, 243) if end else (239, 243, 249))
        pdf.set_draw_color(*NAVY)
        pdf.set_line_width(0.35)
        pdf.rect(x0, y, box_w, box_h, style="DF", round_corners=True,
                 corner_radius=1.2)
        pdf.set_text_color(*TEXT)
        pdf.set_xy(x0, y)
        pdf.cell(box_w, box_h, txt, align="C")
        if i < len(steps) - 1:
            ax = x0 + box_w / 2
            y1, y2 = y + box_h, y + box_h + gap
            pdf.set_draw_color(*NAVY)
            pdf.set_line_width(0.45)
            pdf.line(ax, y1, ax, y2 - 1.5)
            pdf.set_fill_color(*NAVY)
            pdf.polygon([(ax - 1.1, y2 - 1.6), (ax + 1.1, y2 - 1.6), (ax, y2)],
                        style="F")
        y += box_h + gap
    pdf.set_text_color(0, 0, 0)
    pdf.set_y(y_top + total)
    pdf.ln(2)


def wrap_code(pdf: FPDF, text: str, width: float) -> list[str]:
    """把超出页宽的代码行折行；续行按原缩进再加 4 空格，便于区分。"""
    if not text.strip() or pdf.get_string_width(text) <= width:
        return [text]
    hang = " " * (len(text) - len(text.lstrip()) + 4)
    lines, cur = [], ""
    for tok in text.split(" "):
        cand = tok if cur == "" else cur + " " + tok
        if cur and pdf.get_string_width(cand) > width:
            lines.append(cur)
            cur = hang + tok
        else:
            cur = cand
        while pdf.get_string_width(cur) > width and len(cur) > 1:
            cut = len(cur)
            while cut > 1 and pdf.get_string_width(cur[:cut]) > width:
                cut -= 1
            lines.append(cur[:cut])
            cur = hang + cur[cut:]
    if cur:
        lines.append(cur)
    return lines


def add_code(pdf: Doc, buf: list[str], center: bool = False) -> None:
    """底纹代码块：浅底 + 左侧主色竖条；过宽的行自动折行，center 时逐行居中。"""
    if not buf:
        return
    pdf.set_font("hei", size=8.2)
    h = 4.2
    x0 = pdf.l_margin
    w = pdf.epw
    rows: list[str] = []
    for ln in buf:
        text = ln.strip() if center else ln
        rows += wrap_code(pdf, text, w - 6) if text.strip() else [text]
    for text in rows:
        if pdf.get_y() + h > pdf.h - 22:
            pdf.add_page()
        y = pdf.get_y()
        pdf.set_fill_color(*CODE_BG)
        pdf.rect(x0, y, w, h, style="F")
        pdf.set_fill_color(*NAVY)
        pdf.rect(x0, y, 1.1, h, style="F")
        pdf.set_xy(x0 + 3, y)
        pdf.set_text_color(*TEXT)
        pdf.cell(w - 3, h, text if text.strip() else " ",
                 align="C" if center else "L",
                 new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(0, 0, 0)
    pdf.ln(2)


def render_head(pdf: Doc, level: int, txt: str, first_h1: bool) -> bool:
    """渲染标题；返回更新后的 first_h1。"""
    if level == 1 and first_h1:
        pdf.ln(6)
        rule(pdf, pdf.get_y(), NAVY, 0.6)
        pdf.ln(6)
        pdf.set_font("hei", size=19)
        pdf.set_text_color(*TEXT)
        pdf.multi_cell(0, 12, txt, align="C", wrapmode="CHAR",
                       new_x="LMARGIN", new_y="NEXT")
        pdf.ln(5)
        rule(pdf, pdf.get_y(), NAVY, 0.6)
        pdf.ln(4)
        pdf.set_text_color(0, 0, 0)
        return False

    size = {2: 13, 3: 11.5, 4: 10.5}.get(level, 11.5)
    need_space(pdf, size * 0.6 + 8)
    pdf.ln(3.5)
    y = pdf.get_y()
    if level <= 2:
        pdf.set_fill_color(*NAVY)
        pdf.rect(pdf.l_margin, y + 1.2, 1.6, size * 0.42, style="F")
    pdf.set_font("hei", size=size)
    pdf.set_text_color(*TEXT)
    pdf.set_x(pdf.l_margin + (4 if level <= 2 else 0))
    pdf.multi_cell(0, size * 0.62, txt, align="L", wrapmode="CHAR",
                   new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(0, 0, 0)
    if level == 2:
        pdf.ln(1.2)
        rule(pdf, pdf.get_y(), RULE, 0.25)
    pdf.ln(1.2)
    return first_h1


def render_quote(pdf: Doc, txt: str) -> None:
    """引用块：左侧主色竖条；文本先去掉行内标记（反引号等）再渲染。"""
    txt = clean_md(txt)
    pdf.set_font("zh", size=9)
    nl = lines_of(pdf, pdf.epw - 5, 5.0, txt, md=True)
    y = pdf.get_y()
    pdf.set_fill_color(*NAVY)
    pdf.rect(pdf.l_margin, y, 1.1, 5.0 * nl, style="F")
    pdf.set_x(pdf.l_margin + 4)
    pdf.set_text_color(*TEXT)
    pdf.multi_cell(pdf.epw - 5, 5.0, txt, markdown=True, align="L",
                   wrapmode="CHAR", new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(0, 0, 0)


def render_item(pdf: Doc, txt: str) -> None:
    """列表项：主色项目符号 + 悬挂缩进。"""
    mark = re.match(r"^(\d+\.)", txt)
    bullet = mark.group(1) if mark else "•"
    body = _ITEM_RE.sub("", txt, count=1)
    pdf.set_font("zh", size=9.5)
    pdf.set_x(pdf.l_margin + 3)
    pdf.set_text_color(*TEXT)
    pdf.cell(6, 5.4, bullet)
    pdf.multi_cell(pdf.epw - 9, 5.4, clean_md(body), markdown=True,
                   align="L", wrapmode="CHAR", new_x="LMARGIN",
                   new_y="NEXT")
    pdf.set_text_color(0, 0, 0)


def render_para(pdf: Doc, txt: str) -> None:
    """普通段落。"""
    pdf.set_font("zh", size=9.5)
    pdf.set_text_color(*TEXT)
    pdf.multi_cell(0, 5.4, clean_md(txt), markdown=True, align="L",
                   wrapmode="CHAR", new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(0, 0, 0)


def main() -> int:
    src = Path(sys.argv[1])
    dst = Path(sys.argv[2])
    lines = src.read_text(encoding="utf-8").splitlines()

    pdf = Doc(format="A4")
    pdf.add_font("zh", "", str(FONT_BODY))
    pdf.add_font("zh", "B", str(FONT_BODY_BOLD))
    pdf.add_font("hei", "", str(FONT_HEI))
    pdf.set_margins(17, 16, 17)
    pdf.set_auto_page_break(True, 20)
    pdf.add_page()

    blocks = read_blocks(lines)
    first_h1 = True
    for idx, (kind, payload) in enumerate(blocks):
        if kind == "head":
            # 标题与其后紧随的块同页：本页放不下就把标题一并移到下一页
            need = measure_block(pdf, kind, payload)
            nxt = idx + 1
            while nxt < len(blocks) and blocks[nxt][0] == "blank":
                nxt += 1
            if nxt < len(blocks):
                need += measure_block(pdf, blocks[nxt][0], blocks[nxt][1])
            if pdf.get_y() + need > pdf.h - 22:
                pdf.add_page()
            level = len(payload) - len(payload.lstrip("#"))
            first_h1 = render_head(pdf, level,
                                   clean_plain(payload.lstrip("#").strip()),
                                   first_h1)
        elif kind == "code":
            add_code(pdf, payload)
        elif kind == "flow":
            add_flow(pdf, payload)
        elif kind == "table":
            add_table(pdf, parse_table_rows(payload))
        elif kind == "blank":
            pdf.ln(1.2)
        elif kind == "quote":
            render_quote(pdf, payload)
        elif kind == "item":
            render_item(pdf, payload)
        else:
            render_para(pdf, payload)

    pdf.output(str(dst))
    print(f"已写 {dst}（{dst.stat().st_size / 1024:.0f} KB，"
          f"{pdf.page_no()} 页）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

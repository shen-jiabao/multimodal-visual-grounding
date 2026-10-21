// 自证链路答辩 PPT —— 构建脚本（pptxgenjs）
// 主题：取证链路（深墨青 / 青绿 / 琥珀）
const pptxgen = require("pptxgenjs");

const pres = new pptxgen();
pres.layout = "LAYOUT_16x9";
pres.author = "AIC-2026-14375345 熬夜炼丹师";
pres.title = "AIC-2026-14375345 自证链路";

// ============================================================
// 幻灯片尺寸 / 内容框（先声明，后统一使用）
// ============================================================
const SLIDE_W = 10;
const SLIDE_H = 5.625;
const M = 0.45;
const CX = M;
const CY = M;
const CW = SLIDE_W - 2 * M;      // 9.1
const CH = SLIDE_H - 2 * M;      // 4.725

// 配色
const C = {
  ink: "0B2027",
  ink2: "12303A",
  teal: "028090",
  teal2: "00A896",
  amber: "F2A93B",
  pass: "1B7F4B",
  bg: "F5F9FA",
  card: "FFFFFF",
  line: "D6E2E6",
  muted: "5B7280",
};

const FT = "SimSun";              // 标题（衬线）
const FB = "Microsoft YaHei";     // 正文

// 证据附图目录（绝对路径，构建时读入并嵌入 PPTX）
const IMG = "d:/Desktop/2026AiC/答辩材料/AIC-2026-14375345-熬夜炼丹师-复赛-沈嘉宝/"
  + "AIC-2026-14375345-代码与数据/远端现场证据/服务器取证/附图/";

// ============================================================
// 容器系统（文本防溢出 + 相对定位）—— 必需
// ============================================================
function createVirtualNode(type, data, parentX = 0, parentY = 0) {
  const opts = data.opts || {};
  const node = {
    type, data,
    absX: parentX + (opts.x || 0),
    absY: parentY + (opts.y || 0),
    w: opts.w || 0, h: opts.h || 0,
    children: []
  };
  node.addShape = function (shapeType, o = {}) {
    const ch = createVirtualNode("shape", { shapeType, opts: o }, node.absX, node.absY);
    node.children.push(ch); return ch;
  };
  node.addText = function (text, o = {}) {
    const safe = { fit: "shrink", ...o };
    const re = /^(?:[\u2022\u2023\u25E6\u2043\u2219\u00B7\u25CF\u25CB\u2013\u2014]\s*|\-\s+)/;
    if (Array.isArray(text)) {
      text = text.map(it => (it && it.options && it.options.bullet && typeof it.text === "string")
        ? { ...it, text: it.text.replace(re, "") } : it);
    }
    const ch = createVirtualNode("text", { text, opts: safe }, node.absX, node.absY);
    node.children.push(ch); return ch;
  };
  node.addImage = function (o = {}) {
    const ch = createVirtualNode("image", { opts: o }, node.absX, node.absY);
    node.children.push(ch); return ch;
  };
  node.addTable = function (rows, o = {}) {
    const ch = createVirtualNode("table", { tableData: rows, opts: o }, node.absX, node.absY);
    node.children.push(ch); return ch;
  };
  return node;
}
function flattenNode(node, realSlide) {
  const abs = { ...node.data.opts, x: node.absX, y: node.absY };
  if (node.type === "shape") realSlide.addShape(node.data.shapeType, abs);
  else if (node.type === "text") realSlide.addText(node.data.text, abs);
  else if (node.type === "image") realSlide.addImage(abs);
  else if (node.type === "table") realSlide.addTable(node.data.tableData, abs);
  node.children.forEach(c => flattenNode(c, realSlide));
}
const _addSlide = pres.addSlide.bind(pres);
pres.addSlide = function (options) {
  const real = _addSlide(options);
  const v = {
    children: [], _real: real,
    set background(x) { real.background = x; },
    get background() { return real.background; },
    addShape(t, o = {}) { const n = createVirtualNode("shape", { shapeType: t, opts: o }, 0, 0); this.children.push(n); return n; },
    addText(t, o = {}) { const n = createVirtualNode("text", { text: t, opts: { fit: "shrink", ...o } }, 0, 0); this.children.push(n); return n; },
    addImage(o = {}) { const n = createVirtualNode("image", { opts: o }, 0, 0); this.children.push(n); return n; },
    addTable(r, o = {}) { const n = createVirtualNode("table", { tableData: r, opts: o }, 0, 0); this.children.push(n); return n; },
    addChart(t, d, o = {}) { real.addChart(t, d, o); },
    render() { this.children.forEach(c => flattenNode(c, real)); }
  };
  return v;
};
// ============================================================

// ---------- 通用组件 ----------
// 页眉：链环编号 + 标题 + 细副标题（不使用横线分隔）
function header(slide, no, zh, en) {
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x: CX, y: CY, w: 0.52, h: 0.52, rectRadius: 0.08,
    fill: { color: C.teal }, line: { color: C.teal, width: 0 }
  }).addText(no, {
    x: 0, y: 0, w: 0.52, h: 0.52, align: "center", valign: "middle",
    fontFace: FT, fontSize: 16, bold: true, color: "FFFFFF", margin: 0
  });
  slide.addText(zh, {
    x: CX + 0.68, y: CY - 0.02, w: CW - 0.68, h: 0.42,
    fontFace: FT, fontSize: 23, bold: true, color: C.ink,
    charSpacing: 1, valign: "middle", margin: 0
  });
  if (en) {
    slide.addText(en, {
      x: CX + 0.68, y: CY + 0.38, w: CW - 0.68, h: 0.22,
      fontFace: FB, fontSize: 9.5, color: C.muted, charSpacing: 0.5,
      valign: "middle", margin: 0
    });
  }
}
// 卡片（左侧青色粗边）
function card(slide, x, y, w, h, opts = {}) {
  const box = slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x, y, w, h, rectRadius: 0.06,
    fill: { color: opts.fill || C.card },
    line: { color: opts.line || C.line, width: 1 },
    shadow: { type: "outer", color: "9FB3B8", blur: 6, offset: 2, angle: 90, opacity: 0.28 }
  });
  box.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 0.001, w: 0.055, h: h - 0.002,
    fill: { color: opts.accent || C.teal }, line: { color: opts.accent || C.teal, width: 0 }
  });
  return box;
}
function footNote(slide, t) {
  slide.addText(t, {
    x: CX, y: SLIDE_H - M - 0.06, w: CW, h: 0.2,
    fontFace: FB, fontSize: 8, color: C.muted, margin: 0
  });
}

// ============================================================
// 1. 封面
// ============================================================
{
  const s = pres.addSlide();
  s.background = { color: C.ink };
  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 0.22, h: SLIDE_H, fill: { color: C.teal }, line: { width: 0 } });
  s.addShape(pres.shapes.OVAL, { x: 7.55, y: 0.15, w: 2.3, h: 2.3, fill: { color: C.ink2 }, line: { width: 0 } });
  s.addShape(pres.shapes.OVAL, { x: 7.9, y: 3.35, w: 1.95, h: 1.95, fill: { color: "10303C" }, line: { width: 0 } });

  s.addText("自 证 链 路", {
    x: 0.75, y: 1.28, w: 7.6, h: 1.0,
    fontFace: FT, fontSize: 44, bold: true, color: "FFFFFF", charSpacing: 2.5, margin: 0
  });
  s.addText("从赛事数据 → 自研代码 → 产物 → 榜单结果文件的完整可核验证据链", {
    x: 0.78, y: 2.30, w: 7.6, h: 0.42,
    fontFace: FB, fontSize: 14, color: C.teal2, charSpacing: 0.5, margin: 0
  });
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x: 0.78, y: 2.95, w: 4.5, h: 0.62, rectRadius: 0.08,
    fill: { color: "0E3A44" }, line: { color: C.teal, width: 1 }
  }).addText([
    { text: "线上成绩  ", options: { fontFace: FB, fontSize: 13, color: "CFE7EA" } },
    { text: "ACC@0.5 = 0.7963", options: { fontFace: FT, fontSize: 20, bold: true, color: C.amber } }
  ], { x: 0, y: 0, w: 4.5, h: 0.62, align: "center", valign: "middle", margin: 0 });

  s.addText([
    { text: "全球校园人工智能算法精英大赛 · 算法挑战赛\n", options: { fontFace: FB, fontSize: 11.5, color: "9FB9BE" } },
    { text: "基于大模型的多模态视觉理解与推理", options: { fontFace: FB, fontSize: 12, color: "CFE7EA" } }
  ], { x: 0.78, y: 3.78, w: 7.6, h: 0.62, lineSpacing: 18, margin: 0 });

  s.addText("参赛团队编号 AIC-2026-14375345　｜　熬夜炼丹师　｜　队长 沈嘉宝", {
    x: 0.78, y: 4.55, w: 8.4, h: 0.3,
    fontFace: FB, fontSize: 11, color: "7E9AA0", margin: 0
  });
  s.render();
}

// ============================================================
// 2. 目录
// ============================================================
{
  const s = pres.addSlide();
  s.background = { color: C.bg };
  header(s, "目", "目　录", "CONTENTS");

  const items = [
    ["01", "自证目标", "三条可独立核验的主张：归属 · 可复算 · 自主"],
    ["02", "交付件指纹", "0.7963 结果文件的唯一身份与产生方式"],
    ["03", "全流程链路", "14 步：从数据接入到交付归档"],
    ["04", "证据链", "服务器日志原文 · 文件时间戳 · 会话时间线 · 可复算 · 同源指纹"],
    ["05", "反证与核验", "排除五类质疑 + 一键自检 SOP"],
    ["06", "结论与合规", "结论、合规声明与复现边界"],
  ];
  const cw = (CW - 0.32) / 2, ch = 0.98;
  items.forEach((it, i) => {
    const x = CX + (i % 2) * (cw + 0.32);
    const y = CY + 1.05 + Math.floor(i / 2) * (ch + 0.22);
    const box = card(s, x, y, cw, ch, { accent: i % 2 ? C.amber : C.teal });
    box.addText(it[0], {
      x: 0.22, y: 0.14, w: 0.62, h: 0.7, fontFace: FT, fontSize: 24, bold: true,
      color: i % 2 ? C.amber : C.teal, valign: "middle", margin: 0
    });
    box.addText(it[1], {
      x: 0.92, y: 0.13, w: cw - 1.1, h: 0.32, fontFace: FT, fontSize: 15, bold: true,
      color: C.ink, charSpacing: 1, margin: 0
    });
    box.addText(it[2], {
      x: 0.92, y: 0.47, w: cw - 1.1, h: 0.36, fontFace: FB, fontSize: 9.5,
      color: C.muted, margin: 0
    });
  });
  s.render();
}

// ============================================================
// 3. 自证目标：三条可核验主张
// ============================================================
{
  const s = pres.addSlide();
  s.background = { color: C.bg };
  header(s, "01", "自证目标：把「是我们做的」拆成三条可核验主张", "THREE INDEPENDENTLY VERIFIABLE CLAIMS");

  const claim = [
    ["A", "成绩归属本队", "赛事平台的提交谱系", [
      "每版提交的得分 + SHA256 + 读数链",
      "复赛 20+ 次、初赛 12 次提交连续可查",
      "被否件与测量件一并如实留档"
    ], C.teal],
    ["B", "结果可由本队代码产生", "离线确定性复算", [
      "纯 CPU 逐条重建交付件（已实测一致）",
      "六桶互斥装配器 + 20 项硬断言",
      "八组受控夹具回归全过"
    ], C.amber],
    ["C", "过程由本队自主推进", "过程痕迹与现场物证", [
      "判据预登记 17 份（写于跑之前）",
      "云端实例现场证据 + 运维脚本归档 246",
      "环境实录、权重指纹、消融复盘"
    ], C.teal2],
  ];
  const cw = (CW - 0.36) / 3, ch = 2.62;
  claim.forEach((c, i) => {
    const x = CX + i * (cw + 0.18);
    const y = CY + 1.02;
    const box = card(s, x, y, cw, ch, { accent: c[4] });
    box.addShape(pres.shapes.OVAL, {
      x: 0.24, y: 0.22, w: 0.62, h: 0.62,
      fill: { color: c[4] }, line: { width: 0 }
    }).addText(c[0], {
      x: 0, y: 0, w: 0.62, h: 0.62, align: "center", valign: "middle",
      fontFace: FT, fontSize: 25, bold: true, color: "FFFFFF", margin: 0
    });
    box.addText(c[1], {
      x: 0.24, y: 0.98, w: cw - 0.48, h: 0.56,
      fontFace: FT, fontSize: 14.5, bold: true, color: C.ink, charSpacing: 0.8, margin: 0
    });
    box.addText(c[2], {
      x: 0.24, y: 1.52, w: cw - 0.48, h: 0.24,
      fontFace: FB, fontSize: 9, color: c[4], bold: true, margin: 0
    });
    box.addText(c[3].map((t, j) => ({
      text: t, options: { bullet: true, breakLine: j < c[3].length - 1 }
    })), {
      x: 0.24, y: 1.80, w: cw - 0.42, h: 0.76,
      fontFace: FB, fontSize: 9, color: C.muted, lineSpacing: 14, margin: 0
    });
  });

  s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x: CX, y: CY + 3.82, w: CW, h: 0.62, rectRadius: 0.06,
    fill: { color: "E4F1F3" }, line: { color: C.teal, width: 1 }
  }).addText("三条同时成立 ⇒ 该结果文件是本队自主设计与实现的产物，非转载、非他人作品、无人工标注。", {
    x: 0.25, y: 0, w: CW - 0.5, h: 0.62, valign: "middle",
    fontFace: FB, fontSize: 11.5, bold: true, color: C.ink, margin: 0
  });
  s.render();
}

// ============================================================
// 4. 交付件指纹
// ============================================================
{
  const s = pres.addSlide();
  s.background = { color: C.bg };
  header(s, "02", "交付件指纹：唯一的身份与产生方式", "THE DELIVERABLE — IDENTITY & PROVENANCE");

  const b = card(s, CX, CY + 1.0, CW, 1.02, { accent: C.amber, fill: "FFF8EC" });
  b.addText("result_semi_q38_nonord.zip", {
    x: 0.3, y: 0.14, w: 5.2, h: 0.36, fontFace: FB, fontSize: 15, bold: true, color: C.ink, margin: 0
  });
  b.addText("SHA256　FB6C2E471A4F5E8907318FF540A5CAABAEC17A29ECAFC17EB651F512A4AE8F35", {
    x: 0.3, y: 0.55, w: 8.5, h: 0.32, fontFace: "Consolas", fontSize: 10, color: C.muted, margin: 0
  });

  const stats = [
    ["0.7963", "线上 ACC@0.5", C.teal],
    ["5690", "提交条数", C.teal],
    ["519", "本次替换框数", C.amber],
    ["+101", "净增正确条数", C.pass],
  ];
  const sw = (CW - 0.54) / 4;
  stats.forEach((st, i) => {
    const x = CX + i * (sw + 0.18);
    const box = card(s, x, CY + 2.18, sw, 1.02, { accent: st[2] });
    box.addText(st[0], {
      x: 0, y: 0.13, w: sw, h: 0.5, align: "center",
      fontFace: FT, fontSize: 27, bold: true, color: st[2], margin: 0
    });
    box.addText(st[1], {
      x: 0, y: 0.64, w: sw, h: 0.26, align: "center",
      fontFace: FB, fontSize: 9.5, color: C.muted, margin: 0
    });
  });

  const t = card(s, CX, CY + 3.36, CW, 0.86, { accent: C.teal });
  t.addText([
    { text: "技术路线　", options: { fontFace: FB, fontSize: 10, bold: true, color: C.teal } },
    { text: "开源大模型零样本直接定位 ＋ 确定性规则管序数（不依赖模型猜序数）＋ 六桶互斥装配器出件", options: { fontFace: FB, fontSize: 10.5, color: C.ink } },
    { text: "\n交付件来源　", options: { fontFace: FB, fontSize: 10, bold: true, color: C.amber } },
    { text: "L6：Qwen3.8-27B 单变量重跑非序数 4587 条 ⇒ 替换 519 处、净 +101 条（19.4%，8.8σ，非噪声）", options: { fontFace: FB, fontSize: 10.5, color: C.ink } },
  ], { x: 0.28, y: 0.08, w: CW - 0.56, h: 0.72, lineSpacing: 15, margin: 0 });
  s.render();
}

// ============================================================
// 5. 全流程 14 步链路
// ============================================================
{
  const s = pres.addSlide();
  s.background = { color: C.bg };
  header(s, "03", "全流程链路：14 步，从数据接入到交付归档", "FULL PIPELINE — 14 STEPS");

  const steps = [
    ["1", "数据接入", "键集确认"],
    ["2", "格式勘查", "两子群"],
    ["3", "分层", "1103 / 4587"],
    ["4", "L1 · 8B 全量", "0.7459"],
    ["5", "补框链", "零空框"],
    ["6", "L2 · 序数规则", "0.7598"],
    ["7", "停损判定", "封口负向"],
    ["8", "L3 · 32B-AWQ", "0.7742"],
    ["9", "L4 · 同机制", "0.7766"],
    ["10", "L5 · bf16", "0.7786"],
    ["11", "L6 · Qwen3.8-27B", "0.7963"],
    ["12", "六桶装配", "519 处"],
    ["13", "硬约束校验", "4/4"],
    ["14", "交付归档", "本包"],
  ];
  const cols = 5, bw = 1.62, gap = 0.09;
  const rowY = [CY + 0.95, CY + 2.0, CY + 3.05];
  const bh = 0.82;
  const gridW = cols * bw + (cols - 1) * gap;
  steps.forEach((st, i) => {
    const r = Math.floor(i / cols), c = i % cols;
    const x = CX + c * (bw + gap);
    const y = rowY[r];
    const isFinal = i === 10, isKey = i === 11 || i === 12;
    const accent = isFinal ? C.amber : (isKey ? C.teal : C.teal);
    const box = s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
      x, y, w: bw, h: bh, rectRadius: 0.05,
      fill: { color: isFinal ? "FFF4E0" : C.card },
      line: { color: accent, width: isFinal ? 1.6 : 1 }
    });
    box.addShape(pres.shapes.ROUNDED_RECTANGLE, {
      x: 0.08, y: 0.08, w: 0.3, h: 0.22, rectRadius: 0.04,
      fill: { color: accent }, line: { width: 0 }
    }).addText(st[0], {
      x: 0, y: 0, w: 0.3, h: 0.22, align: "center", valign: "middle",
      fontFace: FB, fontSize: 8, bold: true, color: "FFFFFF", margin: 0
    });
    box.addText(st[1], {
      x: 0.42, y: 0.07, w: bw - 0.5, h: 0.24,
      fontFace: FB, fontSize: 8.5, bold: true, color: C.ink, valign: "middle", margin: 0
    });
    box.addText(st[2], {
      x: 0.08, y: 0.38, w: bw - 0.16, h: 0.34,
      fontFace: FT, fontSize: 11.5, bold: true, color: accent, valign: "middle", margin: 0
    });
  });
  // 行末折返标记（表示链路延续到下一行）
  [0, 1].forEach(r => {
    s.addText("▼", {
      x: CX + gridW + 0.06, y: rowY[r] + bh / 2 - 0.12, w: 0.3, h: 0.24,
      fontFace: FB, fontSize: 11, bold: true, color: C.teal, align: "center", valign: "middle", margin: 0
    });
  });
  s.addText("每一步都留下：命令 · 产物 · SHA256 指纹 · 时间戳（详见《全流程逐步实录_模型与提示词.md》）", {
    x: CX, y: SLIDE_H - M - 0.32, w: CW, h: 0.24,
    fontFace: FB, fontSize: 9, color: C.muted, margin: 0
  });
  s.render();
}

// ============================================================
// 6. 提交谱系
// ============================================================
{
  const s = pres.addSlide();
  s.background = { color: C.bg };
  header(s, "04", "提交谱系：每一版都留下得分与指纹", "SUBMISSION LINEAGE — SCORE & HASH PER VERSION");

  const rows = [
    ["0.4378", "GD 三模态", "弃"],
    ["0.7459", "8B 全量 + 补框链", "基线"],
    ["0.7598", "序数确定性规则链", "+79 序数"],
    ["0.7742", "32B-AWQ 重跑非序数", "+82"],
    ["0.7766", "序数同机制替换 B′", "+14"],
    ["0.7786", "32B bf16 精度修复", "+11"],
    ["0.7963", "Qwen3.8-27B 跨代换模型", "★交付件"],
  ];
  const x0 = CX, y0 = CY + 1.0, rh = 0.4, bw2 = 3.5;
  const maxV = 0.8, minV = 0.4;
  rows.forEach((r, i) => {
    const y = y0 + i * (rh + 0.1);
    const v = parseFloat(r[0]);
    const w = Math.max(0.5, (v - minV) / (maxV - minV) * bw2);
    const isLast = i === rows.length - 1, isFirst = i === 0;
    const col = isLast ? C.amber : (isFirst ? "C0605E" : C.teal);
    s.addText(r[0], {
      x: x0, y, w: 0.72, h: rh, fontFace: FT, fontSize: 13, bold: true,
      color: col, valign: "middle", margin: 0
    });
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
      x: x0 + 0.78, y: y + 0.04, w, h: rh - 0.08, rectRadius: 0.03,
      fill: { color: col }, line: { width: 0 }
    });
    s.addText(r[1], {
      x: x0 + 0.88, y, w: 3.0, h: rh, fontFace: FB, fontSize: 9.5,
      color: isLast || isFirst ? "FFFFFF" : "0B2027", valign: "middle", margin: 0
    });
    s.addText(r[2], {
      x: x0 + 4.35, y, w: 1.5, h: rh, fontFace: FB, fontSize: 9.5,
      color: isLast ? C.amber : C.muted, bold: isLast, valign: "middle", margin: 0
    });
  });

  const side = card(s, CX + 6.15, CY + 1.0, CW - 6.15, 3.42, { accent: C.teal2, fill: "EEF7F8" });
  side.addText("谱系的两个特点", {
    x: 0.24, y: 0.2, w: 2.6, h: 0.3, fontFace: FT, fontSize: 14, bold: true, color: C.ink, charSpacing: 0.8, margin: 0
  });
  side.addText([
    { text: "不只报喜", options: { bold: true, color: C.ink } },
    { text: "　被否件（0.4378 / 0.7541 / 0.7566 / 0.7592 / 0.7729 / 0.7492 …）与测量件全部留档\n", options: { color: C.muted } },
    { text: "数字互相约束", options: { bold: true, color: C.ink } },
    { text: "　0.7963×5690 = 4531 = 非序数 3842 + 序数 689；每级都满足该恒等式\n", options: { color: C.muted } },
    { text: "初赛阶段", options: { bold: true, color: C.ink } },
    { text: "　12 次提交 0.6693 → 0.7827，同样逐版留得分与 SHA256\n", options: { color: C.muted } },
    { text: "平台侧证据", options: { bold: true, color: C.ink } },
    { text: "　读数口径：1 条 = 0.000176；测量件按历史最高分记录，因此不拉低成绩", options: { color: C.muted } },
  ], { x: 0.24, y: 0.58, w: 2.72, h: 2.6, fontFace: FB, fontSize: 8.5, lineSpacing: 13.5, margin: 0 });
  s.render();
}

// ============================================================
// 证据① 服务器文件时间戳（full-iso）
// ============================================================
{
  const s = pres.addSlide();
  s.background = { color: C.bg };
  header(s, "04", "证据① 服务器文件时间戳：q38 全链纳秒级 mtime", "EVIDENCE 1 — SERVER FILE TIMESTAMPS (full-iso, read-only stat)");
  s.addImage({ path: IMG + "e1_服务器文件时间戳_1600x699.png", x: 0.9, y: 1.28, w: 8.2, h: 3.582 });
  s.addText("来源：远端实例 /root/rivermind-data 只读 stat 采集（未修改任何文件）；原始采集文本随包：远端现场证据/服务器取证/服务器取证_20261015.txt", {
    x: CX, y: 4.93, w: CW, h: 0.2, fontFace: FB, fontSize: 8, color: C.muted, margin: 0
  });
  s.render();
}

// ============================================================
// 证据② 服务器日志原文：权重逐片双验
// ============================================================
{
  const s = pres.addSlide();
  s.background = { color: C.bg };
  header(s, "04", "证据② 服务器日志原文：权重逐片「大小 + sha256」双验", "EVIDENCE 2 — q38_dl.log (18 SHARDS, DUAL-VERIFIED)");
  s.addImage({ path: IMG + "e2_权重下载双验日志_1600x469.png", x: CX, y: 1.28, w: CW, h: 2.667 });
  const b = card(s, CX, CY + 3.62, CW, 0.96, { accent: C.teal2, fill: "EEF7F8" });
  b.addText("这份日志证明什么", {
    x: 0.26, y: 0.1, w: 3.0, h: 0.26, fontFace: FT, fontSize: 12.5, bold: true, color: C.ink, charSpacing: 0.6, margin: 0
  });
  b.addText([
    { text: "① 逐片双验：", options: { bold: true, color: C.teal } },
    { text: "先比字节数、再比 sha256，两关都过才记「✓」\n", options: { color: C.muted } },
    { text: "② 断点续传 + 跨分片并发：", options: { bold: true, color: C.teal } },
    { text: "w* worker 与 mon 并行，03:04→03:51 共 47 分钟完成 51.75 GiB\n", options: { color: C.muted } },
    { text: "③ 终态可判：", options: { bold: true, color: C.teal } },
    { text: "末行「全部 18 个分片就位，monitor 退出」——逐片校验后的显式终态", options: { color: C.muted } },
  ], { x: 0.26, y: 0.38, w: CW - 0.52, h: 0.5, fontFace: FB, fontSize: 8.5, lineSpacing: 12.5, margin: 0 });
  s.render();
}

// ============================================================
// 证据③ 服务器日志原文：起服自证
// ============================================================
{
  const s = pres.addSlide();
  s.background = { color: C.bg };
  header(s, "04", "证据③ 服务器日志原文：起服自证（回读 /v1/models）", "EVIDENCE 3 — _q38_serve.log (SERVED-MODEL SELF-CHECK)");
  s.addImage({ path: IMG + "e3_起服自证日志_1600x699.png", x: 0.45, y: 1.28, w: 5.5, h: 2.4 });
  const b = card(s, CX + 5.7, CY + 0.83, CW - 5.7, 2.4, { accent: C.amber, fill: "FFF8EC" });
  b.addText("关键三问，日志逐一回答", {
    x: 0.24, y: 0.12, w: 3.1, h: 0.26, fontFace: FT, fontSize: 12.5, bold: true, color: C.ink, charSpacing: 0.6, margin: 0
  });
  b.addText([
    { text: "起的哪个模型？\n", options: { bold: true, color: C.ink } },
    { text: "回读 /v1/models ⇒ id=qwen38-27b、root=models/Qwen3.8-27B（★防「起错模型」）\n", options: { color: C.muted } },
    { text: "权重齐不齐？\n", options: { bold: true, color: C.ink } },
    { text: "分片 18｜缺失 0｜合计 51.75 GiB｜权重条目 1199（本地校验，不联网）\n", options: { color: C.muted } },
    { text: "架构认不认？\n", options: { bold: true, color: C.ink } },
    { text: "架构登记 Qwen3_5ForConditionalGeneration: True", options: { color: C.muted } }
  ], { x: 0.24, y: 0.42, w: 3.16, h: 1.9, fontFace: FB, fontSize: 8.5, lineSpacing: 12.5, margin: 0 });
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x: CX, y: CY + 3.34, w: CW, h: 0.62, rectRadius: 0.06,
    fill: { color: "E4F1F3" }, line: { color: C.teal, width: 1 }
  }).addText("该文件末尾还写明了「下一步」：① 20 条冒烟 → ② 过闸后全量 4587 条非序数 → 装配（锚 ord 689 / 非序数 3741）——与最终交付件的桶分布完全对应。", {
    x: 0.28, y: 0, w: CW - 0.56, h: 0.62, valign: "middle", fontFace: FB, fontSize: 9.5, color: C.ink, margin: 0
  });
  s.render();
}

// ============================================================
// 证据④ vLLM 起服参数 + 目录统计
// ============================================================
{
  const s = pres.addSlide();
  s.background = { color: C.bg };
  header(s, "04", "证据④ vLLM 起服参数（毫秒级时间戳）与服务器目录统计", "EVIDENCE 4 — vLLM ARGS & SERVER DIRECTORY STATS");
  s.addImage({ path: IMG + "e4_vLLM起服参数_1600x469.png", x: 2.0, y: 1.26, w: 6.0, h: 1.758 });
  s.addImage({ path: IMG + "e5_目录统计与目检图时间戳_1600x492.png", x: 2.0, y: 3.14, w: 6.0, h: 1.845 });
  s.addText("环境为 cu130 + 单卡 96GB 现场实例（TP=1）；交付件阶段为 A100-40G 双卡（TP=2），两台实例不同，已在包内说明。", {
    x: CX, y: 5.02, w: CW, h: 0.16, fontFace: FB, fontSize: 7.5, color: C.muted, margin: 0
  });
  s.render();
}

// ============================================================
// 7. 判据预登记
// ============================================================
{
  const s = pres.addSlide();
  s.background = { color: C.bg };
  header(s, "04", "判据预登记：把判据写在结果之前", "PRE-REGISTERED CRITERIA (BEFORE THE RUN)");

  const b1 = card(s, CX, CY + 1.0, 4.42, 1.5, { accent: C.teal });
  b1.addText("17 份预登记件", {
    x: 0.26, y: 0.16, w: 3.9, h: 0.32, fontFace: FT, fontSize: 15, bold: true, color: C.ink, charSpacing: 0.8, margin: 0
  });
  b1.addText("作用域 / 闸门阈值 / 事前预期 / 止损线，全部写进文件后才开跑；禁止事后调阈值。另附冻结稿（D4A4BC35…）与跑后实录（56176E7A…）。", {
    x: 0.26, y: 0.54, w: 3.92, h: 0.84, fontFace: FB, fontSize: 9.5, color: C.muted, lineSpacing: 15, margin: 0
  });

  const b2 = card(s, CX + 4.68, CY + 1.0, CW - 4.68, 1.5, { accent: C.amber, fill: "FFF8EC" });
  b2.addText("★ 交付件路线的预登记", {
    x: 0.26, y: 0.16, w: 3.9, h: 0.32, fontFace: FT, fontSize: 15, bold: true, color: C.ink, charSpacing: 0.8, margin: 0
  });
  b2.addText([
    { text: "《_prereg_I.md》", options: { fontFace: "Consolas", fontSize: 9.5, bold: true, color: C.amber } },
    { text: "（SHA256 28D2F255…）事前明写：迁移条件只满足一半、", options: { color: C.muted } },
    { text: "「不给正数、可能为负」", options: { bold: true, color: C.ink } },
    { text: "、不是到 0.8 的路径，并预设止损线。", options: { color: C.muted } }
  ], { x: 0.26, y: 0.54, w: 3.92, h: 0.84, fontFace: FB, fontSize: 9.5, lineSpacing: 15, margin: 0 });

  const b3 = card(s, CX, CY + 2.68, CW, 1.44, { accent: C.pass, fill: "F0F8F2" });
  b3.addText("实测推翻了自己的判断 —— 这是转载或事后编造不可能具有的痕迹", {
    x: 0.28, y: 0.16, w: CW - 0.56, h: 0.3,
    fontFace: FT, fontSize: 13.5, bold: true, color: C.ink, charSpacing: 0.8, margin: 0
  });
  b3.addText([
    { text: "预登记预期　", options: { bold: true, color: C.muted } },
    { text: "「不给正数，可能为负」", options: { color: C.ink } },
    { text: "　→　实测结果　", options: { bold: true, color: C.muted } },
    { text: "519 处替换、净 +101 条（19.4%，8.8σ）", options: { bold: true, color: C.pass } },
    { text: "　⇒ 本队如实记入跑后实录，不修改冻结稿原文；另有「被撤回的两版说法」一并留档（见 技术报告素材_方法学结论.md）。", options: { color: C.muted } }
  ], { x: 0.28, y: 0.56, w: CW - 0.56, h: 0.76, fontFace: FB, fontSize: 10, lineSpacing: 15.5, margin: 0 });
  s.render();
}

// ============================================================
// 证据⑤ 开发会话时间戳 + 三路交叉印证
// ============================================================
{
  const s = pres.addSlide();
  s.background = { color: C.bg };
  header(s, "04", "证据⑤ 开发会话时间戳（AI 协作留痕）+ 三路时间线交叉印证", "EVIDENCE 5 — SESSION TIMESTAMPS × ARTIFACT MTIMES × SERVER MTIMES");
  s.addImage({ path: IMG + "e6_开发会话时间线_1600x472.png", x: 0.7, y: 1.26, w: 8.6, h: 2.538 });

  const b = card(s, CX, CY + 3.48, CW, 1.1, { accent: C.pass, fill: "F0F8F2" });
  b.addText("三路时间线互相咬合（会话时间 → 产物落地时间）", {
    x: 0.26, y: 0.08, w: CW - 0.52, h: 0.26, fontFace: FT, fontSize: 12, bold: true, color: C.ink, charSpacing: 0.6, margin: 0
  });
  const pairs = [
    ["09-27 16:33 会话「32B 推理进度 + 装配审计」", "→", "result_semi_n32_ordbase.zip　09-27 16:51"],
    ["10-03 12:41 会话「32B 推理实时进度」", "→", "result_semi_n32bf16_nonord.zip　10-03 14:13"],
    ["10-03 15:24 会话「决定用 Qwen3.8-27B 全量推理」", "→", "qwen38_direct.json ＋ ★交付件　10-03 21:04"],
  ];
  pairs.forEach((p, i) => {
    const y = 0.36 + i * 0.24;
    b.addText(p[0], { x: 0.26, y, w: 4.6, h: 0.22, fontFace: FB, fontSize: 8.5, color: C.muted, valign: "middle", margin: 0 });
    b.addText(p[1], { x: 4.88, y, w: 0.24, h: 0.22, fontFace: FB, fontSize: 8.5, color: C.pass, valign: "middle", margin: 0 });
    b.addText(p[2], { x: 5.12, y, w: 3.7, h: 0.22, fontFace: FB, fontSize: 8.5, bold: true, color: C.ink, valign: "middle", margin: 0 });
  });
  s.addText("会话记录完整 36 条（08-30 → 10-06，15 个日期）随包：远端现场证据/服务器取证/开发会话时间线_完整36条.txt（已脱敏）", {
    x: CX, y: 5.0, w: CW, h: 0.17, fontFace: FB, fontSize: 7.5, color: C.muted, margin: 0
  });
  s.render();
}

// ============================================================
// 8. 远端现场证据
// ============================================================
{
  const s = pres.addSlide();
  s.background = { color: C.bg };
  header(s, "04", "云端现场证据：环境自建 · 模型自下 · 闸门自跑", "ON-SITE EVIDENCE FROM THE CLOUD INSTANCE");

  const ev = [
    ["环境构建", "q38_env.log", "vLLM registry 登记 Qwen3_5ForConditionalGeneration（架构总数 365）；torch 2.11.0+cu130"],
    ["权重下载", "q38_dl.log", "18 个分片逐片「大小 + sha256 双验通过」，末行「全部 18 个分片就位」"],
    ["起服自证", "_q38_serve.log", "就绪后回读 /v1/models ⇒ id=qwen38-27b、root=models/Qwen3.8-27B（防起错模型）"],
    ["闸门 G2/G3", "_q38_gate20.json", "20 条冒烟可解析 17/20 通过；G3 人眼判据 4×5 目检拼版图随包"],
  ];
  const cw = (CW - 0.3) / 2, ch = 1.24;
  ev.forEach((e, i) => {
    const x = CX + (i % 2) * (cw + 0.3);
    const y = CY + 1.0 + Math.floor(i / 2) * (ch + 0.16);
    const box = card(s, x, y, cw, ch, { accent: i % 2 ? C.amber : C.teal });
    box.addText(e[0], {
      x: 0.26, y: 0.13, w: 1.5, h: 0.3, fontFace: FT, fontSize: 13.5, bold: true, color: C.ink, charSpacing: 0.8, margin: 0
    });
    box.addText(e[1], {
      x: 1.72, y: 0.15, w: cw - 2.0, h: 0.26, fontFace: "Consolas", fontSize: 9,
      color: i % 2 ? C.amber : C.teal, margin: 0
    });
    box.addText(e[2], {
      x: 0.26, y: 0.50, w: cw - 0.52, h: 0.64, fontFace: FB, fontSize: 9.5,
      color: C.muted, lineSpacing: 14, margin: 0
    });
  });

  s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x: CX, y: CY + 3.82, w: CW, h: 0.62, rectRadius: 0.06,
    fill: { color: "E4F1F3" }, line: { color: C.teal, width: 1 }
  }).addText([
    { text: "运维脚本归档 246 个　", options: { fontFace: FB, fontSize: 10, bold: true, color: C.teal } },
    { text: "其中 _get_serve_q38.sh、_q38_gate.py、_launch_q38.sh 等 6 个脚本的 SHA256 在「工作区归档 = 43504 实例 = 本包」三处逐位相同 ⇒ 同一套自研工具链。", options: { fontFace: FB, fontSize: 10, color: C.ink } }
  ], { x: 0.28, y: 0, w: CW - 0.56, h: 0.62, valign: "middle", margin: 0 });
  s.render();
}

// ============================================================
// 9. 可复算性
// ============================================================
{
  const s = pres.addSlide();
  s.background = { color: C.bg };
  header(s, "04", "可复算性：最硬的一条 —— 交付件可被离线逐条重建", "REPRODUCIBILITY WITHOUT GPU");

  const cmd = card(s, CX, CY + 1.0, CW, 1.24, { accent: C.ink, fill: "0B2027" });
  cmd.addText("python code/_assembly_32b.py --direct _qwen38_direct.json \\", {
    x: 0.28, y: 0.14, w: CW - 0.56, h: 0.24, fontFace: "Consolas", fontSize: 10, color: "8FD8C0", margin: 0
  });
  cmd.addText("    --base result_semi_n32bf16_nonord --base-score 0.7786 \\", {
    x: 0.28, y: 0.4, w: CW - 0.56, h: 0.24, fontFace: "Consolas", fontSize: 10, color: "8FD8C0", margin: 0
  });
  cmd.addText("    --ord-correct 689 --base-correct 3741 --stem result_semi_q38_nonord --zip", {
    x: 0.28, y: 0.66, w: CW - 0.56, h: 0.24, fontFace: "Consolas", fontSize: 10, color: "8FD8C0", margin: 0
  });
  cmd.addText("仅需标准库 + numpy；无需 GPU（推理段无随机性 ⇒ 装配与校验完全确定）", {
    x: 0.28, y: 0.94, w: CW - 0.56, h: 0.22, fontFace: FB, fontSize: 8.5, color: "7E9AA0", margin: 0
  });

  const buckets = [["无框 → 保留基线", 329, C.teal], ["面积>0.35 → 保留", 100, C.muted],
  ["与基线一致 → 不动", 3639, C.teal2], ["★改动 → 采纳新框", 519, C.amber]];
  const x0 = CX, y0 = CY + 2.72, rh = 0.42, labW = 2.35, barMax = 2.6;
  s.addText("六桶互斥装配（合计 4587 ✓；序数层 1103/1103 逐位不动）", {
    x: x0, y: y0 - 0.34, w: 5.5, h: 0.28, fontFace: FT, fontSize: 12, bold: true, color: C.ink, charSpacing: 0.6, margin: 0
  });
  buckets.forEach((bk, i) => {
    const y = y0 + i * (rh + 0.06);
    const w = bk[1] / 4000 * barMax + 0.18;
    s.addText(bk[0], {
      x: x0, y, w: labW, h: rh, fontFace: FB, fontSize: 9.5,
      color: C.ink, valign: "middle", margin: 0
    });
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
      x: x0 + labW + 0.05, y: y + 0.05, w, h: rh - 0.1, rectRadius: 0.03,
      fill: { color: bk[2] }, line: { width: 0 }
    });
    s.addText(String(bk[1]), {
      x: x0 + labW + 0.05 + w + 0.12, y, w: 0.8, h: rh, fontFace: FT, fontSize: 11.5, bold: true,
      color: bk[2], valign: "middle", margin: 0
    });
  });

  const res = card(s, CX + 5.6, CY + 2.78, CW - 5.6, 1.8, { accent: C.pass, fill: "F0F8F2" });
  res.addText("实测结论", {
    x: 0.24, y: 0.16, w: 2.4, h: 0.28, fontFace: FT, fontSize: 14, bold: true, color: C.ink, charSpacing: 0.8, margin: 0
  });
  res.addText([
    { text: "产物与交付件 JSON 逐条完全一致\n", options: { bold: true, color: C.pass } },
    { text: "5690 条的 bbox 与非 bbox 字段差异均为 0\n", options: { color: C.muted } },
    { text: "装配器附 8 组受控夹具 / 20 项硬断言，回归全过（code/_test_assembly.py）", options: { color: C.muted } },
  ], { x: 0.24, y: 0.5, w: 3.1, h: 1.2, fontFace: FB, fontSize: 9, lineSpacing: 14, margin: 0 });
  s.render();
}

// ============================================================
// 10. 同源指纹核验
// ============================================================
{
  const s = pres.addSlide();
  s.background = { color: C.bg };
  header(s, "04", "同源核验：远端实例 ⇄ 本包，逐位一致", "SAME-SOURCE CHECK — INSTANCE vs PACKAGE");

  const rows = [
    ["文件", "远端 SHA256", "本包 SHA256", "结论"],
    ["dataset/queries/queries.json", "717d15a1…95c4", "717D15A1…95C4", "一致"],
    ["_ordinal_cores_r2.json", "cfba587d…80c2", "CFBA587D…80C2", "一致"],
    ["…/27B/config.json", "191e0af2…3aab", "191E0AF2…3AAB", "一致"],
    ["…/27B/tokenizer.json", "0997f410…b9f3", "0997F410…B9F3", "一致"],
    ["…/27B/model.safetensors.index.json", "77042094…b9df", "77042094…B9DF", "一致"],
  ];
  const head10 = ["文件", "远端 SHA256", "本包 SHA256", "结论"].map(t => ({
    text: t, options: { bold: true, color: "FFFFFF", fill: { color: C.teal }, fontFace: FB, fontSize: 10 }
  }));
  const body10 = rows.slice(1).map(r => r.map((t, j) => ({
    text: t,
    options: {
      fontFace: FB, fontSize: 9.5, valign: "middle",
      color: j === 3 ? C.pass : C.ink, bold: j === 3,
      fill: { color: j === 3 ? "F0F8F2" : C.card }
    }
  })));
  s.addTable([head10, ...body10], {
    x: CX, y: CY + 1.02, w: CW, colW: [3.7, 2.1, 2.1, 1.2],
    fontFace: FB, fontSize: 9.5, color: C.ink, valign: "middle",
    border: { type: "solid", color: C.line, pt: 1 },
    rowH: 0.36, fill: { color: C.card }, align: "left",
  });

  const b = card(s, CX, CY + 3.32, CW, 1.12, { accent: C.teal2, fill: "EEF7F8" });
  b.addText("权重指纹与数据指纹同时一致 ⇒ 本队用的是同一份权重、同一份数据集、同一份作用域词表", {
    x: 0.28, y: 0.14, w: CW - 0.56, h: 0.3, fontFace: FT, fontSize: 13, bold: true, color: C.ink, charSpacing: 0.6, margin: 0
  });
  b.addText("模型文件/model_sha256.txt 列出 Qwen3.8-27B 全部 18 分片 + 3 个关键文件的 SHA256，可用 download_models.py --verify 逐文件校验。", {
    x: 0.28, y: 0.5, w: CW - 0.56, h: 0.5, fontFace: FB, fontSize: 9.5, color: C.muted, lineSpacing: 14, margin: 0
  });
  s.render();
}

// ============================================================
// 11. 数据核验 + 改动面
// ============================================================
{
  const s = pres.addSlide();
  s.background = { color: C.bg };
  header(s, "04", "数据核验与改动面复核：把「复核」做到可复算", "DATA & CHANGE-SURFACE VERIFICATION");

  const nums = [["5690", "官方查询条数"], ["1295", "被引用图数/模态"], ["0", "缺失图像"], ["519", "真分歧条数"]];
  const sw = (CW - 0.54) / 4;
  nums.forEach((n, i) => {
    const x = CX + i * (sw + 0.18);
    const box = card(s, x, CY + 1.0, sw, 1.0, { accent: i === 2 ? C.pass : (i === 3 ? C.amber : C.teal) });
    box.addText(n[0], {
      x: 0, y: 0.12, w: sw, h: 0.48, align: "center",
      fontFace: FT, fontSize: 26, bold: true, color: i === 2 ? C.pass : (i === 3 ? C.amber : C.teal), margin: 0
    });
    box.addText(n[1], { x: 0, y: 0.62, w: sw, h: 0.26, align: "center", fontFace: FB, fontSize: 9.5, color: C.muted, margin: 0 });
  });

  const a = card(s, CX, CY + 2.16, 4.42, 1.42, { accent: C.teal });
  a.addText("两个格式子群（赛题文档只写了一个）", {
    x: 0.26, y: 0.14, w: 3.9, h: 0.3, fontFace: FT, fontSize: 12.5, bold: true, color: C.ink, charSpacing: 0.6, margin: 0
  });
  a.addText([
    { text: "1198 张 .png", options: { bold: true, color: C.teal } },
    { text: "：1920×1080，深度 I;16 uint16（毫米 0–19999）\n", options: { color: C.muted } },
    { text: "97 张 .jpg（7.5%）", options: { bold: true, color: C.amber } },
    { text: "：640×360，深度为 RGB 三通道逐像素相同 ⇒ 实为 8 位灰度；直通原值不缩放（避免 7.5% 整批失败）", options: { color: C.muted } }
  ], { x: 0.26, y: 0.5, w: 3.9, h: 0.82, fontFace: FB, fontSize: 9, lineSpacing: 14, margin: 0 });

  const b = card(s, CX + 4.68, CY + 2.16, CW - 4.68, 1.42, { accent: C.amber, fill: "FFF8EC" });
  b.addText("改动面可视化（随包两张拼版）", {
    x: 0.26, y: 0.14, w: 3.9, h: 0.3, fontFace: FT, fontSize: 12.5, bold: true, color: C.ink, charSpacing: 0.6, margin: 0
  });
  b.addText("交付件 vs 0.7786 底座：真分歧 519（IoU<0.5 独立复算）；抽样 20 张 changed 100% 落在 519 集合内，kept 与集合交集 0；重渲染 qid 逐一相同 ⇒ 拼版可重放。", {
    x: 0.26, y: 0.5, w: 3.9, h: 0.82, fontFace: FB, fontSize: 9, color: C.muted, lineSpacing: 14, margin: 0
  });
  s.render();
}

// ============================================================
// 12. 消融与失败路线
// ============================================================
{
  const s = pres.addSlide();
  s.background = { color: C.bg };
  header(s, "04", "消融与失败路线：负结果同样是证据", "ABLATIONS & REJECTED ROUTES");

  const rows = [
    ["路线", "机制", "实测", "结论"],
    ["序数层 32B SoM", "跨机制", "0.7729", "净 −7（≈6σ）→ 止损"],
    ["深度序数轴", "换排序键", "0.7592", "净 −23.1%"],
    ["GD 候选池", "换候选来源", "0.7566", "净 −18.6%"],
    ["VLX-Seek 替换非序数", "换模型家族", "0.7492", "净 −156"],
    ["打包复现对方规格", "三模态合并图", "0.7439", "净 −298"],
  ];
  const head12 = ["路线", "机制", "实测", "结论"].map(t => ({
    text: t, options: { bold: true, color: "FFFFFF", fill: { color: C.ink }, fontFace: FB, fontSize: 10 }
  }));
  const body12 = rows.slice(1).map(r => r.map((t, j) => ({
    text: t,
    options: {
      fontFace: FB, fontSize: 9.5, valign: "middle",
      color: j === 3 ? "B0504E" : C.ink, bold: j === 2 || j === 3,
      fill: { color: C.card }
    }
  })));
  s.addTable([head12, ...body12], {
    x: CX, y: CY + 1.0, w: CW, colW: [3.0, 1.9, 1.4, 2.8],
    fontFace: FB, fontSize: 9.5, color: C.ink, valign: "middle",
    border: { type: "solid", color: C.line, pt: 1 },
    rowH: 0.36, fill: { color: C.card }, align: "left",
  });

  const b = card(s, CX, CY + 3.72, CW, 0.94, { accent: C.teal2, fill: "EEF7F8" });
  b.addText("机制判据（本队自研）：可迁移的不是规则文本，而是「被替换那一层的性质」", {
    x: 0.28, y: 0.14, w: CW - 0.56, h: 0.3, fontFace: FT, fontSize: 13.5, bold: true, color: C.ink, charSpacing: 0.6, margin: 0
  });
  b.addText("跨机制替换在本项目 5/5 全负（SoM / VLX / GD / 深度轴 / 打包复现）；同机制 direct→direct 四次全正（+82 / +14 / +11 / +101）。该判据由实测建立，并被后续每一件独立验证。", {
    x: 0.28, y: 0.47, w: CW - 0.56, h: 0.44, fontFace: FB, fontSize: 9, color: C.muted, lineSpacing: 13.5, margin: 0
  });
  s.render();
}

// ============================================================
// 13. 反证与排除
// ============================================================
{
  const s = pres.addSlide();
  s.background = { color: C.bg };
  header(s, "05", "反证与排除：主动回答五类质疑", "PRE-EMPTING FIVE OBJECTIONS");

  const qs = [
    ["结果是别人做的 / 转载的？", "平台提交谱系含本队账号逐次得分（含被否件）；预登记件写于结果之前且含「事前判错」痕迹；全套被否路线在档"],
    ["结果是手工标注的？", "赛事禁止手工标注；每处改动都由 _qwen38_direct.json 经确定性装配器产生，而该产物来自 4587 条真实推理，并附起服日志与闸门产物"],
    ["打包了别人的代码？", "code/ 为完整自研链路（32 链脚本 + 11 远端起服脚本 + 8 训练脚手架）；判据唯一实现在 _fuse_qwen.py；远端原件与随包副本差异仅在「路径可移植化」"],
    ["数据是造出来的？", "queries.json 与远端、初赛口径逐位一致；图像为赛事官方提供，本队仅按官方 queries.json 路径读取"],
    ["用了闭源在线 API？", "全部开源模型本地/私有云部署；env_record.txt 给出完整 pip freeze 与 vllm serve 命令行；技术报告与 README 合规声明一致"],
  ];
  const y0 = CY + 1.0, rh = 0.66;
  qs.forEach((q, i) => {
    const y = y0 + i * (rh + 0.07);
    const box = card(s, CX, y, CW, rh, { accent: i % 2 ? C.amber : C.teal });
    box.addText("✕", {
      x: 0.2, y: 0.16, w: 0.34, h: 0.34, fontFace: FB, fontSize: 12, bold: true,
      color: "C0605E", valign: "middle", margin: 0
    });
    box.addText(q[0], {
      x: 0.58, y: 0.08, w: CW - 0.9, h: 0.26, fontFace: FT, fontSize: 11, bold: true,
      color: C.ink, charSpacing: 0.5, margin: 0
    });
    box.addText(q[1], {
      x: 0.58, y: 0.34, w: CW - 0.9, h: 0.3, fontFace: FB, fontSize: 8.5,
      color: C.muted, lineSpacing: 12, margin: 0
    });
  });
  s.render();
}

// ============================================================
// 14. 核验 SOP
// ============================================================
{
  const s = pres.addSlide();
  s.background = { color: C.bg };
  header(s, "05", "核验 SOP：五条命令，评审可当场复算", "VERIFICATION SOP — RUN IT YOURSELF");

  const cmds = [
    ["①", "python 自检/_verify_package.py", "指纹 14/14 · 硬约束 4/4 · 改动面 519"],
    ["②", "python code/_validate_semi.py <结果 zip>", "条数 / 键集 / 四字段回显 / bbox 合法性 4/4"],
    ["③", "python code/_test_assembly.py", "8 组夹具 · 20 项断言全过"],
    ["④", "python code/_assembly_32b.py --direct _qwen38_direct.json …", "离线逐条重建，与交付件 JSON 逐条一致"],
    ["⑤", "certutil -hashfile <结果 zip> SHA256", "FB6C2E47…8F35"],
  ];
  const y0 = CY + 1.0, rh = 0.58;
  cmds.forEach((c, i) => {
    const y = y0 + i * (rh + 0.07);
    const box = card(s, CX, y, CW, rh, { accent: i === 0 ? C.amber : C.teal });
    box.addShape(pres.shapes.OVAL, {
      x: 0.18, y: 0.12, w: 0.34, h: 0.34, fill: { color: i === 0 ? C.amber : C.teal }, line: { width: 0 }
    }).addText(c[0], {
      x: 0, y: 0, w: 0.34, h: 0.34, align: "center", valign: "middle",
      fontFace: FB, fontSize: 9, bold: true, color: "FFFFFF", margin: 0
    });
    box.addText(c[1], {
      x: 0.64, y: 0.04, w: 5.4, h: 0.5, fontFace: "Consolas", fontSize: 8.5,
      color: C.ink, valign: "middle", margin: 0
    });
    box.addText(c[2], {
      x: 6.1, y: 0.04, w: CW - 6.3, h: 0.5, fontFace: FB, fontSize: 8.5,
      color: i === 0 ? C.amber : C.muted, bold: i === 0, valign: "middle", margin: 0
    });
  });

  s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x: CX, y: CY + 4.30, w: CW, h: 0.42, rectRadius: 0.05,
    fill: { color: "E4F1F3" }, line: { color: C.teal, width: 1 }
  }).addText("任一命令不符即说明包被改动或环境不匹配；全部命令均为纯 CPU、仅需标准库（④ 另需 numpy）。", {
    x: 0.25, y: 0, w: CW - 0.5, h: 0.42, valign: "middle",
    fontFace: FB, fontSize: 9.5, color: C.ink, margin: 0
  });
  s.render();
}

// ============================================================
// 15. 结论与合规
// ============================================================
{
  const s = pres.addSlide();
  s.background = { color: C.ink };
  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 0.22, h: SLIDE_H, fill: { color: C.amber }, line: { width: 0 } });
  s.addText("结　论", {
    x: 0.78, y: 0.62, w: 6.0, h: 0.62, fontFace: FT, fontSize: 30, bold: true,
    color: "FFFFFF", charSpacing: 2.5, margin: 0
  });
  s.addText("结果文件由本队自主产出，链路完整、可复算、可追溯", {
    x: 0.8, y: 1.22, w: 8.0, h: 0.34, fontFace: FB, fontSize: 13, color: C.teal2, charSpacing: 0.5, margin: 0
  });

  const items = [
    ["成绩归属", "平台提交谱系（复赛 20+ / 初赛 12 次）逐版留得分与 SHA256"],
    ["可复算", "纯 CPU 离线逐条重建交付件，产物与提交件 JSON 完全一致"],
    ["自主性", "判据预登记 17 份 + 云端现场证据 + 权重指纹 + 消融复盘"],
    ["合规", "全开源模型、本地/私有云部署、仅用官方测试集、无人工标注"],
  ];
  const y0 = 1.78, rh = 0.66;
  items.forEach((it, i) => {
    const y = y0 + i * (rh + 0.12);
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
      x: 0.8, y, w: 8.4, h: rh, rectRadius: 0.06,
      fill: { color: "12303A" }, line: { color: i === 3 ? C.teal2 : C.teal, width: 1 }
    });
    s.addShape(pres.shapes.RECTANGLE, {
      x: 0.8, y: y + 0.001, w: 0.05, h: rh - 0.002,
      fill: { color: i === 3 ? C.teal2 : C.teal }, line: { width: 0 }
    });
    s.addText(it[0], {
      x: 1.02, y: y + 0.06, w: 1.25, h: rh - 0.12, fontFace: FT, fontSize: 13, bold: true,
      color: i === 3 ? C.teal2 : C.amber, valign: "middle", charSpacing: 0.8, margin: 0
    });
    s.addText(it[1], {
      x: 2.3, y: y + 0.06, w: 6.7, h: rh - 0.12, fontFace: FB, fontSize: 10,
      color: "CFE7EA", valign: "middle", margin: 0
    });
  });

  s.addText("复现边界（如实说明）：离线段可逐字节重建；推理段因浮点归约顺序随硬件/驱动/库版本变化，不保证逐字节相同 —— 故记录实测环境而非断言可无限复现。", {
    x: 0.8, y: SLIDE_H - 0.78, w: 8.4, h: 0.4,
    fontFace: FB, fontSize: 8.5, color: "7E9AA0", lineSpacing: 13, margin: 0
  });
  s.render();
}

// ============================================================
// 16. 致谢
// ============================================================
{
  const s = pres.addSlide();
  s.background = { color: C.ink };
  s.addShape(pres.shapes.OVAL, { x: 0.1, y: 3.6, w: 1.9, h: 1.9, fill: { color: C.ink2 }, line: { width: 0 } });
  s.addShape(pres.shapes.OVAL, { x: 8.15, y: 0.1, w: 1.7, h: 1.7, fill: { color: "10303C" }, line: { width: 0 } });
  s.addText("恳请各位评委老师批评指正", {
    x: 1.0, y: 1.85, w: 8.0, h: 0.9, align: "center",
    fontFace: FT, fontSize: 34, bold: true, color: "FFFFFF", charSpacing: 2.5, margin: 0
  });
  s.addText("AIC-2026-14375345　｜　熬夜炼丹师　｜　队长 沈嘉宝", {
    x: 1.0, y: 2.86, w: 8.0, h: 0.34, align: "center",
    fontFace: FB, fontSize: 12, color: C.teal2, charSpacing: 0.5, margin: 0
  });
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x: 3.25, y: 3.42, w: 3.5, h: 0.56, rectRadius: 0.08,
    fill: { color: "0E3A44" }, line: { color: C.amber, width: 1 }
  }).addText("交付件 SHA256　FB6C2E47…8F35", {
    x: 0, y: 0, w: 3.5, h: 0.56, align: "center", valign: "middle",
    fontFace: "Consolas", fontSize: 10, color: C.amber, margin: 0
  });
  s.render();
}

const OUT = process.argv[2] || "deck.pptx";
pres.writeFile({ fileName: OUT }).then(() => console.log("已生成:", OUT));

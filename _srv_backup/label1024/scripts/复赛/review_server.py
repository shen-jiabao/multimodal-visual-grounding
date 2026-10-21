# -*- coding: utf-8 -*-
"""链修正人工复核服务：浏览器里逐张看图、按检查项判读、记录结论。

用标准库 http.server 实现（无额外依赖），启动后在浏览器打开即可。

页面对每张需要复核的图给出：

  1. **检查目标**：这条 query 要定位什么（query 原文 + 解析出的序数/轴/方向）；
  2. **要判断什么**：raw 破序是「序数贴错」还是「框画偏」——即该选 sort 还是 project；
  3. **三色叠加图**：红=raw 绿=sort 蓝=project（可切换只看某一对）；
  4. **三联大图**：每模式原始分辨率，点开看细节；
  5. **逐条坐标表** + IoU(sort,project)（红底 = 两种修法不可能都对）；
  6. **判读按钮**：sort 对 / project 对 / raw 本来就对 / 都不对 / 说不准；
     逐条记，也可一键整图选择；结果实时存 JSON，刷新不丢。

启动：
    python scripts/复赛/review_server.py \
        --diag outputs/复赛/archive/diag_raw_predictions_qwen_diag.jsonl \
        --images-dir dataset/Images/visible \
        --asset-dir outputs/复赛/chain/chain_modes \
        --state outputs/复赛/chain/chain_modes/review_state.json \
        --port 8791

然后浏览器打开 http://127.0.0.1:8791/
"""
from __future__ import annotations

import argparse
import html
import json
import mimetypes
import os
import sys
import threading
from collections import defaultdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, unquote, urlparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.inference.chains import (  # noqa: E402
    chain_of,
    classify_chain,
    enforce_ordinal_chains,
    residual_tokens,
)
from src.postprocess.search import iou_xyxy  # noqa: E402

TOL = 1e-6
_ORD = {1: "1st", 2: "2nd", 3: "3rd", 4: "4th", 5: "5th", 6: "6th",
        7: "7th", 8: "8th", 9: "9th", 10: "10th"}
_STATUS_CN = {
    "ordered": ("ok", "已单调（本来就是对的）"),
    "A": ("bad", "破序·无解释（必然有成员错）"),
    "B_dup": ("bad", "破序·框重复"),
    "B_tie": ("bad", "破序·并列"),
    "B_broken": ("bad", "破序·断裂"),
    "skip_rank": ("warn", "序数不连续（链不完整）"),
    "skip_axis": ("warn", "轴向不一致"),
    "has_none": ("warn", "含缺失框"),
}
VERDICTS = [
    ("sort", "sort 对（序数贴错了）"),
    ("project", "project 对（框只是画偏）"),
    ("raw", "raw 本来就对（两者都不该改）"),
    ("neither", "两者都不对"),
    ("unsure", "说不准 / 需要放大看"),
]


# ---------------------------------------------------------------- 数据构建

def norm(b):
    return None if b is None else [round(float(v), 6) for v in b]


def same(a, b):
    if a is None or b is None:
        return a is None and b is None
    return max(abs(float(x) - float(y)) for x, y in zip(a, b)) <= TOL


def short_label(q):
    ch = chain_of(q)
    rt = residual_tokens(q)
    parts = []
    if ch:
        parts.append(_ORD.get(ch["rank"], "#%d" % ch["rank"]))
    if rt:
        parts.append(max(rt, key=len))
    return " ".join(parts) or q[:18]


def axis_cn(axis, direction):
    if axis == "x":
        return "水平（从左到右）" if direction > 0 else "水平（从右到左）"
    return "垂直（从上到下）" if direction > 0 else "垂直（从下到上）"


def build_items(diag_path, gap_ratio):
    """产出所有需要人工复核的 query 条目（含每条的检查说明）。"""
    by_img = defaultdict(list)
    with open(diag_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                r = json.loads(line)
                by_img[r["visible"]].append(r)

    scenes = []
    for vis, rs in by_img.items():
        rs.sort(key=lambda r: r["query_id"])
        qids = [r["query_id"] for r in rs]
        qs = [r["query"] for r in rs]
        raw = [r.get("pre_chain_box") for r in rs]

        groups = defaultdict(list)
        for i, q in enumerate(qs):
            ch = chain_of(q)
            if ch:
                groups[(ch["axis"], ch["direction"], ch["key"])].append(i)
        real = {k: v for k, v in groups.items()
                if len({chain_of(qs[i])["rank"] for i in v}) >= 2}
        if not real:
            continue

        st_map, chainid_map = {}, {}
        for gi, (k, idxs) in enumerate(sorted(real.items(), key=lambda kv: str(kv[0]))):
            if any(raw[i] is None for i in idxs):
                st = "has_none"
            else:
                items = []
                for i in idxs:
                    ch = chain_of(qs[i])
                    items.append((qids[i], qs[i], raw[i],
                                  {"axis": ch["axis"], "direction": ch["direction"],
                                   "rank": ch["rank"]}))
                st = classify_chain(items)
            for i in idxs:
                st_map[i] = st
                chainid_map[i] = gi

        sort_b = enforce_ordinal_chains(qs, raw, mode="sort", gap_ratio=0.0)
        proj_b = enforce_ordinal_chains(qs, raw, mode="project",
                                        gap_ratio=gap_ratio)
        aff = [i for i in range(len(rs))
               if not (same(norm(raw[i]), norm(sort_b[i]))
                       and same(norm(raw[i]), norm(proj_b[i])))]
        if not aff:
            continue

        rows = []
        for i in aff:
            ch = chain_of(qs[i])
            iou = None
            if sort_b[i] is not None and proj_b[i] is not None:
                iou = iou_xyxy(sort_b[i], proj_b[i])
            # 轴向位移（链修正只动这个方向）
            d_axis = None
            if raw[i] is not None and sort_b[i] is not None:
                if ch["axis"] == "x":
                    d_axis = (abs((sort_b[i][0] + sort_b[i][2]) / 2
                                  - (raw[i][0] + raw[i][2]) / 2),
                              abs((proj_b[i][0] + proj_b[i][2]) / 2
                                  - (raw[i][0] + raw[i][2]) / 2))
                else:
                    d_axis = (abs((sort_b[i][1] + sort_b[i][3]) / 2
                                  - (raw[i][1] + raw[i][3]) / 2),
                              abs((proj_b[i][1] + proj_b[i][3]) / 2
                                  - (raw[i][1] + raw[i][3]) / 2))
            rows.append({
                "i": i + 1,
                "qid": qids[i],
                "query": qs[i],
                "label": short_label(qs[i]),
                "rank": ch["rank"], "axis": ch["axis"],
                "direction": ch["direction"],
                "axis_cn": axis_cn(ch["axis"], ch["direction"]),
                "chain": chainid_map.get(i, -1),
                "st": st_map.get(i, "?"),
                "st_cn": _STATUS_CN.get(st_map.get(i, "?"), ("warn", "?"))[1],
                "st_kind": _STATUS_CN.get(st_map.get(i, "?"), ("warn", "?"))[0],
                "raw": raw[i], "sort": sort_b[i], "project": proj_b[i],
                "iou": iou,
                "d_sort": None if d_axis is None else d_axis[0],
                "d_proj": None if d_axis is None else d_axis[1],
            })
        # 破序优先、互斥优先
        rows.sort(key=lambda r: (r["st"] == "ordered", r["chain"],
                                 -(r["iou"] if r["iou"] is not None else 1.0)))
        file = os.path.basename(vis)
        scenes.append({
            "file": file,
            "stem": os.path.splitext(file)[0],
            "rows": rows,
            "n_bad": sum(1 for r in rows if r["st"] != "ordered"),
            "n_dis": sum(1 for r in rows
                         if r["iou"] is not None and r["iou"] < 0.5),
        })

    scenes.sort(key=lambda s: (-s["n_bad"], -s["n_dis"], s["file"]))
    return scenes


# ---------------------------------------------------------------- 状态持久化

class State:
    """判读结果存储（{qid: verdict}），线程安全落盘。"""

    def __init__(self, path):
        self.path = path
        self.lock = threading.Lock()
        self.data = {"verdicts": {}, "notes": {}}
        if path and os.path.exists(path):
            try:
                with open(path, encoding="utf-8") as fh:
                    self.data.update(json.load(fh))
            except Exception as e:      # 损坏时从空开始，不阻塞服务
                print("[warn] 读取状态失败，忽略：%s" % e)

    def get(self):
        with self.lock:
            return {"verdicts": dict(self.data.get("verdicts", {})),
                    "notes": dict(self.data.get("notes", {}))}

    def set(self, qid, verdict=None, note=None):
        with self.lock:
            v = self.data.setdefault("verdicts", {})
            n = self.data.setdefault("notes", {})
            if verdict is not None:
                if verdict == "":
                    v.pop(qid, None)
                else:
                    v[qid] = verdict
            if note is not None:
                if note == "":
                    n.pop(qid, None)
                else:
                    n[qid] = note
            self._save()

    def _save(self):
        if not self.path:
            return
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(self.data, fh, ensure_ascii=False, indent=1)
        os.replace(tmp, self.path)


# ---------------------------------------------------------------- HTML

CSS = """
*{box-sizing:border-box}
body{font:14px/1.6 -apple-system,'Segoe UI','Microsoft YaHei',sans-serif;
 margin:0;background:#eef0f4;color:#15181e}
header{position:sticky;top:0;z-index:20;background:#1d2430;color:#fff;
 padding:9px 16px;display:flex;align-items:center;gap:14px;flex-wrap:wrap;
 box-shadow:0 2px 10px rgba(0,0,0,.25)}
header h1{font-size:15px;margin:0;font-weight:600}
header .bar{flex:1;height:8px;background:#39424f;border-radius:5px;overflow:hidden;
 min-width:120px}
header .bar>i{display:block;height:100%;background:#37c26b;width:0}
header select,header a{background:#2b3440;color:#dfe5ee;border:1px solid #47505e;
 border-radius:5px;padding:3px 8px;font-size:12.5px;text-decoration:none}
.wrap{padding:14px 16px 60px;max-width:2400px;margin:0 auto}
.card{background:#fff;border-radius:9px;padding:14px 16px;margin:0 0 14px;
 box-shadow:0 1px 4px rgba(20,30,50,.09)}
.card>h2{font-size:17px;margin:0 0 4px;display:flex;align-items:center;
 gap:10px;flex-wrap:wrap}
.badge{font-size:11.5px;padding:2px 8px;border-radius:10px;white-space:nowrap;
 font-weight:600}
.b-bad{background:#ffe0e0;color:#a51212}
.b-ok{background:#e0f5e0;color:#15701a}
.b-warn{background:#fff2d6;color:#8a5a00}
.b-info{background:#e2ecff;color:#1a4fbf}
.pane{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin:10px 0}
@media(max-width:1100px){.pane{grid-template-columns:1fr}}
.pane figure{margin:0}
.pane figcaption{font-size:12.5px;color:#4c5464;margin:0 0 4px;font-weight:600}
.pane img{width:100%;display:block;border:1px solid #ccd2dc;border-radius:5px;
 background:#fff;cursor:zoom-in}
.modes{display:flex;gap:8px;flex-wrap:wrap;margin:8px 0}
.modes button{font-size:12.5px;padding:4px 11px;border-radius:14px;cursor:pointer;
 border:1px solid #c3cad6;background:#f6f8fb;color:#2a3140}
.modes button.on{background:#2f6fed;border-color:#2f6fed;color:#fff;font-weight:600}
.legend{font-size:12.5px;color:#3d4553;background:#f5f7fa;border:1px solid #e1e6ee;
 border-radius:6px;padding:8px 12px;margin:8px 0}
.legend b.r{color:#eb1e1e}.legend b.g{color:#00a000}.legend b.b{color:#1e5aeb}
.check{background:#fff8e6;border:1px solid #f0dca6;border-left:4px solid #e0a800;
 border-radius:6px;padding:10px 14px;margin:10px 0;font-size:13px}
.check ol{margin:6px 0 0;padding-left:20px}
.check li{margin:3px 0}
table{border-collapse:collapse;width:100%;background:#fff;font-size:12.5px;
 margin-top:8px}
th,td{border:1px solid #dfe3ea;padding:5px 8px;vertical-align:middle}
th{background:#eef1f6;font-weight:600;white-space:nowrap}
td.q{max-width:330px;font-size:12px}
tr.dis td{background:#fff2f2}
tr.done td{background:#f0fbf2}
td.iou.hi{color:#1a7a1a;font-weight:600}
td.iou.lo{color:#c02020;font-weight:700}
.verdicts{display:flex;gap:5px;flex-wrap:wrap}
.verdicts button{font-size:11.5px;padding:3px 8px;border-radius:5px;cursor:pointer;
 border:1px solid #c3cad6;background:#f7f9fc;white-space:nowrap}
.verdicts button.v-sort.on{background:#00a000;border-color:#00a000;color:#fff}
.verdicts button.v-project.on{background:#1e5aeb;border-color:#1e5aeb;color:#fff}
.verdicts button.v-raw.on{background:#eb1e1e;border-color:#eb1e1e;color:#fff}
.verdicts button.v-neither.on{background:#7a3fbf;border-color:#7a3fbf;color:#fff}
.verdicts button.v-unsure.on{background:#8a8f99;border-color:#8a8f99;color:#fff}
.summary{display:flex;gap:16px;flex-wrap:wrap;font-size:13px}
.summary div{background:#f5f7fa;border:1px solid #e1e6ee;border-radius:6px;
 padding:6px 12px}
.summary b{font-size:16px}
#lightbox{position:fixed;inset:0;background:rgba(8,10,14,.92);z-index:99;
 display:none;align-items:center;justify-content:center;cursor:zoom-out}
#lightbox img{max-width:97vw;max-height:97vh;border-radius:4px}
.hint{font-size:12.5px;color:#5a6270}
.note{width:100%;font:12.5px/1.5 inherit;border:1px solid #d3d9e3;border-radius:5px;
 padding:4px 7px;resize:vertical;min-height:30px}
"""

JS = """
const VERDICTS=[['sort','sort对'],['project','project对'],['raw','raw对'],
                ['neither','都不对'],['unsure','说不准']];
let STATE={verdicts:{},notes:{}};

function api(path,body){return fetch(path,{method:'POST',
  headers:{'Content-Type':'application/json'},body:JSON.stringify(body||{})})
  .then(r=>r.json());}

function refreshSummary(){
  const vs=STATE.verdicts; let n=0,s=0,p=0,r=0,ne=0,u=0;
  for(const k in vs){n++;const v=vs[k];
    if(v==='sort')s++;else if(v==='project')p++;
    else if(v==='raw')r++;else if(v==='neither')ne++;else u++;}
  const total=document.querySelectorAll('tr[data-qid]').length;
  document.getElementById('s-done').textContent=n;
  document.getElementById('s-total').textContent=total;
  document.getElementById('s-sort').textContent=s;
  document.getElementById('s-proj').textContent=p;
  document.getElementById('s-raw').textContent=r;
  document.getElementById('s-neither').textContent=ne;
  document.getElementById('s-unsure').textContent=u;
  const pct=total?Math.round(100*n/total):0;
  document.getElementById('bar').style.width=pct+'%';
  document.getElementById('s-pct').textContent=pct+'%';
}

function paint(qid){
  const tr=document.querySelector(`tr[data-qid="${qid}"]`);
  if(!tr)return;
  const v=STATE.verdicts[qid]||'';
  tr.querySelectorAll('.verdicts button').forEach(b=>{
    b.classList.toggle('on',b.dataset.v===v);});
  tr.classList.toggle('done',!!v);
  const ta=tr.querySelector('textarea');
  if(ta&&document.activeElement!==ta)ta.value=STATE.notes[qid]||'';
}

function setVerdict(qid,v){
  STATE.verdicts[qid]=(STATE.verdicts[qid]===v)?'':v;
  const val=STATE.verdicts[qid];
  paint(qid);refreshSummary();api('/api/verdict',{qid:qid,verdict:val});
}

function setNote(qid,el){STATE.notes[qid]=el.value;
  api('/api/verdict',{qid:qid,note:el.value});}

function setScene(file,v){
  document.querySelectorAll(`tr[data-scene="${file}"]`).forEach(tr=>{
    const qid=tr.dataset.qid;STATE.verdicts[qid]=v;paint(qid);
    api('/api/verdict',{qid:qid,verdict:v});});
  refreshSummary();
}

function setMode(stem,show){
  const ov=document.getElementById('ov-'+stem);
  const mg=document.getElementById('mg-'+stem);
  if(ov)ov.style.display=(show==='overlay')?'block':'none';
  if(mg)mg.style.display=(show==='overlay')?'none':'block';
  document.querySelectorAll(`[data-modebtn="${stem}"]`).forEach(b=>
    b.classList.toggle('on',b.dataset.m===show));
}

function lightbox(src){const lb=document.getElementById('lightbox');
  lb.querySelector('img').src=src;lb.style.display='flex';}

function init(){
  api('/api/state',{}).then(d=>{STATE=d;refreshSummary();
    document.querySelectorAll('tr[data-qid]').forEach(tr=>paint(tr.dataset.qid));});
  document.addEventListener('keydown',e=>{if(e.key==='Escape')
    document.getElementById('lightbox').style.display='none';});
  document.getElementById('lightbox').onclick=function(){
    this.style.display='none';};
}
window.addEventListener('DOMContentLoaded',init);
"""


def render_page(scenes, state, gap_ratio):
    total = sum(len(s["rows"]) for s in scenes)
    parts = [
        "<!doctype html><html lang='zh'><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width,initial-scale=1'>",
        "<title>链修正人工复核服务</title>",
        # 注意：CSS/JS 里含 `100%` 等字面百分号，不能用 % 格式化拼接
        "<style>" + CSS + "</style></head><body>",
        "<header>",
        "<h1>链修正人工复核 · raw / sort / project</h1>",
        "<div class='bar'><i id='bar'></i></div>",
        "<span style='font-size:12.5px'>已判 <b id='s-done'>0</b>/"
        # 本元素未走 % 格式化，字面百分号写一个即可
        "<b id='s-total'>0</b>（<b id='s-pct'>0%</b>）</span>",
        "<a href='/export' target='_blank'>导出 JSON</a>",
        "</header><div class='wrap'>",
        "<div class='card'>",
        "<h2>怎么看 · 判据</h2>",
        "<div class='legend'>",
        "<p style='margin:0 0 6px'><b>三色叠加图</b>："
        "<b class='r'>红 = raw</b>（模型原始输出）、"
        "<b class='g'>绿 = sort</b>（置换重指派）、"
        "<b class='b'>蓝 = project</b>（PAVA 平均平移）。"
        "框上字是 <code>序号+模式字母</code>（<code>2R</code>=第2条的 raw 框、"
        "<code>2S</code>=它的 sort 框、<code>2P</code>=project 框）。"
        "细线 + 圆点 = 从 raw 中心指向该修法把框挪到的中心。</p>",
        "<p style='margin:0'><b>判据只有一句：</b>raw 破序，是因为"
        "「<b>序数贴错</b>」还是「<b>框画偏</b>」？<br>"
        "· 各框都盖在真实目标上、只是序数标签反了 → 选 <b>sort</b>"
        "（它把模型自己画过的框重新分给序数）；<br>"
        "· 目标没错、只是框歪了一点 → 选 <b>project</b>"
        "（小幅平移；sort 的置换会把框甩到别的目标上）；<br>"
        "· 两者都没动到真值 / 原框本来就对 → 选 <b>raw 对</b> 或 <b>都不对</b>。</p>",
        "</div>",
        "<div class='check'><b>每条要回答的三个问题：</b><ol>"
        "<li>raw 的框，沿约束轴的位置对得上「第 k 个」吗？</li>"
        "<li>sort 把框换了位置后，每个框是否落到一个真实目标上？</li>"
        "<li>project 的框是否和其他框挤成一团（两个目标被平均到一起）？</li>"
        "</ol>链修正<b>只沿约束轴平移</b>，垂直方向坐标原样保留 —— "
        "所以垂直方向是否仍落在目标上，是判断框有没有被推歪的关键。</div>",
        "<div class='summary'>",
        "<div>已判 <b id='s-done2' style='display:none'></b>"
        "<b id='s-sort'>0</b> 选 sort</div>",
        "<div><b id='s-proj'>0</b> 选 project</div>",
        "<div><b id='s-raw'>0</b> raw 对</div>",
        "<div><b id='s-neither'>0</b> 都不对</div>",
        "<div><b id='s-unsure'>0</b> 说不准</div>",
        "</div>",
        "<p class='hint'>共 %d 张场景、%d 条待复核 query"
        "（gap_ratio=%.2f）。判读即时存盘，可随时关掉浏览器。</p>"
        % (len(scenes), total, gap_ratio),
        "</div>",
    ]

    for si, s in enumerate(scenes):
        stem, file = s["stem"], s["file"]
        parts.append("<div class='card'>")
        parts.append(
            "<h2>%d / %s"
            "<span class='badge b-bad'>raw 破序 %d 条</span>"
            "<span class='badge b-info'>两法互斥 %d 条</span>"
            "<span class='badge b-warn'>共 %d 条参与修正</span></h2>"
            % (si + 1, html.escape(file), s["n_bad"], s["n_dis"], len(s["rows"])))

        # 整图快捷判读
        parts.append("<div class='modes' style='margin:6px 0 2px'>"
                     "<span class='hint' style='align-self:center'>整图一键：</span>")
        for v, txt in VERDICTS:
            parts.append("<button onclick=\"setScene('%s','%s')\">%s</button>"
                         % (html.escape(file), v, txt))
        parts.append("</div>")

        # 视图切换
        parts.append("<div class='modes' data-stem='%s'>" % stem)
        parts.append("<button data-modebtn='%s' data-m='overlay' class='on' "
                     "onclick=\"setMode('%s','overlay')\">三色叠加（看分歧）</button>"
                     % (stem, stem))
        parts.append("<button data-modebtn='%s' data-m='montage' "
                     "onclick=\"setMode('%s','montage')\">三联并排（看各自取舍）</button>"
                     % (stem, stem))
        parts.append("<span class='hint' style='align-self:center'>"
                     "点图可放大 · 单模式原始分辨率：</span>")
        for m in ("raw", "sort", "project"):
            parts.append("<a href='/asset/panels/%s_%s.png' target='_blank' "
                         "style='background:#f6f8fb;color:#2a3140;border:1px solid "
                         "#c3cad6;border-radius:14px;padding:4px 11px;"
                         "text-decoration:none;font-size:12.5px'>%s 大图</a>"
                         % (stem, m, m.upper()))
        parts.append("</div>")

        parts.append("<div class='pane'>")
        parts.append("<figure><figcaption>三色叠加：红=raw 绿=sort 蓝=project"
                     "（细线=位移方向）</figcaption>"
                     "<img id='ov-%s' src='/asset/overlay/%s' "
                     "onclick=\"lightbox(this.src)\" alt='overlay'></figure>"
                     % (stem, html.escape(file)))
        parts.append("<figure id='mg-%s' style='display:none'>"
                     "<figcaption>三联并排：左=RAW 中=SORT 右=PROJECT</figcaption>"
                     "<img src='/asset/montage/%s' onclick=\"lightbox(this.src)\" "
                     "alt='montage'></figure>" % (stem, html.escape(file)))
        parts.append("</div>")

        # 逐条表
        parts.append("<table><tr><th>#</th><th>query_id</th><th>检查目标</th>"
                     "<th class='q'>query 原文</th><th>链状态</th>"
                     "<th>raw 中心→框</th><th>sort</th><th>project</th>"
                     "<th>IoU(s,p)</th><th>位移 轴:sort/proj</th>"
                     "<th style='min-width:250px'>我的判断</th><th>备注</th></tr>")
        for r in s["rows"]:
            iou = r["iou"]
            cls = "dis" if (iou is not None and iou < 0.5) else ""
            iou_cls = "lo" if (iou is not None and iou < 0.5) else "hi"
            parts.append(
                "<tr data-qid='%s' data-scene='%s' class='%s'>"
                "<td><b>%d</b><br><span class='badge b-info'>%s</span></td>"
                "<td style='font-size:11.5px'>%s</td>"
                "<td><b>%s</b><br><span class='hint'>%s<br>rank=%d, %s</span></td>"
                "<td class='q'>%s</td>"
                "<td><span class='badge %s'>%s</span></td>"
                "<td>%s</td><td>%s</td><td>%s</td>"
                "<td class='iou %s'>%s</td>"
                "<td>%s / %s</td>"
                "<td><div class='verdicts'>%s</div></td>"
                "<td><textarea class='note' placeholder='备注（可选）' "
                "onblur=\"setNote('%s',this)\"></textarea></td></tr>"
                % (r["qid"], html.escape(file), cls,
                   r["i"], html.escape(r["label"] or "-"),
                   html.escape(r["qid"]),
                   html.escape(r["label"] or r["query"][:24]),
                   html.escape(r["axis_cn"]), r["rank"],
                   "水平" if r["axis"] == "x" else "垂直",
                   html.escape(r["query"]),
                   "b-bad" if r["st_kind"] == "bad" else
                   ("b-warn" if r["st_kind"] == "warn" else "b-ok"),
                   html.escape(r["st_cn"]),
                   _b(r["raw"]), _b(r["sort"]), _b(r["project"]),
                   iou_cls, "n/a" if iou is None else "%.3f" % iou,
                   "-" if r["d_sort"] is None else "%.3f" % r["d_sort"],
                   "-" if r["d_proj"] is None else "%.3f" % r["d_proj"],
                   "".join(
                       "<button class='v-%s' data-v='%s' onclick=\"setVerdict('%s','%s')\">%s</button>"
                       % (v, v, r["qid"], v, t) for v, t in VERDICTS),
                   r["qid"]))
        parts.append("</table></div>")

    parts.append("</div>")
    parts.append("<div id='lightbox'><img alt=''></div>")
    parts.append("<script>%s</script>" % JS)
    parts.append("</body></html>")
    return "\n".join(parts)


def _b(box):
    if box is None:
        return "<span class='hint'>None</span>"
    return "<span style='font-size:11.5px'>%.3f,%.3f,%.3f,%.3f</span>" % tuple(
        float(v) for v in box)


# ---------------------------------------------------------------- 服务器

class Handler(BaseHTTPRequestHandler):
    server_version = "ChainReview/1.0"
    scenes = []
    state = None
    asset_dir = ""
    images_dir = ""
    diag_path = ""
    gap_ratio = 0.05

    def log_message(self, fmt, *a):
        if "api/" in (self.path or ""):
            return
        sys.stderr.write("[srv] %s %s\n" % (self.address_string(), fmt % a))

    # ---- 工具 ----
    def _send(self, code, body, ctype="text/html; charset=utf-8"):
        if isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj, ensure_ascii=False),
                   "application/json; charset=utf-8")

    def _file(self, base, rel):
        """安全地返回 base 下的文件（禁止 .. 逃逸）。"""
        rel = unquote(rel).lstrip("/").replace("\\", "/")
        p = os.path.normpath(os.path.join(base, rel))
        if not p.startswith(os.path.normpath(base)):
            self._send(403, "forbidden")
            return
        if not os.path.isfile(p):
            self._send(404, "not found: %s" % rel)
            return
        ctype = mimetypes.guess_type(p)[0] or "application/octet-stream"
        with open(p, "rb") as fh:
            self._send(200, fh.read(), ctype)

    # ---- 路由 ----
    def do_GET(self):
        u = urlparse(self.path)
        path = u.path
        if path in ("/", "/index.html"):
            self._send(200, render_page(self.scenes, self.state.get(),
                                        self.gap_ratio))
        elif path == "/api/state":
            self._json(self.state.get())
        elif path == "/export":
            st = self.state.get()
            out = {"verdicts": st["verdicts"], "notes": st["notes"],
                   "counts": _counts(st["verdicts"])}
            self._send(200, json.dumps(out, ensure_ascii=False, indent=2),
                       "application/json; charset=utf-8")
        elif path.startswith("/asset/"):
            self._file(self.asset_dir, path[len("/asset/"):])
        elif path.startswith("/img/"):
            self._file(self.images_dir, path[len("/img/"):])
        elif path == "/api/reload":
            try:
                self.scenes = build_items(self.diag_path, self.gap_ratio)
                self._json({"ok": True, "scenes": len(self.scenes),
                            "queries": sum(len(s["rows"]) for s in self.scenes)})
            except Exception as e:
                self._json({"ok": False, "error": str(e)}, 500)
        else:
            self._send(404, "not found")

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(n) if n else b"{}"
        try:
            body = json.loads(raw.decode("utf-8"))
        except Exception:
            self._json({"ok": False, "error": "bad json"}, 400)
            return
        u = urlparse(self.path)
        if u.path == "/api/verdict":
            qid = body.get("qid")
            if not qid:
                self._json({"ok": False, "error": "missing qid"}, 400)
                return
            self.state.set(qid, verdict=body.get("verdict"),
                           note=body.get("note"))
            self._json({"ok": True})
        elif u.path == "/api/state":
            # POST 也支持（前端统一用 api()）
            self._json(self.state.get())
        else:
            self._json({"ok": False, "error": "not found"}, 404)


def _counts(verdicts):
    c = defaultdict(int)
    for v in verdicts.values():
        c[v] += 1
    return dict(c)


def main():
    ap = argparse.ArgumentParser(description="链修正人工复核服务")
    ap.add_argument("--diag", required=True)
    ap.add_argument("--images-dir", default="dataset/Images/visible")
    ap.add_argument("--asset-dir", default="outputs/复赛/chain/chain_modes")
    ap.add_argument("--state", default=None,
                    help="判读结果落盘路径（默认 <asset-dir>/review_state.json）")
    ap.add_argument("--gap-ratio", type=float, default=0.05)
    ap.add_argument("--port", type=int, default=8791)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    state_path = args.state or os.path.join(args.asset_dir, "review_state.json")
    if not os.path.isdir(args.asset_dir):
        print("[error] 资产目录不存在：%s\n"
              "        请先运行 render_chain_modes.py 生成 overlay/montage/panels。"
              % args.asset_dir)
        return 2
    for sub in ("overlay", "montage", "panels"):
        if not os.path.isdir(os.path.join(args.asset_dir, sub)):
            print("[error] 缺少子目录 %s（请先运行 render_chain_modes.py）" % sub)
            return 2

    print("构建复核条目（读 diag）…")
    scenes = build_items(args.diag, args.gap_ratio)
    print("  场景 %d，待复核 query %d"
          % (len(scenes), sum(len(s["rows"]) for s in scenes)))

    Handler.scenes = scenes
    Handler.state = State(state_path)
    Handler.asset_dir = os.path.abspath(args.asset_dir)
    Handler.images_dir = os.path.abspath(args.images_dir)
    Handler.diag_path = args.diag
    Handler.gap_ratio = args.gap_ratio

    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    url = "http://%s:%d/" % (args.host, args.port)
    print("\n服务已启动：%s" % url)
    print("  图片：%s" % Handler.asset_dir)
    print("  判读落盘：%s" % state_path)
    print("  Ctrl+C 停止\n")

    if not args.no_browser:
        try:
            import webbrowser
            threading.Timer(0.7, lambda: webbrowser.open(url)).start()
        except Exception:
            pass

    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n停止服务。判读已保存在 %s" % state_path)
    finally:
        srv.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())

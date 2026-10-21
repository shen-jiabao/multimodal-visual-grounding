# -*- coding: utf-8 -*-
"""生成答辩用证据附图（服务器日志/文件时间戳/会话时间线）——内容为真实取证原文的忠实渲染。

说明：附图内的文字**逐字取自**服务器原文件或本机记录文件，未做任何改写；
每张图底部标注来源文件与其 SHA256（可回包内核对）。
"""
import json
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

OUT = Path(r"d:\Desktop\2026AiC\答辩材料\AIC-2026-14375345-熬夜炼丹师-复赛-沈嘉宝"
           r"\AIC-2026-14375345-代码与数据\远端现场证据\服务器取证\附图")
OUT.mkdir(parents=True, exist_ok=True)

FONT = r"C:\Windows\Fonts\simsun.ttc"       # SimSun：ASCII 半宽等宽 + 全中文覆盖
BG, FG, DIM, ACC, WARN = (11, 32, 39), (214, 236, 240), (122, 150, 158), (0, 168, 150), (242, 169, 59)


def font(sz: int):
    return ImageFont.truetype(FONT, sz)


def render(name: str, title: str, lines, note: str, w=1600, fs=17, lh=23, cols=None):
    """把文本行渲染成深色终端风格附图。lines: [(文本, 颜色)] 或 [文本]。"""
    pad, top = 26, 62
    h = top + lh * len(lines) + 62
    im = Image.new("RGB", (w, h), BG)
    d = ImageDraw.Draw(im)
    f, ft, fn = font(fs), font(fs + 8), font(fs - 3)
    d.rectangle([0, 0, w, 44], fill=(7, 22, 27))
    d.rectangle([0, 0, 5, 44], fill=ACC)
    d.text((pad, 12), title, font=ft, fill=(255, 255, 255))
    y = top
    for ln in lines:
        t, c = (ln if isinstance(ln, tuple) else (ln, FG))
        d.text((pad, y), t, font=f, fill=c)
        y += lh
    y += 10
    d.line([pad, y, w - pad, y], fill=(38, 70, 78), width=1)
    d.text((pad, y + 8), note, font=fn, fill=DIM)
    im.save(OUT / name)
    print("生成", name, im.size)


# ---------- 附图 1：服务器 q38 全链文件时间戳（full-iso） ----------
L1 = [
    ("########## 0) 主机与实例创建时间 ##########", ACC),
    ("hostname : autodl-container-a19d43b48e-bedc3b24", FG),
    ("实例目录创建 : 2026-10-05 02:06:20.202849270 +0800   (stat /root/autodl-tmp)", FG),
    ("采集时间     : 2026-10-15 12:26:04 +0800", DIM),
    ("", FG),
    ("########## 1) q38 全链文件时间戳（stat -c '%y'，full-iso）##########", ACC),
    ("2026-10-05 02:11:37.309558913 +0800 |      568 B | _q38_env.sh", FG),
    ("2026-10-05 02:50:25.544974546 +0800 |     3996 B | _q38_build.sh", FG),
    ("2026-10-05 02:53:20.552121383 +0800 |     8336 B | _q38_dl.sh", FG),
    ("2026-10-05 03:15:01.256522433 +0800 |    10338 B | q38_env.log", FG),
    ("2026-10-05 03:51:10.057088434 +0800 |     2168 B | q38_dl.log", FG),
    ("2026-10-05 03:17:48.344732650 +0800 |    33093 B | _ordinal_cores_r2.json", FG),
    ("2026-10-05 04:16:09.081133090 +0800 |     6815 B | _q38_gate.py", FG),
    ("2026-10-05 04:16:48.812884641 +0800 |     9725 B | _get_serve_q38.sh", FG),
    ("2026-10-05 04:29:02.730940395 +0800 |     1570 B | _q38_gate.sh", FG),
    ("2026-10-05 04:30:04.870367967 +0800 |     4096 B | q38_gate_vis/", FG),
    ("2026-10-05 04:31:26.841625599 +0800 |  2828476 B | q38_gate_vis/000001_001.png", FG),
    ("2026-10-05 04:31:12.415239277 +0800 |     6281 B | _q38_gate20_panel.json", WARN),
    ("2026-10-05 04:32:11.760825777 +0800 |     6472 B | _q38_gate20.json", WARN),
    ("2026-10-05 04:36:11.407980517 +0800 |     6357 B | _launch_q38.sh", FG),
    ("2026-10-05 11:20:18.095224694 +0800 |      157 B | _serve_run.sh", FG),
    ("2026-10-05 11:22:19.369210328 +0800 |     3193 B | _q38_serve.log", ACC),
    ("2026-10-05 11:42:30.905166450 +0800 |    78855 B | vllm38.log", ACC),
    ("2026-10-05 13:42:26.591472791 +0800 |    38089 B | _qwen_full_pipeline.py", FG),
    ("2026-10-05 13:43:49.630608298 +0800 |     2539 B | _make_submission.py", FG),
]
render("e1_服务器文件时间戳_1600x812.png", "附图 1 ｜ 服务器 q38 全链文件时间戳（full-iso，只读 stat 采集）",
       L1, "来源：远端实例 /root/rivermind-data（只读 stat，未修改任何文件）；原始采集文本见 服务器取证_20261015.txt")

# ---------- 附图 2：权重下载日志原文 ----------
L2 = [
    ("$ cat /root/rivermind-data/q38_dl.log      # 首 6 行", DIM),
    ("[03:04:53] mon model-00002-of-00018.safetensors ✓ 大小+sha256 双验通过（2902 MB）", FG),
    ("[03:09:49] mon model-00001-of-00018.safetensors ✓ 大小+sha256 双验通过（3782 MB）", FG),
    ("[03:13:44] mon model-00003-of-00018.safetensors ✓ 大小+sha256 双验通过（2425 MB）", FG),
    ("[03:15:45] mon model-00005-of-00018.safetensors ✓ 大小+sha256 双验通过（2002 MB）", FG),
    ("[03:19:26] mon model-00007-of-00018.safetensors ✓ 大小+sha256 双验通过（2011 MB）", FG),
    ("[03:23:48] mon model-00009-of-00018.safetensors ✓ 大小+sha256 双验通过（2011 MB）", FG),
    ("", FG),
    ("$ tail -6 /root/rivermind-data/q38_dl.log", DIM),
    ("[03:48:21] w12 worker 结束", FG),
    ("[03:48:32] mon model-00012-of-00018.safetensors ✓ 大小+sha256 双验通过（3795 MB）", FG),
    ("[03:49:19] mon model-00014-of-00018.safetensors ✓ 大小+sha256 双验通过（3795 MB）", FG),
    ("[03:50:28] mon model-00016-of-00018.safetensors ✓ 大小+sha256 双验通过（3795 MB）", FG),
    ("[03:51:09] mon model-00018-of-00018.safetensors ✓ 大小+sha256 双验通过（3235 MB）", ACC),
    ("[03:51:10] mon 全部 18 个分片就位，monitor 退出", ACC),
]
render("e2_权重下载双验日志_1600x518.png", "附图 2 ｜ 服务器日志原文：Qwen3.8-27B 权重逐片「大小 + sha256」双验",
       L2, "来源：q38_dl.log（2168 B，SHA256 790C26A2…）随包于 远端现场证据/q38_dl.log，可逐字比对")

# ---------- 附图 3：起服自证日志原文 ----------
L3 = [
    ("$ cat /root/rivermind-data/_q38_serve.log", DIM),
    ("=== 0) 目录名断言（★防起错模型） ===", ACC),
    ("  目标 /root/rivermind-data/models/Qwen3.8-27B", FG),
    ("  ✓ 含 Qwen3.8-27B", FG),
    ("=== 1) GPU 前置检查 + 自动定 TP ===", ACC),
    ("  0, NVIDIA RTX PRO 6000 Blackwell Server Edition, 97887 MiB, 97251 MiB", FG),
    ("  GPU 数 1｜单卡最大 95 GB", FG),
    ("  ⇒ TP=1｜util=0.90｜maxlen=16384｜batch=2048｜mm={\"image\":1,\"video\":0}", FG),
    ("  custom all-reduce：关闭（退回 NCCL）", FG),
    ("=== 2) 权重完整性（本地，不联网） ===", ACC),
    ("  分片 18｜缺失 0｜合计 51.75 GiB｜权重条目 1199", FG),
    ("=== 3) 环境 ===", ACC),
    ("    vllm 0.23.0 | torch 2.11.0+cu130", FG),
    ("  架构登记 Qwen3_5ForConditionalGeneration: True", WARN),
    ("=== 4) 起 vLLM（逐后端重试；后台分离） ===", ACC),
    ("    ✓ 就绪（110 秒）", FG),
    ("    (APIServer) INFO 10-05 11:20:33 [model.py:611] Resolved architecture: Qwen3_5ForConditionalGeneration", FG),
    ("    (EngineCore) INFO 10-05 11:22:00 [gpu_worker.py:480] Available KV cache memory: 32.2 GiB", FG),
    ("=== 5) ★起服后自证（确认是 Qwen3.8-27B，非 32B） ===", ACC),
    ("  \"data\":[{\"id\":\"qwen38-27b\"", WARN),
    ("  \"root\":\"/root/rivermind-data/models/Qwen3.8-27B\"", WARN),
    ("=== 7) 下一步 ===", ACC),
    ("  ① 20 条冒烟（G2 可解析 ≥16/20、G3 方向 ≥12/20；<8 停）", FG),
    ("  ② 过闸 → 全量 4587 条非序数 → 装配（锚 ord 689 / 非序数 3741）", FG),
    ("=== SERVE-DONE rc=0 ===", ACC),
]
render("e3_起服自证日志_1600x655.png", "附图 3 ｜ 服务器日志原文：起服自证（回读 /v1/models 确认为 Qwen3.8-27B）",
       L3, "来源：_q38_serve.log（3193 B，SHA256 3D58F4F8…）随包于 远端现场证据/_q38_serve.log")

# ---------- 附图 4：vLLM 日志关键行 ----------
L4 = [
    ("$ grep -nE \"non-default args|Resolved architecture|Model loading took|KV cache\" vllm38.log", DIM),
    ("8:  (APIServer pid=1960) INFO 10-05 11:20:33 [api_utils.py:273] non-default args:", WARN),
    ("      {'model_tag': '/root/rivermind-data/models/Qwen3.8-27B', 'host': '0.0.0.0',", FG),
    ("       'trust_remote_code': True, 'max_model_len': 16384,", FG),
    ("       'served_model_name': ['qwen38-27b'], 'disable_custom_all_reduce': True,", FG),
    ("       'gpu_memory_utilization': 0.9, 'limit_mm_per_prompt': {'image': 1, 'video': 0},", FG),
    ("       'max_num_batched_tokens': 2048, 'max_num_seqs': 8}", FG),
    ("9:  (APIServer pid=1960) INFO 10-05 11:20:33 [model.py:611] Resolved architecture:", WARN),
    ("       Qwen3_5ForConditionalGeneration", FG),
    ("14: (EngineCore pid=2176) INFO 10-05 11:20:44 [core.py:113] Initializing a V1 LLM engine (v0.23.0)", FG),
    ("       … dtype=torch.bfloat16, max_seq_len=16384, tensor_parallel_size=1,", FG),
    ("       disable_custom_all_reduce=True, seed=0, served_model_name=qwen38-27b", FG),
    ("48: (EngineCore pid=2176) INFO 10-05 11:21:49 [gpu_model_runner.py:5187] Model loading took", WARN),
    ("       51.08 GiB memory and 58.398754 seconds", FG),
    ("60: (EngineCore pid=2176) INFO 10-05 11:22:00 [gpu_worker.py:480] Available KV cache memory: 32.2 GiB", WARN),
]
render("e4_vLLM起服参数_1600x500.png", "附图 4 ｜ 服务器日志原文：vLLM 起服参数与架构解析（含毫秒级时间戳）",
       L4, "来源：vllm38.log（78855 B，SHA256 B25506A3…）随包于 远端现场证据/vllm38.log；长行已按原样换行")

# ---------- 附图 5：目录统计 + 目检图时间戳 ----------
L5 = [
    ("$ ls -la --time-style=full-iso q38_gate_vis | head -8", DIM),
    ("drwxr-xr-x 2 root root    4096 2026-10-05 04:30:04.870367967 +0800 .", FG),
    ("-rw-r--r-- 1 root root 2828476 2026-10-05 04:31:26.841625599 +0800 000001_001.png", FG),
    ("-rw-r--r-- 1 root root 2829563 2026-10-05 04:31:27.473555749 +0800 000001_002.png", FG),
    ("-rw-r--r-- 1 root root 2828468 2026-10-05 04:31:30.065269936 +0800 000001_003.png", FG),
    ("", FG),
    ("$ 目录级统计（ls | wc -l ; du -sh）", DIM),
    ("  models/Qwen3.8-27B            31 项   52G      ← 18 分片 + 配置/分词器", FG),
    ("  dataset/Images/visible      1295 项  3.9G      ← 与 queries.json 被引用图数一致", ACC),
    ("  dataset/Images/infrared     1295 项  1.9G", ACC),
    ("  dataset/Images/depth        1295 项  1.9G", ACC),
    ("  q38_gate_vis                  21 项   54M      ← G3 目检图（20 条 + 拼版）", WARN),
    ("  venv_q38                       6 项  9.0G      ← 独立环境（vllm 0.23.0 / torch 2.11.0+cu130）", FG),
    ("  output                         5 项  3.4M      ← 另一条实验路线产物（未交付）", FG),
    ("", FG),
    ("$ last -n 5  →  wtmp begins Mon Jun 22 09:18:28 2026", DIM),
]
render("e5_目录统计与目检图时间戳_1600x500.png", "附图 5 ｜ 服务器文件：G3 目检图时间戳与目录级统计（只读采集）",
       L5, "来源：远端实例只读 ls/du/last；完整原始采集见 服务器取证_20261015.txt")

# ---------- 附图 6：AI 工作会话时间线 ----------
MEM = Path(r"c:\Users\86189\.trae-cn\memory\projects\-d-Desktop-2026AiC--p2-a81c289e5a57af7b7d50")
pat = re.compile(r"\[session_id:\s*(\w+)\s*\|\s*topic_summary_time:\s*([0-9\-: ]+)\]\s*(.*)")
rows = []
for f in sorted(MEM.rglob("topics.md")):
    for ln in f.read_text("utf-8", "replace").splitlines():
        m = pat.search(ln)
        if not m:
            continue
        ts = m.group(2).strip()
        txt = re.sub(r"([Pp]assword)\s+\S+", r"\1 ***", m.group(3))
        txt = re.sub(r"root@\S+|:\d{4,5}\b", "***", txt)
        txt = re.sub(r"9AsgCRIGspNd|VvH1VXW94BFK", "***", txt)
        rows.append((ts, txt.strip()[:96]))
rows.sort()
L6 = [("# 本队开发会话时间线（AI 协作开发留痕 · 已脱敏，共 %d 条 / 15 个日期）" % len(rows), DIM)]
for ts, txt in rows:
    L6.append((f"{ts}   {txt}", FG))
render("e6_开发会话时间线_1600x952.png", "附图 6 ｜ 开发会话时间戳（2026-08-30 → 2026-10-06，共 36 条）",
       L6, "来源：本机开发会话记录（topic_summary_time）；已脱敏（移除凭据/主机串）。用于与产物文件时间戳交叉印证")

print("\n全部附图已生成 →", OUT)

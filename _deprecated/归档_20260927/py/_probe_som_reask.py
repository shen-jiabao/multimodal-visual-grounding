# -*- coding: utf-8 -*-
"""SoM 选择环节重跑小样：验证强化提示词能否修正 RTL 约定分歧。

背景：全量 1260 条可判定序数中 MISMATCH 230 条（18.3%），RTL 差错率 27.6%
远高于 LTR 10.8%。机制之一是模型对反向序数「按自身方向计数」，未遵守
`SEL_PROMPT` 声明的「1 = leftmost」。

本脚本只做小样验证（不铺开、不改任何已有产物）：
  - 分层抽样 15 条（3 类 MISMATCH + 1 类 MATCH 对照）
  - 用 `bbox_all` 重绘编号图（无需重新枚举，只重跑「选择」这一步）
  - 两种提示词变体：
      A 只澄清编号约定（1=最左、n=最右），不给出本题算式
      B 额外显式给出「从右数第 k 个 = 编号 nc-k+1」
  - 对照意义：若 B 合规率远高于 A，说明 B 只是在复述算式（循环验证），
    不能作为「模型真看对了」的证据

判据：
  - 合规率 = 新答案 == 期望编号 的比例
  - 对照条（原本 MATCH）若被改错，即为回归信号

用法：
    设好 SSH_* 环境变量后直接运行
"""
import json
import os
import re
import sys
import tempfile
from collections import Counter
from pathlib import Path

import paramiko

R = Path(r"d:\Desktop\2026AiC")
sys.path.insert(0, str(R))

import _fuse_qwen as F  # noqa: E402

RUNNER = r'''# -*- coding: utf-8 -*-
"""远端执行：对抽样条目用两种强化提示词重跑「选择」步骤（只读，不改产物）。"""
import json, os, re, time
import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

MODEL_PATH = "/srv/models/Qwen3-VL-8B-Instruct"
SAMPLE = "/srv/output/som_reask_sample.json"
OUT = "/srv/output/som_reask_probe.json"
RE_NUM = re.compile(r"\d+")

PROMPT_A = (
    'The image has {n} objects marked with red numbered circles. '
    'The numbering is STRICTLY left to right: number 1 is the LEFTMOST '
    'object and number {n} is the RIGHTMOST object. '
    'Read the question carefully: it may count from the left or from the '
    'right; convert it to the left-to-right numbering above. '
    'Which number corresponds to: "{q}"? Answer with ONLY the number.'
)
PROMPT_B = (
    'The image has {n} objects marked with red numbered circles. '
    'The numbering is STRICTLY left to right: number 1 is the LEFTMOST '
    'object and number {n} is the RIGHTMOST object. '
    'Counting from the right, the {k}-th object from the right is '
    'number {n} - {k} + 1 = {exp}. '
    'Which number corresponds to: "{q}"? Answer with ONLY the number.'
)


def draw_som(img, boxes):
    """与全量管线一致的编号绘制（红框 + 红圆底白字编号）。"""
    im = img.copy()
    d = ImageDraw.Draw(im)
    W, H = im.size
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            max(16, int(H * 0.03)))
    except Exception:
        font = ImageFont.load_default()
    for i, b in enumerate(boxes, 1):
        x1, y1, x2, y2 = b[0] * W, b[1] * H, b[2] * W, b[3] * H
        d.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=3)
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        r = max(12, int(H * 0.022))
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(255, 0, 0))
        tb = d.textbbox((0, 0), str(i), font=font)
        d.text((cx - (tb[2] - tb[0]) / 2, cy - (tb[3] - tb[1]) / 2 - tb[1]),
               str(i), fill=(255, 255, 255), font=font)
    return im


def main():
    sample = json.load(open(SAMPLE, encoding="utf-8"))
    print("小样条数", len(sample), flush=True)
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_PATH, dtype="auto", device_map="auto").eval()
    print("模型就绪 显存 %.1fGB" % (torch.cuda.memory_allocated() / 1e9),
          flush=True)

    def gen(img, prompt, max_new=24):
        msgs = [{"role": "user", "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": prompt}]}]
        inp = processor.apply_chat_template(
            msgs, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt").to("cuda")
        with torch.no_grad():
            g = model.generate(**inp, max_new_tokens=max_new, do_sample=False)
        return processor.batch_decode(
            [o[len(i):] for i, o in zip(inp.input_ids, g)],
            skip_special_tokens=True)[0].strip()

    out = []
    t0 = time.time()
    for j, s in enumerate(sample, 1):
        path = os.path.join("/srv/dataset", s["visible"])
        rec = dict(s)
        try:
            img = Image.open(path).convert("RGB")
        except Exception as ex:
            rec["err"] = repr(ex)
            out.append(rec)
            continue
        marked = draw_som(img, s["cands"])
        for var in ("A", "B"):
            if var == "A":
                p = PROMPT_A.format(n=s["nc"], q=s["query"])
            else:
                p = PROMPT_B.format(n=s["nc"], k=s["k"], q=s["query"],
                                    exp=s["exp"])
            try:
                txt = gen(marked, p)
            except Exception as ex:
                txt = ""
                rec["err_" + var] = repr(ex)[:120]
            m = RE_NUM.search(txt)
            num = int(m.group(0)) if m else None
            rec["new_" + var] = num
            rec["raw_" + var] = txt[:60]
            rec["ok_" + var] = (num == s["exp"])
        out.append(rec)
        print("[%d/%d] %s %s k=%s nc=%s exp=%s old=%s | A=%s ok=%s | B=%s "
              "ok=%s  %.0fs" % (
                  j, len(sample), s["qid"], s["d"], s["k"], s["nc"], s["exp"],
                  s["old_num"], rec.get("new_A"), rec.get("ok_A"),
                  rec.get("new_B"), rec.get("ok_B"), time.time() - t0),
              flush=True)
    json.dump(out, open(OUT, "w", encoding="utf-8"), ensure_ascii=False)
    print("\n写出", OUT, flush=True)


if __name__ == "__main__":
    main()
'''


def build_sample() -> list:
    """分层抽样：3 类 MISMATCH + 1 类 MATCH 对照，确定性取点。"""
    fin = json.loads((R / "_som_verdict_final.json").read_text(encoding="utf-8"))
    som = json.loads((R / "qwen3vl_som_all.json").read_text(encoding="utf-8"))
    test = json.loads(F.TEST_JSON.read_text(encoding="utf-8"))
    rows = fin["rows"]

    pool = {
        "C1_RTL_num_eq_k": [r for r in rows
                            if r["verdict"] == "MISMATCH" and r["d"] == "RTL"
                            and r["num"] == r["k"]],
        "C2_RTL_other": [r for r in rows
                         if r["verdict"] == "MISMATCH" and r["d"] == "RTL"
                         and r["num"] != r["k"]],
        "C3_LTR": [r for r in rows if r["verdict"] == "MISMATCH"
                   and r["d"] == "LTR"],
        "C4_CONTROL_match": [r for r in rows if r["verdict"] == "MATCH"],
    }
    want = {"C1_RTL_num_eq_k": 5, "C2_RTL_other": 4, "C3_LTR": 3,
            "C4_CONTROL_match": 3}

    sample = []
    for cat, n in want.items():
        cand = sorted(pool[cat], key=lambda r: r["qid"])
        if not cand:
            continue
        step = max(1, len(cand) // n)
        picked = cand[::step][:n]
        for r in picked:
            rec = som.get(r["qid"]) or {}
            cands = rec.get("bbox_all") or []
            if len(cands) < 2:
                continue
            sample.append({
                "cat": cat, "qid": r["qid"], "d": r["d"], "k": r["k"],
                "nc": r["nc"], "exp": r["exp"], "old_num": r["num"],
                "old_verdict": r["verdict"], "core": r["core"],
                "query": r["query"],
                "visible": test[r["qid"]]["visible"],
                "cands": cands,
            })
    return sample


def main() -> int:
    sample = build_sample()
    print("小样抽样结果:")
    for c in ("C1_RTL_num_eq_k", "C2_RTL_other", "C3_LTR", "C4_CONTROL_match"):
        sub = [s for s in sample if s["cat"] == c]
        print(f"  {c:20s} {len(sub)} 条")
    for s in sample:
        print(f"    [{s['cat'][:2]}] {s['qid']} {s['d']} k={s['k']} "
              f"nc={s['nc']} exp={s['exp']} old={s['old_num']}  "
              f"{s['query'][:58]!r}")

    tmp = Path(tempfile.mkdtemp())
    sp = tmp / "sample.json"
    rp = tmp / "runner.py"
    sp.write_text(json.dumps(sample, ensure_ascii=False), encoding="utf-8")
    rp.write_text(RUNNER, encoding="utf-8")

    cli = paramiko.SSHClient()
    cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    cli.connect(os.environ["SSH_HOST"], port=int(os.environ["SSH_PORT"]),
                username=os.environ["SSH_USER"], password=os.environ["SSH_PW"],
                timeout=25, banner_timeout=30, auth_timeout=30)
    sftp = cli.open_sftp()
    sftp.put(str(sp), "/srv/output/som_reask_sample.json")
    sftp.put(str(rp), "/root/_som_reask_run.py")
    sftp.close()
    print("\n已上传 sample + runner，开始执行（模型重载 ~3min）...")
    cmd = ("cd /root && /srv/venv-locate/bin/python -u _som_reask_run.py "
           "2>&1 | tail -n 40")
    _, o, _ = cli.exec_command(cmd, timeout=1800)
    print(o.read().decode("utf-8", "replace"))

    # 拉回结果并分析
    local = tmp / "probe.json"
    sftp = cli.open_sftp()
    try:
        sftp.get("/srv/output/som_reask_probe.json", str(local))
    except Exception as ex:
        print("拉回失败:", ex)
    sftp.close()
    cli.close()

    if local.exists():
        res = json.loads(local.read_text(encoding="utf-8"))
        print("\n=== 小样结果汇总 ===")
        for c in ("C1_RTL_num_eq_k", "C2_RTL_other", "C3_LTR",
                  "C4_CONTROL_match"):
            sub = [r for r in res if r.get("cat") == c]
            if not sub:
                continue
            oka = sum(1 for r in sub if r.get("ok_A"))
            okb = sum(1 for r in sub if r.get("ok_B"))
            print(f"  {c:20s} n={len(sub):2d}  A 合规 {oka}/{len(sub)}  "
                  f"B 合规 {okb}/{len(sub)}")
        print()
        print("逐条（A/B 变体的新答案 vs 期望）：")
        print(f"  {'qid':12s} {'cat':3s} {'d':4s} {'k':>2s} {'nc':>3s} "
              f"{'exp':>4s} {'old':>4s} {'A':>4s} {'B':>4s}")
        for r in res:
            print(f"  {r['qid']:12s} {r['cat'][:2]:3s} {r['d']:4s} "
                  f"{r['k']:>2} {r['nc']:>3} {r['exp']:>4} "
                  f"{r.get('old_num'):>4} {str(r.get('new_A')):>4} "
                  f"{str(r.get('new_B')):>4}")
        (R / "_som_reask_probe.json").write_text(
            json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\n已存 {R / '_som_reask_probe.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
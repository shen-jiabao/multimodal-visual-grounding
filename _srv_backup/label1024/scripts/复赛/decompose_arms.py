# -*- coding: utf-8 -*-
"""多变量同轮上云的事后归因器：从一次运行的诊断明细里离线还原「反事实臂」。

为什么需要它
------------
云端一次全量 ≈ 3 小时，逐变量各跑一轮的成本不可接受。但只要**在同一轮里
把每个变量的两种取值都留证**，就能事后离线还原「如果关掉它」的那一版预测，
从而在同一份运行数据上分离出各变量的贡献 —— 无需多跑一次。

复赛配置里满足「同轮留证」的变量有两类（见 docs/说明/云端运行说明书-多变量同轮.md）：

  chain_enforce        诊断已记录 ``pre_chain_box``（链修正**前**的框），
                       故链修正的整段影响可离线撤销 -> 臂 ``no_chain``
  enable_depth_prior   诊断额外记录 ``pos_box``（同一次模型调用下「只用位置
                       匹配」会给出的框）-> 臂 ``no_depth_prior``

这两条都是**纯观测**：留证逻辑不改变 det / predictions，基线行为不变。

另外还提供一份「长度账本」，用来在 ``max_new_tokens`` 与 ``multi_box_id``
两个变量之间分摊占位框的修复功劳：对每个分块，用实测字符/词元密度把
「ID 化回复长度」外推成「回显整句 label 时的长度」，再看它在 512 / 1024 两种
预算下是否会截断。密度与每框节省量都是**实测常数**（见下），不是猜测。

用法
----
    # 只做归因（不生成文件）
    python scripts/复赛/decompose_arms.py \\
        --diag outputs/diag/diag_raw_predictions_<desc>.jsonl \\
        --predictions outputs/复赛/predictions_<desc>.json

    # 额外把各臂落盘，便于直接送交评测 / 人工复核
    python scripts/复赛/decompose_arms.py --diag ... --predictions ... \\
        --out-dir outputs/复赛/audit/arms --json outputs/复赛/audit/arms/attribution.json

    # 与 09-24 基线对照（给出官方口径的 A/B 形态）
    python scripts/复赛/decompose_arms.py --diag ... --predictions ... \\
        --baseline outputs/复赛/archive/run_20260924_090232_qwen_multiimg.tar.gz

本脚本只读输入，不修改任何已有结果文件。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
for _p in (ROOT, SCRIPT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.postprocess.bbox import sanitize_bbox  # noqa: E402
import provable_errors as pe  # noqa: E402  同目录只读守门器

# 实测常数（09-28 诊断轮，5690 条 query）
# 字符/词元密度：用 45 个截断块的 reply_len（正是 512 词元的产出）反推
CHARS_PER_TOKEN = 2.912
# 每框节省字符：整句 label 回显 27.7 token/框 -> ID 回显 15.1 token/框，
# 差 12.6 token × 2.912 chars/token ≈ 36.7 chars
SAVE_CHARS_PER_BOX = 36.7
# 预算候选（词元）：512 = 09-24 基线，1024 = 本轮
BUDGETS = (512, 1024)

DEFAULT_FALLBACK = (0.49, 0.49, 0.51, 0.51)


# ---------------------------------------------------------------- 载入
def load_jsonl(path: str) -> list[dict]:
    """读取诊断 JSONL：直接 .jsonl / 归档目录 / run_*.tar.gz（内含 diag_raw.jsonl）。"""
    if path.endswith((".tar.gz", ".tgz")):
        import tarfile
        with tarfile.open(path, "r:gz") as tf:
            name = next((m.name for m in tf.getmembers()
                         if m.name.endswith("diag_raw.jsonl")), None)
            if name is None:
                raise SystemExit(f"[error] 压缩包内找不到 diag_raw.jsonl: {path}")
            with tf.extractfile(name) as fh:
                text = fh.read().decode("utf-8")
        return [json.loads(l) for l in text.splitlines() if l.strip()]
    if os.path.isdir(path):
        cand = os.path.join(path, "diag_raw.jsonl")
        if os.path.exists(cand):
            return load_jsonl(cand)
        hits = sorted(f for f in os.listdir(path) if f.endswith(".jsonl"))
        if hits:
            return load_jsonl(os.path.join(path, hits[0]))
        raise SystemExit(f"[error] 目录内找不到 diag_raw.jsonl: {path}")
    with open(path, encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def load_json(path: str) -> dict:
    return pe.load_predictions(path) if path.endswith((".tar.gz", ".tgz")) \
        or os.path.isdir(path) else json.load(open(path, encoding="utf-8"))


# ---------------------------------------------------------------- 臂还原
def _finalize(box, fallback) -> list:
    """与 scripts/infer.py 的后处理完全一致：None -> 占位框，再 sanitize。"""
    return [float(v) for v in sanitize_bbox(
        [float(v) for v in box] if box is not None else list(fallback))]


def build_arms(records: list[dict], fallback) -> tuple[dict, dict]:
    """从诊断明细还原各臂预测（已含 infer.py 的后处理，可直接对比）。

    返回 ``(arms, flags)``：

    ``arms``  臂名 -> {query_id: [x1,y1,x2,y2]}
      main             本轮实际预测
      no_chain         撤销 chain_enforce（用 pre_chain_box）
      no_depth_prior   关掉 enable_depth_prior（用 pos_box）
    ``flags`` 每个 query 上两个变量的实际取值，供判定「该臂是否可能不同」
    """
    arms = {"main": {}, "no_chain": {}, "no_depth_prior": {}}
    flags = {}
    for rec in records:
        q = rec.get("query_id")
        if not q:
            continue
        fb = rec.get("final_box")
        arms["main"][q] = _finalize(fb, fallback)
        arms["no_chain"][q] = _finalize(rec.get("pre_chain_box"), fallback)
        dp_on = bool(rec.get("enable_depth_prior"))
        # 只有真的走了深度分支时才用 pos_box 覆盖；否则两臂同源
        arms["no_depth_prior"][q] = _finalize(
            rec.get("pos_box") if dp_on else fb, fallback)
        flags[q] = {"depth_prior_on": dp_on}
    return arms, flags


def verify_main(arm_main: dict, predictions: dict) -> list[str]:
    """自洽校验：还原出的 main 必须与 predictions.json 逐条相同。

    这证明诊断明细忠实记录了最终预测；校验失败则其余臂不可信。
    """
    bad = []
    for q, b in arm_main.items():
        p = predictions.get(q)
        if p is None:
            bad.append(q)
            continue
        if any(abs(float(b[i]) - float(p[i])) > 1e-6 for i in range(4)):
            bad.append(q)
    for q in predictions:
        if q not in arm_main:
            bad.append(q)
    return bad


# ---------------------------------------------------------------- 长度账本
def build_ledger(records: list[dict], budgets=BUDGETS) -> dict:
    """按分块统计回复长度，并把 ID 臂的长度外推到「回显整句 label」的长度。

    同一分块的多条 query 共享同一次模型调用，故 reply_lens / parsed_pairs
    取自该块的第一条记录，避免重复计数。
    """
    by_chunk: dict[tuple, list[dict]] = defaultdict(list)
    for rec in records:
        by_chunk[(rec.get("visible"), rec.get("chunk_index"))].append(rec)

    chunks = []
    for (vis, ci), rs in by_chunk.items():
        first = rs[0]
        lens = [l for l in (first.get("reply_lens") or []) if l]
        reply_len = max(lens) if lens else 0
        n_boxes = len(first.get("parsed_pairs") or [])
        unclosed = max(first.get("brace_unclosed") or [0]) if \
            first.get("brace_unclosed") else 0
        chunks.append({
            "visible": vis, "chunk_index": ci, "n_queries": len(rs),
            "chunk_size": first.get("chunk_size"),
            "reply_len": reply_len,
            "n_boxes": n_boxes,
            "brace_unclosed": unclosed,
            # 实测（ID 臂）
            "tok_id": reply_len / CHARS_PER_TOKEN,
            # 外推（label 臂）
            "tok_label": (reply_len + SAVE_CHARS_PER_BOX * n_boxes) / CHARS_PER_TOKEN,
        })

    grid = {}
    for b in budgets:
        for mode, key in (("multi_box_id", "tok_id"), ("label_echo", "tok_label")):
            over = [c for c in chunks if c[key] > b]
            grid[f"{mode}@{b}"] = {
                "chunks_over": len(over),
                "queries_over": sum(c["n_queries"] for c in over),
                "max_tokens": round(max((c[key] for c in chunks), default=0.0), 1),
            }
    return {"chunks": len(chunks), "grid": grid,
            "worst_chunk": max(chunks, key=lambda c: c["tok_label"], default=None)}


def print_ledger(led: dict, budgets=BUDGETS) -> None:
    print("\n----- 长度账本（max_new_tokens × multi_box_id 的功劳分摊）-----")
    print(f"  分块数 {led['chunks']}")
    print(f"  {'输出形态':16s} {'预算':>6s} {'超预算块':>9s} {'受影响 query':>13s} {'峰值词元':>9s}")
    for mode in ("multi_box_id", "label_echo"):
        for b in budgets:
            r = led["grid"][f"{mode}@{b}"]
            print(f"  {mode:16s} {b:>6d} {r['chunks_over']:>9d} "
                  f"{r['queries_over']:>13d} {r['max_tokens']:>9.1f}")
    print("  说明：multi_box_id 行是实测值；label_echo 行由「每框省 "
          f"{SAVE_CHARS_PER_BOX} 字符」外推，密度 {CHARS_PER_TOKEN} chars/token。")
    print("  读法：若 label_echo@512 超预算、multi_box_id@512 不超，则修好占位框的是 ID 化、"
          "1024 冗余；若 multi_box_id@1024 仍超，则还需分块。")


# ---------------------------------------------------------------- 主流程
def main() -> None:
    ap = argparse.ArgumentParser(
        description="多变量同轮的事后归因：还原反事实臂 + 长度账本")
    ap.add_argument("--diag", required=True, help="outputs/diag/diag_raw_*.jsonl")
    ap.add_argument("--predictions", required=True, help="本轮 predictions.json（或归档/压缩包）")
    ap.add_argument("--root", default="dataset", help="数据根（含 queries/queries.json）")
    ap.add_argument("--baseline", default=None, help="可选：09-24 基线，做官方口径对照")
    ap.add_argument("--out-dir", default=None, help="把各臂预测落盘到该目录")
    ap.add_argument("--json", default=None, help="导出归因结果")
    ap.add_argument("--fallback", default=None, help="占位框，默认读 configs/config.yaml")
    args = ap.parse_args()

    fallback = DEFAULT_FALLBACK
    if args.fallback:
        fallback = tuple(float(v) for v in args.fallback.split(","))
    else:
        try:
            from src.config import load_config
            fallback = tuple(load_config(None).postprocess.fallback_bbox)
        except Exception:  # noqa: BLE001  配置不可读时退回默认值
            pass

    records = load_jsonl(args.diag)
    predictions = load_json(args.predictions)
    qpath = os.path.join(os.path.abspath(args.root), "queries", "queries.json")
    with open(qpath, encoding="utf-8") as fh:
        qraw = json.load(fh)

    print(f"诊断记录 {len(records)} 条 / 预测 {len(predictions)} 条 / "
          f"查询 {len(qraw)} 条")

    arms, flags = build_arms(records, fallback)

    # ---- 自洽校验
    bad = verify_main(arms["main"], predictions)
    if bad:
        print(f"\n!! 自洽校验失败：{len(bad)} 条与 predictions.json 不一致，"
              f"例如 {bad[:5]}；其余臂不可信，请先排查。")
    else:
        print("自洽校验通过：还原的 main 臂与 predictions.json 逐条一致。")

    n_dp = sum(1 for f in flags.values() if f["depth_prior_on"])
    print(f"变量取值：enable_depth_prior=true 的 query {n_dp} 条"
          f"（{100.0 * n_dp / max(len(flags), 1):.1f}%）")

    # ---- 各臂 vs main
    ra = pe.analyze(arms["main"], qraw)
    pe.print_report("main（本轮实际预测）", ra)
    summary = {}
    for name in ("no_chain", "no_depth_prior"):
        arm = arms[name]
        diff = [q for q in arms["main"] if arms["main"][q] != arm.get(q)]
        print(f"\n===== 反事实臂 {name}：与 main 差异 {len(diff)} 条 =====")
        if not diff:
            print("  该变量在本轮对预测零影响（两臂完全一致）—— 可以安全删除，"
                  "不占任何收益也不担任何风险。")
            summary[name] = {"diff": 0}
            continue
        rb = pe.analyze(arm, qraw)
        pe.print_report(name, rb, base=ra)
        pe.print_gate(ra, rb, arms["main"], arm)
        summary[name] = {
            "diff": len(diff),
            "panel_main": ra["panel_total"], "panel_arm": rb["panel_total"],
            "placeholder_main": ra["placeholder"], "placeholder_arm": rb["placeholder"],
            "delta_net": len(ra["strict_qids"] - rb["strict_qids"])
            - len(rb["strict_qids"] - ra["strict_qids"]),
        }

    # ---- 长度账本
    led = build_ledger(records)
    print_ledger(led)

    # ---- 与 09-24 基线对照
    if args.baseline:
        base = pe.load_predictions(args.baseline)
        rbase = pe.analyze(base, qraw)
        pe.print_report("09-24 基线", rbase)
        pe.print_report("本轮 main（对照 09-24）", ra, base=rbase)
        pe.print_gate(rbase, ra, base, arms["main"])

    # ---- 落盘
    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
        for name, arm in arms.items():
            p = os.path.join(args.out_dir, f"predictions_{name}.json")
            with open(p, "w", encoding="utf-8") as fh:
                json.dump(arm, fh, ensure_ascii=False, indent=1)
            print(f"已写出臂预测：{p}")

    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        payload = {
            "diag": args.diag, "predictions": args.predictions,
            "baseline": args.baseline,
            "fallback": list(fallback),
            "self_check_failed": bad,
            "constants": {"chars_per_token": CHARS_PER_TOKEN,
                          "save_chars_per_box": SAVE_CHARS_PER_BOX},
            "metrics_main": {k: v for k, v in ra.items()
                             if k not in ("strict_qids", "error_qids", "by_image")},
            "arms": summary,
            "ledger": led,
        }
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=1)
        print(f"\n已导出：{args.json}")


if __name__ == "__main__":
    main()
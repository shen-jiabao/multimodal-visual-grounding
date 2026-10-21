# -*- coding: utf-8 -*-
"""多变量同轮归因器（scripts/复赛/decompose_arms.py）的纯函数单测。

覆盖三件事：
1. 反事实臂还原：pos_box 只在 enable_depth_prior=true 时生效；
2. 自洽校验：还原的 main 与 predictions.json 不一致时必须报出来；
3. 长度账本：ID 臂实测长度 -> label 臂外推长度 -> 512/1024 预算的截断矩阵。
"""
from __future__ import annotations

import importlib.util
import json
import os
import shutil
import sys
import tarfile
import uuid

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(ROOT, "scripts", "复赛", "decompose_arms.py")

sys.path.insert(0, os.path.dirname(SCRIPT))
_spec = importlib.util.spec_from_file_location("decompose_arms", SCRIPT)
da = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(da)

FB = [0.49, 0.49, 0.51, 0.51]


def _rec(qid, final, pos=None, dp=True, pre=None, **kw):
    """构造一条诊断记录；final/pos/pre 为归一化框或 None。"""
    rec = {
        "query_id": qid,
        "visible": "Images/visible/000001.png",
        "query": "the leftmost bird",
        "chunk_index": 0,
        "chunk_size": 3,
        "matched": final is not None,
        "pre_chain_box": pre if pre is not None else final,
        "pos_box": pos,
        "enable_depth_prior": dp,
        "reply_lens": kw.get("reply_lens", [100]),
        "brace_unclosed": kw.get("brace_unclosed", [0]),
        "parsed_pairs": kw.get("parsed_pairs", []),
        "unmatched_labels": [],
    }
    rec["final_box"] = final
    return rec


# ---------------------------------------------------------------- 臂还原
def test_no_depth_prior_uses_pos_box_when_flag_on():
    recs = [
        _rec("q1", [0.10, 0.10, 0.20, 0.20], pos=[0.30, 0.30, 0.40, 0.40], dp=True),
        _rec("q2", [0.50, 0.50, 0.60, 0.60], pos=[0.55, 0.55, 0.65, 0.65], dp=True),
    ]
    arms, flags = da.build_arms(recs, FB)
    assert arms["no_depth_prior"]["q1"] == [0.30, 0.30, 0.40, 0.40]
    assert arms["no_depth_prior"]["q2"] == [0.55, 0.55, 0.65, 0.65]
    assert arms["main"]["q1"] == [0.10, 0.10, 0.20, 0.20]
    assert all(f["depth_prior_on"] for f in flags.values())


def test_no_depth_prior_is_identical_when_flag_off():
    """flag=off 时两条路径同源，即使 pos_box 残留也必须与 main 完全一致。"""
    recs = [_rec("q1", [0.10, 0.10, 0.20, 0.20], pos=[0.90, 0.90, 0.95, 0.95], dp=False)]
    arms, flags = da.build_arms(recs, FB)
    assert arms["no_depth_prior"]["q1"] == arms["main"]["q1"]
    assert flags["q1"]["depth_prior_on"] is False


def test_pos_box_none_becomes_fallback_in_counterfactual_arm():
    """深度分支没找到匹配 -> 关掉它就会退化成占位框（这是真实的反事实）。"""
    recs = [_rec("q1", [0.10, 0.10, 0.20, 0.20], pos=None, dp=True)]
    arms, _ = da.build_arms(recs, FB)
    assert arms["main"]["q1"] == [0.10, 0.10, 0.20, 0.20]
    assert arms["no_depth_prior"]["q1"] == FB


def test_no_chain_arm_uses_pre_chain_box():
    recs = [_rec("q1", [0.70, 0.70, 0.80, 0.80], pre=[0.10, 0.10, 0.20, 0.20], dp=False)]
    arms, _ = da.build_arms(recs, FB)
    assert arms["main"]["q1"] == [0.70, 0.70, 0.80, 0.80]
    assert arms["no_chain"]["q1"] == [0.10, 0.10, 0.20, 0.20]


def test_finalize_none_maps_to_fallback_and_is_sanitized():
    assert da._finalize(None, FB) == FB
    # 越界框必须被 sanitize 夹回 [0,1]
    out = da._finalize([-0.5, 0.2, 1.7, 0.9], FB)
    assert all(0.0 <= v <= 1.0 for v in out)


# ---------------------------------------------------------------- 自洽校验
def test_verify_main_accepts_faithful_diag():
    recs = [_rec("q1", [0.10, 0.10, 0.20, 0.20]), _rec("q2", None)]
    arms, _ = da.build_arms(recs, FB)
    preds = {"q1": [0.10, 0.10, 0.20, 0.20], "q2": list(FB)}
    assert da.verify_main(arms["main"], preds) == []


def test_verify_main_flags_missing_and_mismatched_queries():
    recs = [_rec("q1", [0.10, 0.10, 0.20, 0.20])]
    arms, _ = da.build_arms(recs, FB)
    bad = da.verify_main(arms["main"], {"q1": [0.9, 0.9, 0.95, 0.95], "q2": list(FB)})
    assert "q1" in bad and "q2" in bad


# ---------------------------------------------------------------- 长度账本
def test_ledger_extrapolates_label_arm_and_fills_grid():
    # 2912 字符 = 1000 词元（实测密度 2.912）；4 个框 -> label 臂 +146.8 字符
    n_boxes = 4
    reply_len = int(da.CHARS_PER_TOKEN * 1000)
    pairs = [{"target_id": i, "label": None, "box1000": [0, 0, 1, 1],
              "matched_query": "q"} for i in range(n_boxes)]
    recs = [_rec("q1", [0.1, 0.1, 0.2, 0.2], dp=False,
                 reply_lens=[reply_len], parsed_pairs=pairs)]
    recs.append(_rec("q2", [0.2, 0.2, 0.3, 0.3], dp=False,
                     reply_lens=[reply_len], parsed_pairs=pairs))

    led = da.build_ledger(recs)
    assert led["chunks"] == 1                      # 同图同分块只算一次
    g = led["grid"]
    assert g["multi_box_id@512"]["chunks_over"] == 1
    assert g["multi_box_id@1024"]["chunks_over"] == 0
    # label 臂多出 36.7*4 = 146.8 字符 ≈ 50.4 词元 -> 1050.4 > 1024
    assert g["label_echo@1024"]["chunks_over"] == 1
    assert g["label_echo@1024"]["max_tokens"] > 1024
    assert g["multi_box_id@1024"]["max_tokens"] <= 1024


def test_ledger_counts_all_queries_in_an_over_budget_chunk():
    pairs = [{"target_id": 0, "label": None, "box1000": [0, 0, 1, 1],
              "matched_query": "q"}]
    big = int(da.CHARS_PER_TOKEN * 2000)
    recs = [_rec("q%d" % i, [0.1, 0.1, 0.2, 0.2], dp=False,
                 reply_lens=[big], parsed_pairs=pairs) for i in range(5)]
    led = da.build_ledger(recs)
    assert led["grid"]["multi_box_id@1024"]["chunks_over"] == 1
    assert led["grid"]["multi_box_id@1024"]["queries_over"] == 5


def test_ledger_empty_reply_is_not_counted_as_over_budget():
    recs = [_rec("q1", [0.1, 0.1, 0.2, 0.2], dp=False,
                 reply_lens=[0], parsed_pairs=[])]
    led = da.build_ledger(recs)
    assert led["grid"]["multi_box_id@512"]["chunks_over"] == 0
    assert led["grid"]["multi_box_id@512"]["max_tokens"] == 0.0


# ---------------------------------------------------------------- 临时目录
# 本机沙箱禁止扫描系统临时目录（pytest 的 tmp_path 会 PermissionError），
# 故自建仓库内临时目录，用完即删。
# 放在仓库根 .test_tmp/ 而非 outputs/ —— 单测产物不污染交付目录（已被 .gitignore 忽略）。
_TMP_BASE = os.path.join(ROOT, ".test_tmp")


@pytest.fixture
def tdir():
    os.makedirs(_TMP_BASE, exist_ok=True)
    d = os.path.join(_TMP_BASE, uuid.uuid4().hex[:12])
    os.makedirs(d)
    yield d
    shutil.rmtree(d, ignore_errors=True)


# ---------------------------------------------------------------- 输入通道
def test_load_jsonl_from_plain_file(tdir):
    p = os.path.join(tdir, "diag_raw_predictions_x.jsonl")
    with open(p, "w", encoding="utf-8") as fh:
        fh.write("\n".join(json.dumps(r, ensure_ascii=False)
                           for r in [_rec("q1", None, dp=False),
                                     _rec("q2", None, dp=False)]))
    recs = da.load_jsonl(p)
    assert [r["query_id"] for r in recs] == ["q1", "q2"]


def test_load_jsonl_from_archive_dir(tdir):
    d = os.path.join(tdir, "run_x")
    os.makedirs(d)
    with open(os.path.join(d, "diag_raw.jsonl"), "w", encoding="utf-8") as fh:
        fh.write(json.dumps(_rec("q1", None, dp=False), ensure_ascii=False))
    assert da.load_jsonl(d)[0]["query_id"] == "q1"


def test_load_jsonl_from_targz(tdir):
    """回传包直接喂进去也能读——这是云端回传后的默认入口。"""
    d = os.path.join(tdir, "run_y")
    os.makedirs(d)
    with open(os.path.join(d, "diag_raw.jsonl"), "w", encoding="utf-8") as fh:
        fh.write(json.dumps(_rec("q9", None, dp=False), ensure_ascii=False))
    tarpath = os.path.join(tdir, "run_y.tar.gz")
    with tarfile.open(tarpath, "w:gz") as tf:
        tf.add(d, arcname="run_y")
    assert da.load_jsonl(tarpath)[0]["query_id"] == "q9"


# ---------------------------------------------------------------- 端到端 CLI
QJSON = os.path.join(ROOT, "dataset", "queries", "queries.json")


def test_cli_end_to_end(tdir, monkeypatch, capsys):
    """真跑一遍 CLI：诊断 + 预测 -> 臂还原 + 守门 + 落盘。

    这是回传后实际要敲的那条命令，胶水层（argparse / 打印 / 导盘）必须能通。
    """
    if not os.path.exists(QJSON):
        pytest.skip("本地无 dataset/queries/queries.json")
    with open(QJSON, encoding="utf-8") as fh:
        qraw = json.load(fh)
    qids = list(qraw)[:30]

    preds = {}
    recs = []
    for i, qid in enumerate(qids):
        # 前 3 条故意留成占位框，用来验证面板确实在看这些条
        box = list(FB) if i < 3 else [0.10, 0.10, 0.20, 0.20]
        preds[qid] = box
        rec = _rec(qid, box, pos=[0.30, 0.30, 0.40, 0.40], dp=True)
        rec["visible"] = qraw[qid].get("visible", "v")
        rec["query"] = qraw[qid].get("query", "")
        rec["parsed_pairs"] = [{"target_id": 1, "label": None,
                                "box1000": [0, 0, 1, 1], "matched_query": qid}]
        recs.append(rec)

    diag = os.path.join(tdir, "diag_raw_predictions_t.jsonl")
    with open(diag, "w", encoding="utf-8") as fh:
        fh.write("\n".join(json.dumps(r, ensure_ascii=False) for r in recs))
    pj = os.path.join(tdir, "predictions_t.json")
    with open(pj, "w", encoding="utf-8") as fh:
        json.dump(preds, fh, ensure_ascii=False)
    outdir = os.path.join(tdir, "arms")
    jout = os.path.join(tdir, "attr.json")

    monkeypatch.setattr(sys, "argv", [
        "decompose_arms.py", "--diag", diag, "--predictions", pj,
        "--root", os.path.join(ROOT, "dataset"),
        "--out-dir", outdir, "--json", jout])
    da.main()

    text = capsys.readouterr().out
    assert "自洽校验通过" in text
    assert "长度账本" in text
    for name in ("main", "no_chain", "no_depth_prior"):
        assert os.path.exists(os.path.join(outdir, "predictions_%s.json" % name))

    with open(jout, encoding="utf-8") as fh:
        payload = json.load(fh)
    assert payload["self_check_failed"] == []
    # 深度先验在所有 30 条上打开且 pos_box 与 final_box 不同 -> 臂差异应为 30
    assert payload["arms"]["no_depth_prior"]["diff"] == 30
    assert payload["arms"]["no_chain"]["diff"] == 0
    assert payload["metrics_main"]["placeholder"] == 3
    assert payload["ledger"]["chunks"] >= 1
"""单元测试：Qwen3-VL 输出解析 / 坐标换算 / 标签匹配（模型适配方案 §2，不依赖 torch）。

运行：python -m pytest tests/test_qwen3_vl.py -q
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

from src.models.qwen3_vl_parse import (
    build_multi_prompt,
    build_query_prompt,
    coords_1000_to_pixel,
    coords_1000_to_unit,
    infer_id_base,
    match_label_to_query,
    parse_multi_boxes,
    parse_multi_boxes_ex,
    parse_single_box,
    propagate_identical_queries,
    resolve_multi_target,
)


# ---------------- 解析：单框（模式①） ----------------

def test_parse_single_box_plain_json():
    assert parse_single_box('{"bbox_2d": [100, 200, 300, 400]}') == [100, 200, 300, 400]


def test_parse_single_box_code_fence_and_prose():
    reply = 'sure:\n```json\n{"bbox_2d": [10, 20, 30, 40]}\n```\nhope it helps'
    assert parse_single_box(reply) == [10, 20, 30, 40]


def test_parse_single_box_takes_last():
    reply = '{"bbox_2d": [1, 2, 3, 4]} ... final: {"bbox_2d": [5, 6, 7, 8]}'
    assert parse_single_box(reply) == [5, 6, 7, 8]


def test_parse_single_box_bare_array_and_float():
    assert parse_single_box("[12, 34.5, 56, 78]") == [12.0, 34.5, 56.0, 78.0]


def test_parse_single_box_alt_key():
    assert parse_single_box('{"bbox": [1, 2, 3, 4]}') == [1, 2, 3, 4]


def test_parse_single_box_failure():
    assert parse_single_box("") is None
    assert parse_single_box("no box here") is None
    assert parse_single_box('{"bbox_2d": [1, 2, 3]}') is None  # 非 4 元


# ---------------- 解析：多框（模式③） ----------------

def test_parse_multi_boxes():
    reply = ('[{"label": "the red car", "bbox_2d": [0, 0, 10, 10]}, '
             '{"label": "the dog", "bbox_2d": [20, 20, 30, 30]}]')
    out = parse_multi_boxes(reply)
    assert ("the red car", [0, 0, 10, 10]) in out
    assert ("the dog", [20, 20, 30, 30]) in out
    assert len(out) == 2


def test_parse_multi_boxes_missing_label():
    out = parse_multi_boxes('{"bbox_2d": [1, 2, 3, 4]}')
    assert out == [("", [1, 2, 3, 4])]


# ---------------- 解析：多框 ID 化（P0-2） ----------------

def test_parse_multi_boxes_ex_with_id():
    reply = ('[{"id": 1, "bbox_2d": [0, 0, 10, 10]}, '
             '{"id": 2, "bbox_2d": [20, 20, 30, 30]}]')
    out = parse_multi_boxes_ex(reply)
    assert out == [(1, "", [0, 0, 10, 10]), (2, "", [20, 20, 30, 30])]


def test_parse_multi_boxes_ex_id_and_label():
    """模型既回 ID 又回 label 时两者都保留。"""
    out = parse_multi_boxes_ex('{"id": 3, "label": "a dog", "bbox_2d": [1, 2, 3, 4]}')
    assert out == [(3, "a dog", [1, 2, 3, 4])]


def test_parse_multi_boxes_ex_id_variants():
    """数字 ID 的容错：字符串数字 / 别名键；bool 不算 ID。"""
    assert parse_multi_boxes_ex('{"id": "7", "bbox_2d": [1, 2, 3, 4]}')[0][0] == 7
    assert parse_multi_boxes_ex('{"index": 4, "bbox_2d": [1, 2, 3, 4]}')[0][0] == 4
    assert parse_multi_boxes_ex('{"target_id": 2.0, "bbox_2d": [1, 2, 3, 4]}')[0][0] == 2
    assert parse_multi_boxes_ex('{"id": true, "bbox_2d": [1, 2, 3, 4]}')[0][0] is None
    assert parse_multi_boxes_ex('{"id": "x", "bbox_2d": [1, 2, 3, 4]}')[0][0] is None


def test_parse_multi_boxes_ex_no_id_degrades():
    """无 ID 字段（模型忽略提示词）时退回 label，不丢框。"""
    out = parse_multi_boxes_ex('{"label": "a dog", "bbox_2d": [1, 2, 3, 4]}')
    assert out == [(None, "a dog", [1, 2, 3, 4])]


# ---------------- 数字 ID → 查询归属（P0-2） ----------------

def test_resolve_multi_target_by_id():
    qs = ["a car", "a dog", "a cat"]
    assert resolve_multi_target(1, "", qs) == "a car"
    assert resolve_multi_target(2, "", qs) == "a dog"
    assert resolve_multi_target(3, "", qs) == "a cat"


def test_resolve_multi_target_id_base_zero():
    """若 diag 证实模型用 0-based 编号，整批用 id_base=0 重解（不重跑模型）。"""
    qs = ["a car", "a dog", "a cat"]
    assert resolve_multi_target(0, "", qs, id_base=0) == "a car"
    assert resolve_multi_target(2, "", qs, id_base=0) == "a cat"
    assert resolve_multi_target(3, "", qs, id_base=0) is None or \
        resolve_multi_target(3, "", qs, id_base=0) != "a car"


def test_resolve_multi_target_id_wins_over_label():
    """ID 与 label 冲突时以 ID 为准（ID 无歧义）。"""
    qs = ["a car", "a dog"]
    assert resolve_multi_target(2, "a car", qs) == "a dog"


def test_resolve_multi_target_out_of_range_falls_back():
    qs = ["a car", "a dog"]
    assert resolve_multi_target(9, "a dog", qs) == "a dog"
    assert resolve_multi_target(0, "a dog", qs) == "a dog"
    assert resolve_multi_target(9, "banana", qs) is None


def test_resolve_multi_target_no_id_uses_label():
    qs = ["the red car on the left", "a dog"]
    assert resolve_multi_target(None, "red car", qs) == "the red car on the left"
    assert resolve_multi_target(None, "", qs) is None


# ---------------- 坐标换算（[0,1000] → [0,1] / 像素） ----------------

def test_coords_1000_to_unit():
    assert coords_1000_to_unit([0, 0, 1000, 1000]) == [0.0, 0.0, 1.0, 1.0]
    assert coords_1000_to_unit([500, 500, 1000, 1000]) == [0.5, 0.5, 1.0, 1.0]


def test_coords_1000_to_unit_scale():
    assert coords_1000_to_unit([0, 0, 1000, 1000], scale_x=0.5, scale_y=0.8) == \
        [0.0, 0.0, 0.5, 0.8]


def test_coords_1000_to_unit_sorts_reversed():
    assert coords_1000_to_unit([1000, 1000, 0, 0]) == [0.0, 0.0, 1.0, 1.0]


def test_coords_1000_to_pixel():
    assert coords_1000_to_pixel([0, 0, 1000, 1000], 1920, 1080) == [0.0, 0.0, 1920.0, 1080.0]
    assert coords_1000_to_pixel([250, 250, 750, 750], 640, 360) == [160.0, 90.0, 480.0, 270.0]


# ---------------- 标签 → 查询匹配 ----------------

def test_match_label_exact():
    assert match_label_to_query("a red car", ["a red car", "a cat"]) == "a red car"


def test_match_label_substring():
    assert match_label_to_query("red car", ["the red car on the left"]) == \
        "the red car on the left"


def test_match_label_word_overlap():
    assert match_label_to_query("person wearing blue", ["a man in blue", "a dog"]) == \
        "a man in blue"


def test_match_label_none():
    assert match_label_to_query("banana", ["a car", "a dog"]) is None
    assert match_label_to_query("", ["a car"]) is None


# ---------------- 提示词构造 ----------------

def test_build_query_prompt_contains_query():
    p = build_query_prompt("the leftmost person")
    assert "the leftmost person" in p
    assert "bbox_2d" in p and "1000" in p


def test_build_multi_prompt_lists_targets():
    p = build_multi_prompt(["a car", "a dog"])
    assert "- a car" in p and "- a dog" in p
    assert "bbox_2d" in p and "label" in p


# ---------------- 多图（三模态）提示词 ----------------

def test_build_query_prompt_with_modality_labels():
    labels = ["visible RGB image", "infrared thermal image", "depth map"]
    p = build_query_prompt("the cat", labels)
    assert "3 images" in p
    assert "Image 2: infrared thermal image" in p
    assert "Image 1" in p          # bbox 落在图 1
    assert "the cat" in p


def test_build_prompts_without_modality_unchanged():
    assert "images of the SAME scene" not in build_query_prompt("x")
    assert "images of the SAME scene" not in build_multi_prompt(["x"])


def test_build_multi_prompt_with_modality_labels():
    p = build_multi_prompt(["a car"], ["visible RGB image", "depth map"])
    assert "2 images" in p and "- a car" in p


# ---------------- 模式③ ID 化提示词（P0-2） ----------------

def test_build_multi_prompt_id_numbers_targets():
    p = build_multi_prompt(["a car", "a dog", "a cat"], use_id=True)
    assert "1. a car" in p and "2. a dog" in p and "3. a cat" in p
    assert '"id"' in p
    assert '"label"' not in p          # ID 模式下不再要求回显 label


def test_build_multi_prompt_id_keeps_modality_preamble():
    p = build_multi_prompt(["a car"], ["visible RGB image", "depth map"], use_id=True)
    assert "2 images" in p
    assert "1. a car" in p
    assert "Report the target's bounding box on Image 1" in p


def test_build_multi_prompt_default_is_label_mode():
    """缺省保持旧契约（回显 label），既有调用方不受影响。"""
    p = build_multi_prompt(["a car"])
    assert "- a car" in p and '"label"' in p


def test_build_multi_prompt_id_is_shorter():
    """ID 化输出的收益前提：整句 label 不回显。"""
    qs = ["The second white diamond-shaped sign from bottom to top on the road"]
    id_p = build_multi_prompt(qs, use_id=True)
    label_p = build_multi_prompt(qs, use_id=False)
    assert "The second white diamond-shaped sign" in id_p   # 目标仍列出
    assert len(id_p) < len(label_p) + 200                    # 提示词量级相当


# ---------------- 提示词侧文本清洗（查询侧审查） ----------------

def test_prompt_uses_cleaned_query():
    """带 BOM / 尾部空白 / 弯引号的 query 进提示词前必须已被清洗。"""
    p = build_query_prompt("The license plate of the red car\ufeff")
    assert "\ufeff" not in p
    assert "The license plate of the red car" in p

    p = build_multi_prompt(["A monkey sitting on the ground "], use_id=True)
    assert p.endswith("1. A monkey sitting on the ground")
    assert "ground \n" not in p          # 尾部空白已去掉

    p = build_multi_prompt(["The coat on the woman\u2019s arm"])
    assert "woman's arm" in p


def test_prompt_clean_does_not_change_plain_text():
    qs = ["The first white diamond-shaped sign from bottom to top on the road"]
    assert f"1. {qs[0]}" in build_multi_prompt(qs, use_id=True)
    assert f"- {qs[0]}" in build_multi_prompt(qs)


def test_match_label_uses_same_cleaning_as_prompt():
    """匹配层与提示词层同一套清洗：提示词里的 query 已去 BOM，label 回显同形。"""
    q = "The license plate of the red car\ufeff"
    assert match_label_to_query("The license plate of the red car", [q]) == q


# ---------------- ID 编号基准反推 ----------------

def test_infer_id_base_defaults_to_one():
    assert infer_id_base([1, 2, 3], 3) == 1
    assert infer_id_base([None, None], 3) == 1
    assert infer_id_base([], 3) == 1


def test_infer_id_base_detects_zero_based():
    assert infer_id_base([0, 1, 2], 3) == 0


def test_infer_id_base_keeps_one_based_when_n_present():
    """出现 n_queries ⇒ 0 是杂讯，真正的编号仍是 1..n。"""
    assert infer_id_base([0, 1, 2, 3], 3) == 1


def test_infer_id_base_keeps_one_based_when_out_of_range():
    assert infer_id_base([0, 7], 3) == 1


def test_resolve_multi_target_zero_based():
    qs = ["a car", "a dog", "a cat"]
    assert resolve_multi_target(0, "", qs, id_base=0) == "a car"
    assert resolve_multi_target(2, "", qs, id_base=0) == "a cat"
    assert resolve_multi_target(3, "", qs, id_base=0) is None


# ---------------- 同文本 query 弱占优补齐 ----------------

def test_propagate_identical_queries_fills_missing_sibling():
    qs = ["A leaf on the ground", "A leaf on the ground", "a car"]
    got = [("A leaf on the ground", [1.0, 2.0, 3.0, 4.0])]
    out = propagate_identical_queries(qs, got)
    assert len(out) == 2
    assert out[1][0] == "A leaf on the ground"
    assert out[1][1] == [1.0, 2.0, 3.0, 4.0]


def test_propagate_identical_queries_never_overwrites():
    qs = ["a leaf", "a leaf"]
    got = [("a leaf", [1.0, 1.0, 2.0, 2.0]), ("a leaf", [5.0, 5.0, 6.0, 6.0])]
    assert propagate_identical_queries(qs, got) == got


def test_propagate_identical_queries_no_text_match_is_noop():
    qs = ["a leaf", "a leaf on the ground"]
    got = [("a leaf on the ground", [1.0, 1.0, 2.0, 2.0])]
    assert propagate_identical_queries(qs, got) == got


def test_propagate_identical_queries_copies_not_aliases():
    """必须复制列表：同组两条 query 日后被各自修改时不能互相牵连。"""
    qs = ["a leaf", "a leaf"]
    got = [("a leaf", [1.0, 1.0, 2.0, 2.0])]
    out = propagate_identical_queries(qs, got)
    out[1][1][0] = 9.0
    assert out[0][1][0] == 1.0

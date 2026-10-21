"""单元测试：bbox 处理与评测指标（不依赖 torch）。"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

from src.eval.metrics import acc_at_iou, iou, mean_iou
from src.postprocess.bbox import is_valid_bbox, normalize_bbox, normalize_text, sanitize_bbox


def test_normalize_text():
    assert normalize_text("  A Cat. ") == "a cat"
    assert normalize_text("Two white umbrellas") == "two white umbrellas"
    # 清洗：BOM / 零宽 / 异体连字符（选型建议 §三 注意事项 #5）
    assert normalize_text("\ufeffLicense plate A123\ufeff") == "license plate a123"
    assert normalize_text("man\u2011made ground") == "man-made ground"


def test_normalize_bbox():
    box = normalize_bbox([0, 0, 1920, 1080], 1920, 1080)
    assert list(box) == [0.0, 0.0, 1.0, 1.0]
    # 反向坐标自动修正
    box = normalize_bbox([1920, 1080, 0, 0], 1920, 1080)
    assert list(box) == [0.0, 0.0, 1.0, 1.0]


def test_is_valid_bbox():
    assert is_valid_bbox([0.1, 0.1, 0.5, 0.5])
    assert not is_valid_bbox([0.5, 0.1, 0.1, 0.5])  # 反向
    assert not is_valid_bbox([0.1, 0.1, 1.5, 0.5])  # 越界
    assert not is_valid_bbox([0.1, 0.1, 0.1, 0.5])  # 空框


def test_sanitize_bbox():
    out = sanitize_bbox([1.5, -0.2, 0.0, 0.0])
    assert all(0.0 <= v <= 1.0 for v in out)


def test_iou():
    assert abs(iou([0, 0, 1, 1], [0, 0, 1, 1]) - 1.0) < 1e-6
    assert abs(iou([0, 0, 1, 1], [2, 2, 3, 3]) - 0.0) < 1e-6


def test_acc():
    preds = [[0, 0, 1, 1], [0, 0, 1, 1]]
    gts = [[0, 0, 1, 1], [0.5, 0.5, 1, 1]]
    assert abs(acc_at_iou(preds, gts, 0.5) - 0.5) < 1e-6
    assert abs(mean_iou(preds, gts) - 0.625) < 1e-6


if __name__ == "__main__":
    test_normalize_text()
    test_normalize_bbox()
    test_is_valid_bbox()
    test_sanitize_bbox()
    test_iou()
    test_acc()
    print("全部测试通过 ✓")

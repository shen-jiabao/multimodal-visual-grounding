# -*- coding: utf-8 -*-
"""搜索式推理原语测试：IoU / IoU 聚类 / 候选投票聚合。"""
import pytest

from src.postprocess.search import aggregate_candidates, cluster_by_iou, iou_xyxy


class TestIouXyxy:
    def test_identical(self):
        assert iou_xyxy([0, 0, 1, 1], [0, 0, 1, 1]) == pytest.approx(1.0)

    def test_disjoint(self):
        assert iou_xyxy([0, 0, 1, 1], [2, 2, 3, 3]) == 0.0

    def test_one_third_overlap(self):
        # 交 1x2=2，并 4+4-2=6
        assert iou_xyxy([0, 0, 2, 2], [1, 0, 3, 2]) == pytest.approx(1 / 3)

    def test_zero_area_is_zero(self):
        assert iou_xyxy([0, 0, 0, 0], [0, 0, 1, 1]) == 0.0

    def test_symmetric(self):
        a, b = [0.1, 0.2, 0.3, 0.5], [0.2, 0.1, 0.4, 0.4]
        assert iou_xyxy(a, b) == pytest.approx(iou_xyxy(b, a))


class TestClusterByIou:
    def test_two_clusters(self):
        boxes = [[0.10, 0.1, 0.20, 0.2], [0.11, 0.1, 0.21, 0.2],
                 [0.70, 0.6, 0.80, 0.9]]
        assert cluster_by_iou(boxes, 0.5) == [[0, 1], [2]]

    def test_every_box_separate(self):
        boxes = [[0, 0, 0.1, 0.1], [0.5, 0.5, 0.6, 0.6], [0.9, 0.9, 1.0, 1.0]]
        assert cluster_by_iou(boxes, 0.5) == [[0], [1], [2]]

    def test_threshold_matters(self):
        boxes = [[0, 0, 1, 1], [0, 0, 0.6, 1]]  # IoU = 0.6
        assert cluster_by_iou(boxes, 0.5) == [[0, 1]]
        assert cluster_by_iou(boxes, 0.7) == [[0], [1]]

    def test_empty(self):
        assert cluster_by_iou([]) == []

    def test_members_in_input_order(self):
        boxes = [[0.10, 0.1, 0.20, 0.2], [0.70, 0.6, 0.80, 0.9],
                 [0.11, 0.1, 0.21, 0.2]]
        clusters = cluster_by_iou(boxes, 0.5)
        assert clusters == [[0, 2], [1]]


class TestAggregateCandidates:
    def test_empty_returns_none(self):
        assert aggregate_candidates([]) is None

    def test_two_of_three_vote(self):
        cands = [[0.10, 0.1, 0.20, 0.2], [0.11, 0.1, 0.21, 0.2],
                 [0.70, 0.6, 0.80, 0.9]]
        res = aggregate_candidates(cands, threshold=0.5, min_votes=1)
        assert res["votes"] == 2 and res["total"] == 3
        assert res["box"] == pytest.approx([0.105, 0.1, 0.205, 0.2])
        assert res["confidence"] == pytest.approx(2 / 3)

    def test_single_candidate_full_confidence(self):
        res = aggregate_candidates([[0.1, 0.2, 0.3, 0.4]])
        assert res["votes"] == 1 and res["confidence"] == pytest.approx(1.0)
        assert res["box"] == pytest.approx([0.1, 0.2, 0.3, 0.4])

    def test_count_dominates_weight_when_choosing_cluster(self):
        """选簇判据 = (票数, 权重和)：2 票的簇优先于 1 票的高权重离群。"""
        cands = [[0.10, 0.1, 0.20, 0.2], [0.11, 0.1, 0.21, 0.2],
                 [0.70, 0.6, 0.80, 0.9]]
        res = aggregate_candidates(cands, weights=[0.1, 0.1, 0.9], threshold=0.5)
        assert res["votes"] == 2
        assert res["box"] == pytest.approx([0.105, 0.1, 0.205, 0.2])
        assert res["confidence"] == pytest.approx(0.2 / 1.1)

    def test_weights_used_in_merge(self):
        cands = [[0.10, 0.1, 0.20, 0.2], [0.12, 0.1, 0.22, 0.2]]
        res = aggregate_candidates(cands, weights=[3.0, 1.0], threshold=0.5)
        assert res["votes"] == 2
        assert res["box"] == pytest.approx([0.105, 0.1, 0.205, 0.2])
        assert res["confidence"] == pytest.approx(1.0)

    def test_min_votes_rejects_weak_cluster(self):
        cands = [[0.10, 0.1, 0.20, 0.2], [0.11, 0.1, 0.21, 0.2],
                 [0.70, 0.6, 0.80, 0.9]]
        assert aggregate_candidates(cands, threshold=0.5, min_votes=3) is None

    def test_box_coords_normalized_order(self):
        res = aggregate_candidates([[0.3, 0.4, 0.1, 0.2]])
        assert res["box"] == pytest.approx([0.1, 0.2, 0.3, 0.4])

    def test_weights_length_mismatch(self):
        with pytest.raises(ValueError):
            aggregate_candidates([[0, 0, 1, 1]], weights=[1.0, 2.0])
"""Metric tests (S10). Each metric is checked against a case whose value is known
by hand, so the implementation is pinned to a definition rather than to itself."""

from __future__ import annotations

import math

import numpy as np
import pytest

from metrics import (
    Result,
    aggregate,
    auroc,
    average_precision,
    evaluate,
    f1_max,
    pro_score,
    recall_at_fpr,
    roc_curve,
)


def test_auroc_of_a_perfect_ranking_is_one():
    # Arrange
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.8, 0.9])

    # Act / Assert
    assert auroc(labels, scores) == 1.0


def test_auroc_of_an_inverted_ranking_is_zero():
    assert auroc(np.array([0, 0, 1, 1]), np.array([0.9, 0.8, 0.2, 0.1])) == 0.0


def test_auroc_of_all_tied_scores_is_one_half():
    """The k=1 degenerate case: every patch shares its nearest neighbour distance."""
    assert auroc(np.array([0, 1, 0, 1]), np.ones(4)) == 0.5


def test_auroc_matches_a_hand_computed_value():
    # Arrange: 1 positive above 2 negatives, 1 positive below both -> (2 + 0) / 4
    labels = np.array([1, 0, 0, 1])
    scores = np.array([0.9, 0.5, 0.4, 0.1])

    # Act / Assert
    assert auroc(labels, scores) == pytest.approx(0.5)


def test_auroc_needs_both_classes():
    with pytest.raises(ValueError, match="both classes"):
        auroc(np.array([1, 1]), np.array([0.2, 0.7]))


def test_auroc_rejects_non_binary_labels():
    with pytest.raises(ValueError, match="labels must be 0/1"):
        auroc(np.array([0, 2]), np.array([0.1, 0.2]))


def test_auroc_rejects_size_mismatch():
    with pytest.raises(ValueError, match="size mismatch"):
        auroc(np.array([0, 1, 1]), np.array([0.1, 0.2]))


def test_average_precision_of_a_perfect_ranking_is_one():
    assert average_precision(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9])) == 1.0


def test_average_precision_matches_a_hand_computed_value():
    # Arrange: ranking P N P -> precisions 1.0 at rank 1 and 2/3 at rank 3
    labels = np.array([1, 0, 1])
    scores = np.array([0.9, 0.6, 0.3])

    # Act / Assert
    assert average_precision(labels, scores) == pytest.approx((1.0 + 2 / 3) / 2)


def test_average_precision_needs_a_positive():
    with pytest.raises(ValueError, match="at least one positive"):
        average_precision(np.zeros(4, dtype=int), np.arange(4.0))


def test_roc_curve_starts_and_ends_at_the_corners():
    # Arrange
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.4, 0.6, 0.9])

    # Act
    fpr, tpr, _ = roc_curve(labels, scores)

    # Assert
    assert fpr[-1] == 1.0 and tpr[-1] == 1.0
    assert fpr[0] <= 0.5


def test_f1_max_finds_the_best_threshold():
    # Arrange: a threshold exists that separates the classes perfectly
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.7, 0.8])

    # Act
    best_f1, threshold = f1_max(labels, scores)

    # Assert
    assert best_f1 == pytest.approx(1.0)
    assert threshold == pytest.approx(0.7)


def test_recall_at_fpr_respects_the_budget():
    # Arrange: 10 normals, 10 defects, one normal scoring above every defect
    labels = np.array([0] * 10 + [1] * 10)
    scores = np.concatenate([np.linspace(0.0, 0.3, 9), [0.99], np.linspace(0.5, 0.8, 10)])

    # Act
    recall, threshold = recall_at_fpr(labels, scores, fpr_budget=0.10)

    # Assert: the one high-scoring normal costs exactly 10% FPR, so all defects pass
    assert recall == pytest.approx(1.0)
    assert threshold <= 0.5


def test_recall_at_fpr_returns_zero_when_no_threshold_fits():
    # Arrange: every normal outranks every defect
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.9, 0.8, 0.2, 0.1])

    # Act
    recall, _ = recall_at_fpr(labels, scores, fpr_budget=0.01)

    # Assert
    assert recall == 0.0


def test_recall_at_fpr_rejects_an_impossible_budget():
    with pytest.raises(ValueError, match="fpr_budget"):
        recall_at_fpr(np.array([0, 1]), np.array([0.1, 0.9]), fpr_budget=0.0)


def test_pro_of_a_perfect_localisation_is_one():
    # Arrange: the map is exactly the mask
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[8:16, 8:16] = 1
    anomaly_map = mask.astype(float)

    # Act / Assert
    assert pro_score(mask[None], anomaly_map[None]) == pytest.approx(1.0, abs=1e-6)


def test_pro_of_a_constant_map_is_far_below_one():
    # Arrange: no information -> overlap only ever matches the false-positive rate
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[8:16, 8:16] = 1
    anomaly_map = np.full((32, 32), 0.5)

    # Act / Assert
    assert pro_score(mask[None], anomaly_map[None]) < 0.2


def test_pro_weights_regions_not_pixels():
    """A small region found and a large region missed must score ~0.5, not ~0.1."""
    # Arrange: one 2x2 defect (detected) and one 10x10 defect (missed)
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[4:6, 4:6] = 1
    mask[40:50, 40:50] = 1
    anomaly_map = np.zeros((64, 64))
    anomaly_map[4:6, 4:6] = 1.0  # only the small region is highlighted

    # Act
    pro = pro_score(mask[None], anomaly_map[None])

    # Assert: pixel-weighted scoring would give ~0.04, region-weighted gives ~0.5
    assert 0.4 < pro < 0.6


def test_pro_needs_a_defect_region():
    with pytest.raises(ValueError, match="at least one defect region"):
        pro_score(np.zeros((1, 8, 8), dtype=np.uint8), np.zeros((1, 8, 8)))


def test_pro_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="shape mismatch"):
        pro_score(np.ones((1, 8, 8), dtype=np.uint8), np.zeros((1, 4, 4)))


def _perfect_run():
    """Two normal and two defective images, localised exactly."""
    masks = np.zeros((4, 32, 32), dtype=np.uint8)
    masks[2, 8:16, 8:16] = 1
    masks[3, 20:28, 20:28] = 1
    maps = masks.astype(float)
    labels = np.array([0, 0, 1, 1])
    scores = maps.reshape(4, -1).max(axis=1)
    return labels, scores, masks, maps


def test_evaluate_returns_the_whole_metric_set():
    # Arrange
    labels, scores, masks, maps = _perfect_run()

    # Act
    result = evaluate(labels, scores, masks, maps)

    # Assert
    assert result.image_auroc == 1.0
    assert result.pixel_auroc == 1.0
    assert result.pro == pytest.approx(1.0, abs=1e-6)
    assert result.recall_at_fpr == 1.0
    assert set(result.as_dict()) == {
        "I-AUROC", "P-AUROC", "PRO", "pixel-AP", "F1-max", "recall@FPR<=0.1", "threshold",
        "I-AUROC(top1%)", "recall@tau", "FPR@tau",
    }
    assert math.isnan(result.as_dict()["recall@tau"])  # S7 fills it in, not S10


def test_aggregate_reports_mean_and_spread():
    # Arrange: two runs that disagree, as k=1 draws typically do
    results = [
        Result(0.9, 0.9, 0.8, 0.7, 0.8, 0.9, 0.5),
        Result(0.7, 0.9, 0.6, 0.5, 0.6, 0.7, 0.5),
    ]

    # Act
    summary = aggregate(results)

    # Assert
    assert summary["I-AUROC"] == (pytest.approx(0.8), pytest.approx(0.1))
    assert summary["P-AUROC"][1] == 0.0


def test_aggregate_rejects_an_empty_list():
    with pytest.raises(ValueError, match="no results"):
        aggregate([])


def test_pro_accepts_a_single_two_dimensional_pair():
    """Convenience path: one image passed as HxW instead of 1xHxW."""
    # Arrange
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[8:16, 8:16] = 1

    # Act / Assert
    assert pro_score(mask, mask.astype(float)) == pytest.approx(1.0, abs=1e-6)


def test_pro_needs_normal_pixels():
    # Arrange: an entirely defective image has no false-positive rate to measure
    mask = np.ones((1, 8, 8), dtype=np.uint8)

    # Act / Assert
    with pytest.raises(ValueError, match="normal pixels"):
        pro_score(mask, mask.astype(float))

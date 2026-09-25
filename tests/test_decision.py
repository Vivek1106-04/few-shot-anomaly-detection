"""S7 decision-stage tests: the threshold is fitted on normals only, and it keeps the
validation false-alarm rate inside the budget by construction."""

from __future__ import annotations

import math

import numpy as np
import pytest

from data.mvtec import MVTecAD
from data.sampler import draw_support
from decision import (
    Decision,
    OperatingPoint,
    fit_decision,
    fit_threshold,
    operating_point,
    validation_normals,
)


def test_the_threshold_keeps_validation_false_alarms_inside_the_budget():
    # Arrange
    scores = np.arange(1.0, 11.0)  # ten normal validation scores

    # Act
    threshold = fit_threshold(scores, fpr_budget=0.10)

    # Assert: reject means score > threshold, so exactly one of ten is a false alarm
    assert threshold == 9.0
    assert float((scores > threshold).mean()) <= 0.10


def test_a_zero_tolerance_budget_puts_the_threshold_at_the_largest_normal():
    # Arrange
    scores = np.array([0.2, 0.9, 0.4])

    # Act
    threshold = fit_threshold(scores, fpr_budget=1e-9)

    # Assert
    assert threshold == 0.9


@pytest.mark.parametrize("budget", [0.0, 1.5])
def test_an_impossible_budget_is_rejected(budget):
    with pytest.raises(ValueError, match="fpr_budget"):
        fit_threshold(np.ones(3), budget)


def test_a_threshold_needs_at_least_one_normal():
    with pytest.raises(ValueError, match="at least one"):
        fit_threshold(np.array([]), 0.1)


def test_the_decision_rejects_strictly_above_the_threshold():
    # Arrange
    decision = Decision(threshold=0.5, fpr_budget=0.1, n_validation=10, pixel_threshold=0.7,
                        low=0.0)

    # Act / Assert
    assert decision.verdict(0.5) == "PASS"
    assert decision.verdict(0.51) == "REJECT"


def test_fit_decision_reads_both_thresholds_from_normal_maps_only():
    # Arrange: four normal maps, each constant at its index
    maps = np.stack([np.full((8, 8), float(value)) for value in range(4)])

    # Act
    decision = fit_decision(maps, fpr_budget=0.25)

    # Assert
    assert decision.n_validation == 4
    assert decision.threshold == 2.0  # one of four normals above it
    assert decision.low == 0.0
    assert 2.0 <= decision.pixel_threshold <= 3.0


def test_fit_decision_rejects_an_empty_validation_set():
    with pytest.raises(ValueError, match="validation"):
        fit_decision(np.zeros((0, 4, 4)), 0.1)


def test_a_decision_serialises_to_plain_numbers():
    # Arrange
    decision = Decision(threshold=0.5, fpr_budget=0.1, n_validation=10, pixel_threshold=0.7,
                        low=0.1)

    # Act
    payload = decision.as_dict()

    # Assert
    assert payload == {"threshold": 0.5, "fpr_budget": 0.1, "n_validation": 10,
                       "pixel_threshold": 0.7, "low": 0.1}


def test_the_operating_point_counts_recall_and_false_alarms_at_the_threshold():
    # Arrange
    labels = np.array([0, 0, 0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.9, 0.8, 0.4])

    # Act
    point = operating_point(labels, scores, threshold=0.35, fpr_budget=0.25, target_recall=0.95)

    # Assert
    assert point.recall == 1.0
    assert point.fpr == 0.25
    assert point.meets_target


def test_the_operating_point_fails_when_recall_is_short():
    # Act
    point = operating_point(np.array([0, 1, 1]), np.array([0.1, 0.9, 0.2]), threshold=0.5,
                            fpr_budget=0.1, target_recall=0.95)

    # Assert
    assert point.recall == 0.5
    assert not point.meets_target


def test_an_undefined_operating_point_is_nan_not_zero():
    """k = full leaves no held-out normals, so there is no threshold to evaluate."""
    # Act
    point = OperatingPoint.undefined()

    # Assert
    assert math.isnan(point.recall) and math.isnan(point.fpr)
    assert not point.meets_target


def test_validation_normals_never_overlap_the_support_set(mvtec_root):
    # Arrange
    dataset = MVTecAD(mvtec_root)
    normals = dataset.train_samples("widget")
    support = draw_support(normals, 3, 0, 7, "widget")

    # Act
    validation = validation_normals(normals, support.samples, n_max=5, seed=support.seed)

    # Assert
    assert len(validation) == 5
    assert not {s.image_id for s in validation} & {s.image_id for s in support.samples}


def test_validation_normals_are_reproducible_from_the_seed(mvtec_root):
    # Arrange
    normals = MVTecAD(mvtec_root).train_samples("widget")

    # Act
    first = validation_normals(normals, normals[:2], n_max=4, seed=11)
    second = validation_normals(normals, normals[:2], n_max=4, seed=11)

    # Assert
    assert first == second


def test_validation_takes_every_remaining_normal_when_fewer_than_the_cap(mvtec_root):
    # Arrange
    normals = MVTecAD(mvtec_root).train_samples("widget")

    # Act
    validation = validation_normals(normals, normals[:10], n_max=50, seed=0)

    # Assert
    assert len(validation) == len(normals) - 10


def test_the_ceiling_run_leaves_no_validation_normals(mvtec_root):
    # Arrange
    normals = MVTecAD(mvtec_root).train_samples("widget")

    # Act / Assert
    assert validation_normals(normals, normals, n_max=5, seed=0) == ()


def test_the_conformal_rule_takes_one_rank_higher_so_unseen_normals_fit_the_budget():
    # Arrange: n = 20, b = 0.1 -> quantile rank 18, conformal rank ceil(0.9 * 21) = 19
    scores = np.arange(1.0, 21.0)

    # Act / Assert
    assert fit_threshold(scores, 0.1) == 18.0
    assert fit_threshold(scores, 0.1, rule="conformal") == 19.0


def test_the_conformal_rule_falls_back_to_the_largest_normal_when_n_is_small():
    assert fit_threshold(np.array([0.2, 0.9, 0.4]), 0.1, rule="conformal") == 0.9


def test_an_unknown_rule_is_rejected():
    with pytest.raises(ValueError, match="threshold rule"):
        fit_threshold(np.ones(3), 0.1, rule="vibes")


def test_fit_decision_passes_the_rule_through():
    maps = np.stack([np.full((4, 4), float(v)) for v in range(20)])
    assert fit_decision(maps, 0.1, "conformal").threshold == 18.0

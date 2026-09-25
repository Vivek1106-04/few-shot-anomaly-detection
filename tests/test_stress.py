"""S11a/b measurement tests: which images are queried, which normals fit the threshold,
and what each stress test reports — including when there is nothing to report."""

from __future__ import annotations

import math

import numpy as np
import pytest

from config.schema import RunConfig
from data.mvtec import MVTecAD
from data.sampler import draw_support
from stress import (
    contamination_metrics,
    coverage_metrics,
    draw_for_mode,
    held_out_normals,
    query_samples,
    validation_pool,
)


@pytest.fixture
def dataset(mvtec_root) -> MVTecAD:
    return MVTecAD(mvtec_root)


def test_a_uniform_run_queries_the_whole_test_split_and_measures_nothing_extra(dataset):
    # Arrange
    normals = dataset.train_samples("widget")
    test = dataset.test_samples("widget")
    support = draw_for_mode(normals, test, 2, 0, RunConfig(), "widget")

    # Act / Assert
    assert support.mode == "uniform"
    assert query_samples(test, support) == test
    assert validation_pool(normals, support) == normals
    assert held_out_normals(normals, support) == []


def test_contaminants_are_removed_from_the_queries(dataset):
    # Arrange
    normals = dataset.train_samples("widget")
    test = dataset.test_samples("widget")
    config = RunConfig(support_mode="contaminated", n_contaminated=2)

    # Act
    support = draw_for_mode(normals, test, 4, 0, config, "widget")
    queries = query_samples(test, support)

    # Assert
    assert len(support.contaminated) == 2
    assert len(queries) == len(test) - 2
    assert not set(support.contaminated) & set(queries)


def test_a_coverage_gap_never_validates_on_the_withheld_group(dataset):
    # Arrange: the fixture's ids 000-011 all fall in one default block, so the gap
    # sampler needs two groups; fake them by overriding the support's group label
    normals = dataset.train_samples("widget")
    support = draw_support(normals, 2, 0, 1, "widget")
    gapped = type(support)(support.samples, 2, 0, 1, "coverage_gap", held_out_group="block0")

    # Act / Assert
    assert validation_pool(normals, gapped) == []
    assert held_out_normals(normals, gapped) == normals[:20]


def test_the_coverage_gap_mode_calls_the_gap_sampler(dataset, monkeypatch):
    # Arrange
    import stress

    original = stress.id_block_group
    monkeypatch.setattr(stress, "id_block_group", lambda sample: original(sample, 4))
    normals = dataset.train_samples("widget")

    # Act
    support = draw_for_mode(normals, [], 2, 0, RunConfig(support_mode="coverage_gap"), "widget")

    # Assert
    assert support.mode == "coverage_gap"
    assert support.held_out_group in {"block0", "block1", "block2"}


def test_contamination_recall_is_split_by_defect_type(dataset):
    # Arrange
    normals = dataset.train_samples("widget")
    test = dataset.test_samples("widget")
    support = draw_for_mode(normals, test, 3, 0, RunConfig(support_mode="contaminated"),
                            "widget")
    queries = query_samples(test, support)
    poisoned = support.contaminated[0].defect_type
    rejected = np.array([q.label == 1 and q.defect_type != poisoned for q in queries])

    # Act
    metrics = contamination_metrics(support, queries, rejected)

    # Assert: the method "missed" exactly the poisoned type
    assert metrics["same_type_recall"] == 0.0
    assert metrics["other_type_recall"] == 1.0
    assert metrics["n_same_type"] == 2


def test_contamination_without_same_type_queries_is_undefined_not_zero(dataset):
    # Arrange
    normals = dataset.train_samples("widget")
    test = dataset.test_samples("widget")
    support = draw_for_mode(normals, test, 3, 0, RunConfig(support_mode="contaminated"),
                            "widget")
    normals_only = [q for q in test if q.label == 0]

    # Act
    metrics = contamination_metrics(support, normals_only, np.zeros(len(normals_only)))

    # Assert
    assert math.isnan(metrics["same_type_recall"]) and math.isnan(metrics["other_type_recall"])


def test_coverage_metrics_report_the_withheld_false_alarm_rate():
    assert coverage_metrics(np.array([0.1, 0.9, 0.95, 0.2]), threshold=0.5) == {
        "held_out_fpr": 0.5, "n_held_out": 4}


def test_coverage_metrics_with_nothing_withheld_are_undefined():
    metrics = coverage_metrics(np.array([]), threshold=0.5)
    assert math.isnan(metrics["held_out_fpr"]) and metrics["n_held_out"] == 0

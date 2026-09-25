"""Task 6 table tests: each table's numbers are the category means claimed, NaNs are
left out rather than averaged in, and the ceiling ratio divides by the right entry."""

from __future__ import annotations

import math

import pytest

from tables import ceiling_ratios, ceiling_table, ceiling_values, cost_table, decision_table, \
    stress_table


def entry(auroc: float, recall_tau: float = 0.96, fpr_tau: float = 0.05, **extra) -> dict:
    return {"I-AUROC": [auroc, 0.0], "P-AUROC": [auroc, 0.0], "PRO": [auroc, 0.0],
            "recall@FPR<=0.1": [0.9, 0.0], "recall@tau": [recall_tau, 0.0],
            "FPR@tau": [fpr_tau, 0.0], "latency_ms": 50.0, "enrol_ms": 10.0,
            "n_draws": 3, **extra}


@pytest.fixture
def summary() -> dict:
    return {"patchcore": {
        "bottle": {"4": entry(0.9, memory={"enrol_peak_mb": 10.0, "score_peak_mb": 2.0,
                                           "reference_mb": 1.0})},
        "screw": {"4": entry(0.5, recall_tau=0.5, fpr_tau=0.3,
                             memory={"enrol_peak_mb": 30.0, "score_peak_mb": 4.0,
                                     "reference_mb": 3.0})},
    }}


def row(table: str, prefix: str) -> list[str]:
    line = next(line for line in table.splitlines() if line.startswith(prefix))
    return [cell.strip() for cell in line.strip("|").split("|")]


def test_the_decision_table_counts_categories_meeting_both_targets(summary):
    cells = row(decision_table(summary), "| patchcore | 4")
    assert cells[2:] == ["0.900", "0.730", "0.175", "1 of 2"]


def test_undefined_thresholds_are_left_out_of_the_mean():
    # Arrange: the ceiling has no threshold
    summary = {"padim": {"bottle": {"250": entry(0.9, recall_tau=math.nan, fpr_tau=math.nan)}}}

    # Act
    cells = row(decision_table(summary), "| padim | 250")

    # Assert
    assert cells[3:] == ["-", "-", "0 of 0"]


def test_the_cost_table_averages_the_memory_measurements(summary):
    assert row(cost_table(summary), "| patchcore | 4")[2:] == [
        "50.000", "10.000", "20.000", "3.000", "2.000"]


def test_a_run_without_memory_shows_a_dash():
    summary = {"padim": {"bottle": {"1": entry(0.8)}}}
    assert row(cost_table(summary), "| padim | 1")[4:] == ["-", "-", "-"]


def test_the_ceiling_ratio_divides_by_the_same_categorys_ceiling(summary):
    # Arrange
    ceiling = {"patchcore+full": {"bottle": {"209": entry(1.0)},
                                  "screw": {"320": entry(0.5)}}}

    # Act
    ratios = ceiling_ratios(summary, ceiling, "patchcore", "I-AUROC")

    # Assert
    assert ratios == {"4": [0.9, 1.0]}
    assert row(ceiling_table(summary, ceiling), "| patchcore | 4")[2:] == ["0.950", "2"]


def test_a_category_without_a_ceiling_is_skipped(summary):
    ceiling = {"patchcore": {"bottle": {"209": entry(0.0)}}}  # untagged name, zero ceiling
    assert ceiling_ratios(summary, ceiling, "patchcore", "I-AUROC") == {}


def test_the_ceiling_values_table_lists_each_method(summary):
    ceiling = {"padim+full": {"bottle": {"209": entry(1.0)}, "screw": {"320": entry(0.8)}}}
    assert row(ceiling_values(ceiling), "| padim+full")[1] == "0.900"


def test_the_stress_table_reads_both_stress_measurements():
    # Arrange
    summary = {
        "patchcore+gap": {"bottle": {"4": entry(0.9, stress={"held_out_fpr": 0.4})}},
        "patchcore+contaminated": {"bottle": {"4": entry(
            0.8, stress={"same_type_recall": 0.2, "other_type_recall": 0.9})}},
    }

    # Act
    table = stress_table(summary)

    # Assert
    assert row(table, "| patchcore+gap | 4")[4] == "0.400"
    assert row(table, "| patchcore+contaminated | 4")[5:] == ["0.200", "0.900"]

"""Reporting tests (S8). The figures cannot be asserted pixel by pixel, so what is
tested is what the report's *numbers* depend on: that the aggregation across
categories is the one claimed, that a missing k is not silently filled in, and that
every artefact is actually written."""

from __future__ import annotations

import json

import numpy as np
import pytest

from report import (
    merge_results,
    k_axis,
    load_results,
    main,
    markdown_table,
    mean_over_categories,
    per_category_table,
    plot_category_curves,
    plot_k_curves,
    floor_summary,
)

METRICS = ("I-AUROC", "P-AUROC", "PRO")


def entry(value: float, spread: float = 0.01) -> dict[str, object]:
    return {metric: [value, spread] for metric in METRICS} | {
        "latency_ms": 30.0, "enrol_ms": 5.0, "n_draws": 3
    }


@pytest.fixture
def summary() -> dict:
    """Two methods, two categories, k in {1, 4} — small enough to verify by hand."""
    return {
        "patchcore": {
            "bottle": {"1": entry(0.90), "4": entry(0.96)},
            "screw": {"1": entry(0.70), "4": entry(0.80)},
        },
        "padim": {
            "bottle": {"1": entry(0.80), "4": entry(0.90)},
            "screw": {"1": entry(0.60), "4": entry(0.70)},
        },
    }


def test_k_axis_is_sorted_and_deduplicated(summary):
    assert k_axis(summary, "patchcore") == [1, 4]


def test_the_curve_averages_over_categories(summary):
    # Act
    ks, means, _ = mean_over_categories(summary, "patchcore", "I-AUROC")

    # Assert: (0.90 + 0.70) / 2 and (0.96 + 0.80) / 2
    assert ks == [1, 4]
    assert means == pytest.approx([0.80, 0.88])


def test_the_error_bar_is_the_within_category_spread_over_draws(summary):
    """Not the spread between categories, which is much larger and means something else."""
    # Act
    _, _, spreads = mean_over_categories(summary, "patchcore", "I-AUROC")

    # Assert
    assert spreads == pytest.approx([0.01, 0.01])


def test_a_category_missing_one_k_is_skipped_not_imputed(summary):
    # Arrange
    del summary["patchcore"]["screw"]["4"]

    # Act
    _, means, _ = mean_over_categories(summary, "patchcore", "I-AUROC")

    # Assert: k=4 is now bottle alone
    assert means[1] == pytest.approx(0.96)


def test_the_headline_table_has_one_row_per_method_and_k(summary):
    # Act
    table = markdown_table(summary)

    # Assert
    rows = [line for line in table.splitlines() if line.startswith("| p")]
    assert len(rows) == 4
    assert "0.880 +/- 0.010" in table


def test_the_per_category_table_lists_every_category(summary):
    # Act
    table = per_category_table(summary)

    # Assert
    assert "| bottle |" in table and "| screw |" in table


def test_the_per_category_table_marks_a_combination_that_was_not_run(summary):
    # Arrange
    del summary["padim"]["screw"]

    # Act
    table = per_category_table(summary)

    # Assert
    assert "-" in table.split("| screw |")[1]


def test_both_figures_are_written(tmp_path, summary):
    # Act
    curves = plot_k_curves(summary, tmp_path / "curves.png")
    per_category = plot_category_curves(summary, tmp_path / "per_category.png")

    # Assert
    assert curves.stat().st_size > 0 and per_category.stat().st_size > 0


def test_the_floor_baseline_can_be_drawn_alongside(tmp_path, summary):
    # Arrange: the week04 result file's shape
    floor_file = tmp_path / "baseline_results.json"
    floor_file.write_text(json.dumps({
        "seed": 1, "draws": 3,
        "results": {"bottle": {"1": entry(0.77), "4": entry(0.85)}},
    }))

    # Act
    path = plot_k_curves(summary, tmp_path / "curves.png", floor=floor_summary(floor_file))

    # Assert
    assert path.stat().st_size > 0


def test_a_missing_floor_file_is_not_an_error(tmp_path):
    assert floor_summary(tmp_path / "absent.json") is None


def test_main_writes_the_figures_and_the_tables(tmp_path, summary, monkeypatch, capsys):
    # Arrange
    import sys

    results = tmp_path / "results.json"
    results.write_text(json.dumps({"config": {}, "summary": summary, "runs": []}))
    monkeypatch.setattr(sys, "argv", [
        "report", "--results", str(results), "--floor", str(tmp_path / "absent.json"),
        "--output", str(tmp_path / "out"),
    ])

    # Act
    main()

    # Assert
    written = sorted(p.name for p in (tmp_path / "out").iterdir())
    assert written == ["k_vs_accuracy.png", "k_vs_accuracy_per_category.png",
                       "results_tables.md"]
    assert "Wrote" in capsys.readouterr().out


def test_load_results_reads_a_written_result_file(tmp_path, summary):
    # Arrange
    path = tmp_path / "results.json"
    path.write_text(json.dumps({"config": {"seed": 7}, "summary": summary, "runs": []}))

    # Act / Assert
    assert load_results(path)["config"]["seed"] == 7


def test_many_categories_lay_out_without_leaving_empty_axes_visible(tmp_path):
    # Arrange: 6 categories over a 5-column grid leaves 4 unused panels
    summary = {"patchcore": {f"c{i}": {"1": entry(0.9)} for i in range(6)}}

    # Act
    path = plot_category_curves(summary, tmp_path / "grid.png")

    # Assert
    assert path.stat().st_size > 0


def test_a_category_run_for_only_one_method_still_plots(tmp_path, summary):
    """The split sweep can leave one method short of a category; skip, do not crash."""
    # Arrange
    del summary["padim"]["screw"]

    # Act
    path = plot_category_curves(summary, tmp_path / "per_category.png")

    # Assert
    assert path.stat().st_size > 0


def test_an_unknown_method_name_still_plots(tmp_path):
    """The style table holds the two implemented methods; a third must not crash it."""
    # Arrange
    summary = {"fewshot": {"bottle": {"1": entry(0.5), "4": entry(0.6)}}}

    # Act
    path = plot_k_curves(summary, tmp_path / "curves.png", metrics=("I-AUROC",))

    # Assert
    assert path.stat().st_size > 0


def test_mean_over_categories_returns_arrays_the_plot_can_use(summary):
    # Act
    _, means, spreads = mean_over_categories(summary, "padim", "PRO")

    # Assert
    assert isinstance(means, np.ndarray) and means.shape == spreads.shape


# --- merging the split sweep ------------------------------------------------


def test_merging_two_files_of_different_categories_keeps_both(summary):
    # Arrange: split the fixture by category, as the parallel sweep does
    first = {"config": {"seed": 1}, "summary": {"padim": {"bottle": summary["padim"]["bottle"]}},
             "runs": [{"category": "bottle"}]}
    second = {"config": {"seed": 1}, "summary": {"padim": {"screw": summary["padim"]["screw"]}},
              "runs": [{"category": "screw"}]}

    # Act
    merged = merge_results([first, second])

    # Assert
    assert sorted(merged["summary"]["padim"]) == ["bottle", "screw"]
    assert len(merged["runs"]) == 2


def test_merging_refuses_a_category_that_appears_twice(summary):
    """Silently overwriting one process's numbers with another's is the failure mode."""
    # Arrange
    payload = {"config": {}, "summary": {"padim": {"bottle": summary["padim"]["bottle"]}},
               "runs": []}

    # Act / Assert
    with pytest.raises(ValueError, match="more than one file"):
        merge_results([payload, payload])


def test_main_merges_several_result_files_and_can_save_the_merge(tmp_path, summary, monkeypatch):
    # Arrange
    import sys

    paths = []
    for index, category in enumerate(("bottle", "screw")):
        path = tmp_path / f"part{index}.json"
        path.write_text(json.dumps({
            "config": {"seed": 1},
            "summary": {method: {category: summary[method][category]} for method in summary},
            "runs": [],
        }))
        paths.append(str(path))
    merged = tmp_path / "merged.json"
    monkeypatch.setattr(sys, "argv", [
        "report", "--results", *paths, "--floor", str(tmp_path / "absent.json"),
        "--output", str(tmp_path / "out"), "--merged", str(merged),
    ])

    # Act
    main()

    # Assert
    assert sorted(json.loads(merged.read_text())["summary"]["padim"]) == ["bottle", "screw"]


def test_variants_keep_their_method_colour_and_get_their_own_line_style():
    from report import style

    assert style("patchcore")[0] == style("patchcore+clip")[0]
    assert style("patchcore")[2] != style("patchcore+clip")[2]
    assert style("mystery+odd") == ("tab:purple", "v", "--")


def test_a_single_k_variant_is_drawn_as_a_reference_line(tmp_path, summary):
    # Arrange: a zero-shot run exists at one k only
    summary = {**summary, "winclip+zeroshot": {"bottle": {"1": entry(0.85)},
                                               "screw": {"1": entry(0.65)}}}

    # Act
    curves = plot_k_curves(summary, tmp_path / "k.png")
    per_category = plot_category_curves(summary, tmp_path / "c.png")

    # Assert
    assert curves.exists() and per_category.exists()


def test_main_adds_the_ceiling_and_stress_sections(tmp_path, summary, monkeypatch):
    # Arrange
    import sys

    def written(name, content):
        path = tmp_path / name
        path.write_text(json.dumps({"config": {}, "summary": content, "runs": []}))
        return str(path)

    results = written("r.json", summary)
    ceiling = written("c.json", {"patchcore+full": {"bottle": {"200": entry(1.0)}}})
    stress = written("s.json", {"patchcore+gap": {"bottle": {"4": entry(0.9)}}})
    monkeypatch.setattr(sys, "argv", ["report", "--results", results, "--floor", "none",
                                      "--output", str(tmp_path), "--ceiling", ceiling,
                                      "--stress", stress])

    # Act
    main()

    # Assert
    text = (tmp_path / "results_tables.md").read_text()
    for heading in ("S7 decision stage", "Cost:", "Ceiling (k = full)",
                    "fraction of the ceiling", "S11 stress tests"):
        assert heading in text

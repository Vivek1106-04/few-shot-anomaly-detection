"""Failure-catalogue tests (S11c): each bin's rule on a constructed case, and the
catalogue end to end on the miniature dataset with a fake backbone."""

from __future__ import annotations

import json

import numpy as np
import pytest

from config.schema import RunConfig
from data.mvtec import MVTecAD
from experiment import FeatureCache
from failures import (
    BINS,
    Entry,
    build_catalogue,
    classify,
    collect,
    count_table,
    examples,
    peak,
)
from test_experiment import CountingExtractor, toy_parameters

SIZE = 224


def blank() -> np.ndarray:
    return np.zeros((SIZE, SIZE), dtype=np.float32)


def map_peaking_at(row: int, column: int) -> np.ndarray:
    anomaly_map = blank()
    anomaly_map[row, column] = 1.0
    return anomaly_map


def mask_box(top: int, left: int, side: int) -> np.ndarray:
    mask = np.zeros((SIZE, SIZE), dtype=np.uint8)
    mask[top: top + side, left: left + side] = 1
    return mask


def test_peak_finds_the_maximum():
    assert peak(map_peaking_at(30, 40)) == (30, 40)


def test_a_tiny_missed_defect_is_sub_stride():
    # 10 x 10 = 100 px < 4 patches of 64 px
    assert classify(1, 0.1, blank(), mask_box(50, 50, 10), threshold=0.5) == "sub-stride defect"


def test_a_large_missed_defect_is_weak():
    assert classify(1, 0.1, blank(), mask_box(50, 50, 40), threshold=0.5) == "weak defect"


def test_a_false_alarm_peaking_at_the_edge_is_a_border_artefact():
    assert classify(0, 0.9, map_peaking_at(3, 100), blank(), 0.5) == "border artefact"


def test_a_false_alarm_in_the_middle_is_unmodelled_normal_variation():
    assert classify(0, 0.9, map_peaking_at(112, 112), blank(), 0.5) == \
        "unmodelled normal variation"


def test_a_detection_peaking_outside_the_defect_is_mislocalised():
    assert classify(1, 0.9, map_peaking_at(200, 200), mask_box(20, 20, 20), 0.5) == "mislocalised"


def test_a_detection_peaking_just_outside_the_defect_is_within_tolerance():
    # the box ends at row/column 39; the peak is 5 px past it
    assert classify(1, 0.9, map_peaking_at(44, 30), mask_box(20, 20, 20), 0.5) is None


def test_correct_passes_are_not_catalogued():
    assert classify(0, 0.1, blank(), blank(), 0.5) is None


@pytest.fixture
def dataset(mvtec_root) -> MVTecAD:
    return MVTecAD(mvtec_root)


@pytest.fixture
def toy_config() -> RunConfig:
    return RunConfig(method_parameters={"patchcore": toy_parameters("patchcore")})


def test_collect_catalogues_errors_at_the_fitted_threshold(dataset, toy_config):
    # Arrange
    cache = FeatureCache(dataset=dataset, extractor=CountingExtractor())

    # Act
    entries, size = collect(cache, "patchcore", "widget", 2, toy_config)

    # Assert
    assert size == len(dataset.test_samples("widget"))
    assert all(entry.bin in BINS for entry in entries)
    assert all(entry.anomaly_map.shape == (SIZE, SIZE) for entry in entries)


def test_collect_refuses_the_ceiling_run(dataset, toy_config):
    cache = FeatureCache(dataset=dataset, extractor=CountingExtractor())
    with pytest.raises(ValueError, match="held-out normals"):
        collect(cache, "patchcore", "widget", 12, toy_config)


def fake_entry(category: str, bin_name: str, margin: float, dataset: MVTecAD) -> Entry:
    sample = dataset.test_samples(category)[0]
    return Entry(category, sample, bin_name, 0.5 + margin, 0.5, blank())


def test_the_count_table_has_a_row_per_category_and_a_total(dataset):
    # Arrange
    entries = [fake_entry("widget", "weak defect", 0.1, dataset),
               fake_entry("widget", "mislocalised", 0.2, dataset),
               fake_entry("gasket", "border artefact", 0.3, dataset)]

    # Act
    table = count_table(entries, {"widget": 10, "gasket": 10, "empty": 0})

    # Assert
    rows = {line.split("|")[1].strip(): line for line in table.splitlines()[2:]}
    assert rows["widget"].rstrip("| ").endswith("0.100")  # mislocalised is not an error
    assert "| 0.000 |" in rows["empty"]
    assert rows["**all**"].count("| 1 ") == 3


def test_examples_take_the_clearest_cases_of_each_bin(dataset):
    # Arrange
    entries = [fake_entry("widget", "weak defect", margin, dataset)
               for margin in (0.1, 0.5, 0.3, 0.2)]

    # Act
    chosen = examples(entries, per_bin=2)

    # Assert
    assert [round(entry.score - entry.threshold, 1) for entry in chosen] == [0.5, 0.3]


def test_build_catalogue_writes_the_table_the_figure_and_the_bins(dataset, toy_config,
                                                                  tmp_path):
    # Act
    payload = build_catalogue(dataset, CountingExtractor(), ["widget", "gasket"], "patchcore",
                              2, toy_config, tmp_path)

    # Assert
    assert (tmp_path / "failure_catalogue.md").read_text().count("|") > 10
    assert (tmp_path / "failure_catalogue.png").stat().st_size > 0
    written = json.loads((tmp_path / "failure_catalogue.json").read_text())
    assert written["test_images"] == payload["test_images"]


def test_main_catalogues_with_the_configured_backbone(mvtec_root, tmp_path, monkeypatch,
                                                      capsys):
    # Arrange
    import sys

    import failures

    monkeypatch.setattr(failures, "build_extractor", lambda config: CountingExtractor())
    monkeypatch.setattr(sys, "argv", ["failures", "--data-root", str(mvtec_root),
                                      "--categories", "widget", "--k", "2",
                                      "--output", str(tmp_path)])

    # Act
    failures.main()

    # Assert
    assert "Catalogued" in capsys.readouterr().out
    assert (tmp_path / "failure_catalogue.md").exists()


def test_an_error_free_catalogue_still_writes_an_empty_figure(dataset, tmp_path):
    from failures import example_figure

    assert example_figure(dataset, [], tmp_path / "none.png").stat().st_size > 0

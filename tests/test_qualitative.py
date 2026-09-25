"""Qualitative-output tests (S8). The picture itself cannot be asserted, but the
two claims made about it can be: the five panels are the ones described, and the
decision threshold is fitted on held-out train normals only (S7)."""

from __future__ import annotations

import numpy as np
import pytest

from data.mvtec import MVTecAD
from qualitative import (
    build_panels,
    category_row,
    pixel_maps,
    qualitative_grid,
)
from test_experiment import CountingExtractor, toy_parameters

TOY = toy_parameters("patchcore")


@pytest.fixture
def dataset(mvtec_root) -> MVTecAD:
    return MVTecAD(mvtec_root)


def test_the_mask_threshold_is_fitted_on_held_out_train_normals_only(dataset, monkeypatch):
    """No test image — normal or defective — may inform the decision."""
    # Arrange
    import qualitative

    seen = {}
    original = qualitative.fit_decision

    def spy(maps, budget):
        seen["n"] = len(maps)
        return original(maps, budget)

    monkeypatch.setattr(qualitative, "fit_decision", spy)

    # Act
    category_row(dataset, CountingExtractor(), "widget", "patchcore", 2, 7, TOY)

    # Assert: 12 train normals minus the 2 in the support set; the 4 test normals unused
    assert seen["n"] == 10


def test_pixel_maps_are_produced_at_input_resolution(dataset):
    # Arrange
    from methods import build_method

    extractor = CountingExtractor()
    samples = dataset.test_samples("widget")[:2]
    method = build_method("patchcore", **TOY)
    reference = method.enrol(extractor.extract(
        np.stack([dataset.load_image(s) for s in dataset.train_samples("widget")[:2]])
    ))

    # Act
    maps = pixel_maps(extractor, method, reference, dataset, samples)

    # Assert
    assert maps.shape == (2, 224, 224)


def test_a_row_has_the_five_panels_the_report_describes(dataset):
    # Act
    panels = category_row(dataset, CountingExtractor(), "widget", "patchcore", 2, 7, TOY)

    # Assert
    titles = [title for title, _, _ in panels]
    assert titles[1:] == ["ground truth", "anomaly map", "overlay",
                          "decision @ S7 pixel tau"]
    assert titles[0].startswith("widget /")


def test_the_panels_are_all_displayable_arrays(dataset):
    # Arrange
    sample = next(s for s in dataset.test_samples("widget") if s.mask_path is not None)
    anomaly_map = np.zeros((224, 224), dtype=np.float32)

    # Act
    panels = build_panels(dataset, sample, anomaly_map, threshold=0.5)

    # Assert
    for _, data, _ in panels:
        assert data.shape[:2] == (224, 224)


def test_the_grid_writes_one_row_per_category(dataset, tmp_path):
    # Act
    path = qualitative_grid(
        dataset, CountingExtractor(), ["widget", "gasket"], "patchcore", 2, 7,
        tmp_path / "grid.png", TOY,
    )

    # Assert
    assert path.exists() and path.stat().st_size > 0


def test_the_grid_works_for_a_single_category(dataset, tmp_path):
    """One row is the layout case where matplotlib returns a 1-D axes array."""
    # Act
    path = qualitative_grid(
        dataset, CountingExtractor(), ["widget"], "padim", 2, 7,
        tmp_path / "one.png", toy_parameters("padim"),
    )

    # Assert
    assert path.exists()


def test_main_renders_with_the_real_backbone(mvtec_root, tmp_path, monkeypatch, capsys):
    pytest.importorskip("torch")
    import sys

    import qualitative

    # Arrange
    output = tmp_path / "qualitative.png"
    monkeypatch.setattr(sys, "argv", [
        "qualitative", "--root", str(mvtec_root), "--categories", "widget",
        "--method", "patchcore", "--k", "2", "--output", str(output),
    ])

    # Act
    qualitative.main()

    # Assert
    assert output.exists()
    assert "Wrote" in capsys.readouterr().out


def test_a_text_method_gets_its_prompts_from_the_extractor(dataset):
    # Arrange
    from test_experiment import TextExtractor

    extractor = TextExtractor()
    parameters = {"language_dim": 2, "n_visual_blocks": 1}

    # Act
    panels = category_row(dataset, extractor, "widget", "winclip", 2, 7, parameters)

    # Assert
    assert extractor.text_calls == 1
    assert len(panels) == 5

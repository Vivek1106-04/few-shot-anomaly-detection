"""Floor-baseline tests: the end-to-end path S1 -> score -> S5/S6 -> S10."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

from baseline import enrol, main, qualitative_figure, run_draw, score
from data.mvtec import MVTecAD
from data.sampler import draw_support
from data.transforms import CROP_SIZE


def test_enrol_averages_the_support_images(mvtec_root: Path):
    # Arrange
    dataset = MVTecAD(mvtec_root)
    support = draw_support(dataset.train_samples("widget"), k=4, draw=0, seed=1)

    # Act
    reference = enrol(dataset, support.samples)

    # Assert: the fixture's train images are identical, so the mean equals any of them
    assert reference.shape == (3, CROP_SIZE, CROP_SIZE)
    assert np.allclose(reference, dataset.load_image(support.samples[0]))


def test_score_is_low_for_a_normal_query_and_high_for_a_defect(mvtec_root: Path):
    # Arrange
    dataset = MVTecAD(mvtec_root)
    support = draw_support(dataset.train_samples("widget"), k=4, draw=0, seed=1)
    reference = enrol(dataset, support.samples)
    test = dataset.test_samples("widget")
    normal = next(s for s in test if s.label == 0)
    defective = next(s for s in test if s.label == 1)

    # Act
    normal_map = score(dataset, normal, reference)
    defect_map = score(dataset, defective, reference)

    # Assert
    assert normal_map.shape == (CROP_SIZE, CROP_SIZE)
    assert defect_map.max() > normal_map.max()


def test_run_draw_returns_metrics_and_a_latency(mvtec_root: Path):
    # Act
    result, latency_ms = run_draw(MVTecAD(mvtec_root), "widget", k=2, draw=0, seed=1234)

    # Assert
    assert 0.0 <= result.image_auroc <= 1.0
    assert 0.0 <= result.pro <= 1.0
    assert latency_ms > 0.0


def test_run_draw_is_reproducible(mvtec_root: Path):
    # Act
    first, _ = run_draw(MVTecAD(mvtec_root), "widget", k=2, draw=1, seed=7)
    second, _ = run_draw(MVTecAD(mvtec_root), "widget", k=2, draw=1, seed=7)

    # Assert: same seed and coordinates -> bit-identical metrics
    assert first.as_dict() == second.as_dict()


def test_qualitative_figure_is_written(mvtec_root: Path, tmp_path: Path):
    # Arrange
    path = tmp_path / "qualitative.png"

    # Act
    qualitative_figure(MVTecAD(mvtec_root), "widget", k=2, seed=1, path=path)

    # Assert
    assert path.exists() and path.stat().st_size > 0


def test_main_writes_results_json(mvtec_root: Path, tmp_path: Path, monkeypatch, capsys):
    # Arrange
    output = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", [
        "baseline", "--root", str(mvtec_root), "--categories", "widget",
        "--k", "1", "2", "--draws", "2", "--output", str(output),
    ])

    # Act
    main()

    # Assert
    payload = json.loads((output / "baseline_results.json").read_text(encoding="utf-8"))
    assert set(payload["results"]["widget"]) == {"1", "2"}
    assert "I-AUROC" in capsys.readouterr().out


def test_defect_free_reference_is_not_assumed_to_be_perfect(mvtec_root: Path, tmp_path: Path):
    """A query differing everywhere must score higher than the defect-only query."""
    # Arrange: write one very different test image into a copy of the fixture
    import shutil

    root = tmp_path / "copy"
    shutil.copytree(mvtec_root, root)
    odd = root / "widget" / "test" / "dent" / "999.png"
    cv2.imwrite(str(odd), np.zeros((300, 300, 3), dtype=np.uint8))
    dataset = MVTecAD(root)
    support = draw_support(dataset.train_samples("widget"), k=4, draw=0, seed=1)
    reference = enrol(dataset, support.samples)
    odd_sample = next(s for s in dataset.test_samples("widget") if s.path == odd)
    defect = next(s for s in dataset.test_samples("widget")
                  if s.label == 1 and s.path != odd)

    # Act / Assert
    assert score(dataset, odd_sample, reference).max() > score(dataset, defect, reference).max()

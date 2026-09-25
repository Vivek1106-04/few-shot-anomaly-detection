"""Demonstration-path tests: enrolment (from the dataset or from files), the cache,
inspection with its verdict and boxes, and the rendered figure."""

from __future__ import annotations

import json

import numpy as np
import pytest

from data.mvtec import MVTecAD
from decision import Decision
from demo import (
    Enrolled,
    Inspection,
    boxes_of,
    cached,
    enrol_from_dataset,
    enrol_from_files,
    inspect,
    read_image,
    render,
)
from test_experiment import CountingExtractor, TextExtractor


@pytest.fixture
def dataset(mvtec_root) -> MVTecAD:
    return MVTecAD(mvtec_root)


def test_enrolment_from_the_dataset_fits_a_threshold_on_held_out_normals(dataset):
    # Act
    enrolled = enrol_from_dataset(CountingExtractor(), dataset, "padim", "widget", 2)

    # Assert
    assert len(enrolled.support_ids) == 2
    assert enrolled.decision.n_validation == 10


def test_enrolment_refuses_a_k_that_leaves_no_normals(dataset):
    with pytest.raises(ValueError, match="no held-out normals"):
        enrol_from_dataset(CountingExtractor(), dataset, "padim", "widget", 12)


def test_enrolment_from_files_uses_the_given_support_and_normals(dataset):
    # Arrange
    paths = [sample.path for sample in dataset.train_samples("widget")]

    # Act
    enrolled = enrol_from_files(CountingExtractor(), "padim", "widget", paths[:2], paths[2:6])

    # Assert
    assert enrolled.decision.n_validation == 4
    assert enrolled.support_ids == tuple(str(p) for p in paths[:2])


def test_enrolment_from_files_needs_both_lists():
    with pytest.raises(ValueError, match="--support and --normals"):
        enrol_from_files(CountingExtractor(), "padim", "widget", [], [])


def test_a_text_method_is_enrolled_with_its_prompts(dataset):
    # Arrange
    extractor = TextExtractor()

    # Act: default WinCLIP parameters need 512 language channels, so expect that error
    with pytest.raises(ValueError, match="text must be"):
        enrol_from_dataset(extractor, dataset, "winclip", "widget", 2)

    # Assert: the prompts were requested before the shape check
    assert extractor.text_calls == 1


def test_the_cache_builds_once_and_then_loads(tmp_path):
    # Arrange
    calls = []

    def build():
        calls.append(1)
        return {"reference": np.arange(3)}

    # Act
    first, first_hit = cached(tmp_path / "r.pkl", build)
    second, second_hit = cached(tmp_path / "r.pkl", build)

    # Assert
    assert (first_hit, second_hit) == (False, True)
    assert len(calls) == 1
    assert np.array_equal(first["reference"], second["reference"])


def test_boxes_skip_specks_and_come_largest_first():
    # Arrange
    mask = np.zeros((50, 50), dtype=np.uint8)
    mask[1:3, 1:3] = 1  # 4 px speck
    mask[10:20, 10:20] = 1  # 100 px
    mask[30:45, 30:45] = 1  # 225 px

    # Act
    boxes = boxes_of(mask)

    # Assert
    assert boxes == ((30, 30, 15, 15), (10, 10, 10, 10))


def test_inspection_gives_a_verdict_a_map_and_boxes_only_on_reject(dataset):
    # Arrange
    enrolled = enrol_from_dataset(CountingExtractor(), dataset, "padim", "widget", 2)
    defective = next(s for s in dataset.test_samples("widget") if s.label == 1)
    normal = dataset.train_samples("widget")[0]

    # Act
    bad = inspect(CountingExtractor(), enrolled, read_image(defective.path), "widget")
    good = inspect(CountingExtractor(), enrolled, read_image(normal.path), "widget")

    # Assert
    assert bad.anomaly_map.shape == (224, 224)
    assert bad.verdict in ("PASS", "REJECT") and good.verdict in ("PASS", "REJECT")
    assert bad.latency_ms > 0.0
    for result in (bad, good):
        assert (len(result.boxes) == 0) or result.verdict == "REJECT"
    assert set(bad.as_dict()) == {"score", "verdict", "boxes", "latency_ms"}


def test_an_unreadable_image_fails_loudly(tmp_path):
    (tmp_path / "x.png").write_text("not an image")
    with pytest.raises(OSError, match="could not read"):
        read_image(tmp_path / "x.png")


def test_render_writes_a_figure_for_both_verdicts(dataset, tmp_path):
    # Arrange
    image = read_image(dataset.train_samples("widget")[0].path)
    decision = Decision(0.5, 0.1, 10, pixel_threshold=0.8, low=0.0)
    mask = np.zeros((224, 224), dtype=np.uint8)
    mask[50:90, 60:100] = 1
    rejected = Inspection(0.9, "REJECT", np.random.default_rng(0).uniform(size=(224, 224)),
                          mask, ((60, 50, 40, 40),), 5.0)
    passed = Inspection(0.1, "PASS", np.zeros((224, 224)), np.zeros_like(mask), (), 5.0)

    # Act / Assert
    for name, result in (("r.png", rejected), ("p.png", passed)):
        assert render(image, result, decision, tmp_path / name).stat().st_size > 0


@pytest.mark.parametrize("use_files", [False, True])
def test_main_enrols_once_and_reports_every_image(mvtec_root, tmp_path, monkeypatch, capsys,
                                                  use_files):
    # Arrange
    import sys

    import demo

    monkeypatch.setattr(demo, "build_extractor", lambda config: CountingExtractor())
    dataset = MVTecAD(mvtec_root)
    images = [str(s.path) for s in dataset.test_samples("widget")[:2]]
    train = [str(s.path) for s in dataset.train_samples("widget")]
    source = (["--support", *train[:2], "--normals", *train[2:6]] if use_files
              else ["--data-root", str(mvtec_root), "--k", "2"])
    argv = ["demo", "--category", "widget", "--method", "padim", "--image", *images,
            "--cache", str(tmp_path / "cache"), "--output", str(tmp_path), *source]
    monkeypatch.setattr(sys, "argv", argv)

    # Act
    demo.main()
    demo.main()

    # Assert
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].startswith("enrolment fitted") and lines[0].endswith("-conformal")
    assert lines[3].startswith("enrolment loaded from cache")
    reports = [json.loads(line) for line in lines if line.startswith("{")]
    assert len(reports) == 4
    assert all(r["verdict"] in ("PASS", "REJECT") for r in reports)
    assert len({r["figure"] for r in reports}) == 2  # one figure per distinct input image


def test_an_enrolment_is_a_frozen_record():
    enrolled = Enrolled("padim", {}, None, Decision(0.5, 0.1, 1, 0.5, 0.0), ())
    with pytest.raises(AttributeError):
        enrolled.method_name = "x"  # type: ignore[misc]

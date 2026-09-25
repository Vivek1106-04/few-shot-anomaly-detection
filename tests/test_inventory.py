"""Dataset-collection check tests: the integrity report must actually catch damage."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

from data.mvtec import MVTecAD
from inventory import check_category, main, overlay_mask, sanity_figure, write_report


def test_check_category_reports_clean_fixture(mvtec_root: Path):
    # Act
    row = check_category(MVTecAD(mvtec_root), "widget")

    # Assert
    assert row["missing_masks"] == []
    assert row["misaligned_masks"] == []
    assert row["empty_after_crop"] == []
    assert row["image_size"] == "300x300"


def test_check_category_detects_a_missing_mask(mvtec_root: Path, tmp_path: Path):
    # Arrange: copy the fixture and delete one ground-truth mask
    root = tmp_path / "damaged"
    shutil.copytree(mvtec_root, root)
    (root / "widget" / "ground_truth" / "scratch" / "000_mask.png").unlink()

    # Act
    row = check_category(MVTecAD(root), "widget")

    # Assert
    assert row["missing_masks"] == ["widget/test/scratch/000"]
    assert row["masks_present"] < row["test_anomalous"]


def test_check_category_detects_a_mask_of_the_wrong_size(mvtec_root: Path, tmp_path: Path):
    # Arrange: replace one mask with a differently sized one
    import cv2

    root = tmp_path / "resized"
    shutil.copytree(mvtec_root, root)
    path = root / "widget" / "ground_truth" / "scratch" / "000_mask.png"
    small = np.zeros((100, 100), dtype=np.uint8)
    small[40:60, 40:60] = 255
    cv2.imwrite(str(path), small)

    # Act
    row = check_category(MVTecAD(root), "widget")

    # Assert: image and mask no longer agree, so pixel metrics would be meaningless
    assert row["misaligned_masks"] == ["widget/test/scratch/000"]


def test_check_category_detects_a_defect_cropped_away(mvtec_root: Path, tmp_path: Path):
    # Arrange: put the defect in a corner that the 224 centre crop discards
    import cv2

    root = tmp_path / "cropped"
    shutil.copytree(mvtec_root, root)
    path = root / "widget" / "ground_truth" / "scratch" / "000_mask.png"
    corner = np.zeros((300, 300), dtype=np.uint8)
    corner[0:5, 0:5] = 255
    cv2.imwrite(str(path), corner)

    # Act
    row = check_category(MVTecAD(root), "widget")

    # Assert
    assert row["empty_after_crop"] == ["widget/test/scratch/000"]


def test_overlay_mask_paints_only_masked_pixels():
    # Arrange
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[2:4, 2:4] = 1

    # Act
    painted = overlay_mask(image, mask)

    # Assert
    assert painted[2, 2, 0] > 0
    assert painted[8, 8].sum() == 0


def test_write_report_lists_counts_and_clean_verdict(mvtec_root: Path, tmp_path: Path):
    # Arrange
    dataset = MVTecAD(mvtec_root)
    rows = [check_category(dataset, c) for c in dataset.categories]
    path = tmp_path / "inventory.md"

    # Act
    write_report(rows, path)
    text = path.read_text(encoding="utf-8")

    # Assert
    assert "| widget |" in text
    assert "No problems found." in text
    assert "**total (2)**" in text


def test_write_report_lists_problems(mvtec_root: Path, tmp_path: Path):
    # Arrange
    dataset = MVTecAD(mvtec_root)
    row = check_category(dataset, "widget")
    row["missing_masks"] = ["widget/test/scratch/001"]
    path = tmp_path / "inventory.md"

    # Act
    write_report([row], path)

    # Assert
    assert "missing masks" in path.read_text(encoding="utf-8")


def test_sanity_figure_is_written(mvtec_root: Path, tmp_path: Path):
    # Arrange
    dataset = MVTecAD(mvtec_root)
    path = tmp_path / "preprocessing_check.png"

    # Act
    sanity_figure(dataset, dataset.categories, path)

    # Assert
    assert path.exists() and path.stat().st_size > 0


def test_main_writes_both_artefacts(mvtec_root: Path, tmp_path: Path, monkeypatch, capsys):
    # Arrange
    output = tmp_path / "out"
    monkeypatch.setattr(
        sys, "argv", ["inventory", "--root", str(mvtec_root), "--output", str(output)]
    )

    # Act
    main()

    # Assert
    assert (output / "dataset_inventory.md").exists()
    assert (output / "preprocessing_check.png").exists()
    assert "Integrity problems: 0" in capsys.readouterr().out


def test_main_fails_on_a_missing_root(tmp_path: Path, monkeypatch):
    # Arrange
    monkeypatch.setattr(sys, "argv", ["inventory", "--root", str(tmp_path / "absent")])

    # Act / Assert
    with pytest.raises(FileNotFoundError):
        main()

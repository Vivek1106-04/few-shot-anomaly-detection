"""MVTec AD loader tests, against the miniature fixture dataset."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from conftest import DEFECT_TYPES, N_TEST_DEFECT, N_TEST_NORMAL, N_TRAIN
from data.mvtec import MVTecAD
from data.transforms import CROP_SIZE


def test_discovers_every_category(mvtec_root: Path):
    # Act
    dataset = MVTecAD(mvtec_root)

    # Assert
    assert dataset.categories == ["gasket", "widget"]


def test_missing_root_fails_fast(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="does not exist"):
        MVTecAD(tmp_path / "nope")


def test_root_without_categories_fails_fast(tmp_path: Path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError, match="no category"):
        MVTecAD(tmp_path)


def test_train_split_is_normal_only(mvtec_root: Path):
    # Act
    samples = MVTecAD(mvtec_root).train_samples("widget")

    # Assert: this is the one-class property the whole project rests on
    assert len(samples) == N_TRAIN
    assert {s.label for s in samples} == {0}
    assert {s.mask_path for s in samples} == {None}


def test_train_samples_are_returned_in_deterministic_order(mvtec_root: Path):
    # Act
    first = MVTecAD(mvtec_root).train_samples("widget")
    second = MVTecAD(mvtec_root).train_samples("widget")

    # Assert: the seeded sampler indexes into this list, so its order must be stable
    assert [s.path for s in first] == [s.path for s in second]
    assert [s.path for s in first] == sorted(s.path for s in first)


def test_test_split_counts_normals_and_defects(mvtec_root: Path):
    # Act
    samples = MVTecAD(mvtec_root).test_samples("widget")

    # Assert
    assert len(samples) == N_TEST_NORMAL + len(DEFECT_TYPES) * N_TEST_DEFECT
    assert sum(s.label for s in samples) == len(DEFECT_TYPES) * N_TEST_DEFECT


def test_every_defective_test_image_has_a_mask(mvtec_root: Path):
    # Act
    samples = MVTecAD(mvtec_root).test_samples("gasket")

    # Assert
    assert all(s.mask_path is not None for s in samples if s.label == 1)
    assert all(s.mask_path is None for s in samples if s.label == 0)


def test_unknown_category_is_rejected(mvtec_root: Path):
    with pytest.raises(KeyError, match="unknown category"):
        MVTecAD(mvtec_root).train_samples("sprocket")


def test_load_image_returns_preprocessed_tensor(mvtec_root: Path):
    # Arrange
    dataset = MVTecAD(mvtec_root)
    sample = dataset.train_samples("widget")[0]

    # Act
    tensor = dataset.load_image(sample)

    # Assert
    assert tensor.shape == (3, CROP_SIZE, CROP_SIZE)


def test_load_mask_of_a_normal_image_is_all_zero(mvtec_root: Path):
    # Arrange
    dataset = MVTecAD(mvtec_root)
    normal = next(s for s in dataset.test_samples("widget") if s.label == 0)

    # Act
    mask = dataset.load_mask(normal)

    # Assert: a normal image ships no mask file, but must still contribute pixels
    assert mask.shape == (CROP_SIZE, CROP_SIZE)
    assert mask.sum() == 0


def test_load_mask_of_a_defective_image_marks_pixels(mvtec_root: Path):
    # Arrange
    dataset = MVTecAD(mvtec_root)
    defective = next(s for s in dataset.test_samples("widget") if s.label == 1)

    # Act
    mask = dataset.load_mask(defective)

    # Assert
    assert mask.sum() > 0
    assert set(np.unique(mask)).issubset({0, 1})


def test_unreadable_image_raises(mvtec_root: Path, tmp_path: Path):
    # Arrange: a file with an image extension that is not an image
    dataset = MVTecAD(mvtec_root)
    broken = dataset.train_samples("widget")[0]
    corrupt = tmp_path / "broken.png"
    corrupt.write_bytes(b"not an image")

    # Act / Assert
    with pytest.raises(OSError, match="could not read image"):
        dataset.load_raw(type(broken)(corrupt, "widget", "train", "good", 0, None))


def test_defect_types_exclude_good(mvtec_root: Path):
    # Act
    types = MVTecAD(mvtec_root).defect_types("widget")

    # Assert
    assert types == sorted(DEFECT_TYPES)


def test_inventory_counts_match_the_fixture(mvtec_root: Path):
    # Act
    counts = MVTecAD(mvtec_root).inventory("widget")

    # Assert
    assert counts == {
        "train_normal": N_TRAIN,
        "test_normal": N_TEST_NORMAL,
        "test_anomalous": len(DEFECT_TYPES) * N_TEST_DEFECT,
        "defect_types": len(DEFECT_TYPES),
        "masks_present": len(DEFECT_TYPES) * N_TEST_DEFECT,
    }


def test_image_id_is_unique_per_sample(mvtec_root: Path):
    # Act
    dataset = MVTecAD(mvtec_root)
    ids = [s.image_id for s in dataset.test_samples("widget") + dataset.train_samples("widget")]

    # Assert: result files are keyed by image id, so collisions would merge images
    assert len(ids) == len(set(ids))


def test_unreadable_mask_raises(mvtec_root: Path, tmp_path: Path):
    # Arrange: point a defective sample at a mask file that is not an image
    dataset = MVTecAD(mvtec_root)
    defective = next(s for s in dataset.test_samples("widget") if s.label == 1)
    corrupt = tmp_path / "broken_mask.png"
    corrupt.write_bytes(b"not an image")
    broken = type(defective)(
        defective.path, "widget", "test", "scratch", 1, corrupt
    )

    # Act / Assert
    with pytest.raises(OSError, match="could not read mask"):
        dataset.load_mask(broken)

"""VisA loader tests, on a miniature tree in the official layout."""

from __future__ import annotations

import numpy as np
import pytest

from conftest import N_TEST_DEFECT, N_TEST_NORMAL, N_TRAIN
from data.visa import VisA, defect_slug


@pytest.fixture
def visa(visa_root) -> VisA:
    return VisA(visa_root)


def test_categories_come_from_the_split_file(visa):
    assert visa.categories == ["candle9", "pcb9"]
    assert visa.name == "visa"


def test_the_train_split_is_normal_only(visa):
    # Act
    train = visa.train_samples("pcb9")

    # Assert
    assert len(train) == N_TRAIN
    assert all(s.label == 0 and s.split == "train" and s.mask_path is None for s in train)


def test_the_test_split_has_normals_and_masked_defects(visa):
    # Act
    test = visa.test_samples("pcb9")
    defective = [s for s in test if s.label == 1]

    # Assert
    assert len(test) == N_TEST_NORMAL + N_TEST_DEFECT
    assert all(s.mask_path is not None and s.mask_path.exists() for s in defective)


def test_defect_names_are_read_from_the_annotation_file(visa):
    assert visa.defect_types("pcb9") == ["bent", "melt_scratch"]


def test_a_category_without_annotations_falls_back_to_a_generic_defect_name(visa):
    assert visa.defect_types("candle9") == ["anomaly"]


def test_sample_order_is_deterministic(visa):
    assert visa.test_samples("pcb9") == visa.test_samples("pcb9")


def test_images_and_masks_go_through_the_shared_s1_transform(visa):
    # Arrange
    defective = next(s for s in visa.test_samples("pcb9") if s.label == 1)

    # Act
    image = visa.load_image(defective)
    mask = visa.load_mask(defective)

    # Assert
    assert image.shape == (3, 224, 224)
    assert mask.shape == (224, 224) and set(np.unique(mask)) == {0, 1}


def test_the_inventory_counts_the_split(visa):
    assert visa.inventory("pcb9")["test_anomalous"] == N_TEST_DEFECT


def test_an_unknown_category_is_rejected(visa):
    with pytest.raises(KeyError, match="unknown category"):
        visa.train_samples("widget")


def test_a_missing_split_file_is_reported(tmp_path):
    with pytest.raises(FileNotFoundError, match="split file"):
        VisA(tmp_path)


def test_an_empty_split_file_is_reported(tmp_path):
    # Arrange
    (tmp_path / "split_csv").mkdir()
    (tmp_path / "split_csv" / "1cls.csv").write_text("object,split,label,image,mask\n")

    # Act / Assert
    with pytest.raises(FileNotFoundError, match="no rows"):
        VisA(tmp_path)


@pytest.mark.parametrize("label, slug", [
    ("chunk of wax missing", "chunk_of_wax_missing"),
    ("bent,melt", "bent_melt"),
    ("  ", "anomaly"),
])
def test_defect_labels_become_file_safe_slugs(label, slug):
    assert defect_slug(label) == slug

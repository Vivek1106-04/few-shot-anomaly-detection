"""Post-processing tests (S5/S6). These functions are applied to every method's
output, so their properties are the ones that must not drift: geometry preserved
by the upsample, monotonicity preserved by the normalisation, and an image score
that reacts to a small defect."""

from __future__ import annotations

import numpy as np
import pytest

from data.transforms import CROP_SIZE
from metrics import auroc
from postprocess import (
    binary_mask,
    image_score,
    normalise,
    robust_image_score,
    overlay,
    smooth,
    to_pixel_map,
    upsample_map,
)


def patch_map_with_hot_corner() -> np.ndarray:
    """A 28x28 patch map whose only response is the top-left patch."""
    patch_map = np.zeros((28, 28), dtype=np.float32)
    patch_map[0, 0] = 1.0
    return patch_map


def test_upsampling_takes_the_patch_grid_to_the_input_resolution():
    # Act
    upsampled = upsample_map(patch_map_with_hot_corner())

    # Assert
    assert upsampled.shape == (CROP_SIZE, CROP_SIZE)


def test_upsampling_keeps_the_response_where_the_patch_was():
    # Act
    upsampled = upsample_map(patch_map_with_hot_corner())
    row, column = np.unravel_index(np.argmax(upsampled), upsampled.shape)

    # Assert: still in the top-left corner, not moved by the resampling
    assert row < CROP_SIZE // 4 and column < CROP_SIZE // 4


def test_upsampling_rejects_a_non_two_dimensional_map():
    with pytest.raises(ValueError, match="2-D"):
        upsample_map(np.zeros((2, 28, 28)))


def test_smoothing_spreads_a_response_without_moving_its_centre():
    # Arrange
    sharp = np.zeros((CROP_SIZE, CROP_SIZE), dtype=np.float32)
    sharp[100, 100] = 1.0

    # Act
    blurred = smooth(sharp, sigma=4.0)

    # Assert
    assert np.unravel_index(np.argmax(blurred), blurred.shape) == (100, 100)
    assert (blurred > 0).sum() > 1


def test_smoothing_with_a_non_positive_sigma_is_a_no_op():
    # Arrange
    anomaly_map = np.arange(16, dtype=np.float32).reshape(4, 4)

    # Act / Assert
    assert np.array_equal(smooth(anomaly_map, sigma=0.0), anomaly_map)


def test_the_pixel_map_pipeline_produces_an_input_sized_smooth_map():
    # Act
    pixel_map = to_pixel_map(patch_map_with_hot_corner())

    # Assert
    assert pixel_map.shape == (CROP_SIZE, CROP_SIZE)
    assert pixel_map.max() <= 1.0


def test_image_score_is_the_maximum_so_one_small_defect_is_enough():
    """A mean would let the normal area outvote a small defect (Gap G3)."""
    # Arrange
    mostly_normal = np.zeros((CROP_SIZE, CROP_SIZE), dtype=np.float32)
    mostly_normal[10:14, 10:14] = 3.0

    # Act / Assert
    assert image_score(mostly_normal) == pytest.approx(3.0)
    assert image_score(mostly_normal) > mostly_normal.mean()


def test_normalisation_maps_the_stated_range_onto_zero_one():
    # Arrange
    anomaly_map = np.array([[0.0, 5.0, 10.0]], dtype=np.float32)

    # Act
    normalised = normalise(anomaly_map, 0.0, 10.0)

    # Assert
    assert normalised.tolist() == [[0.0, 0.5, 1.0]]


def test_normalisation_clips_outside_the_stated_range():
    # Act
    normalised = normalise(np.array([[-1.0, 11.0]]), 0.0, 10.0)

    # Assert
    assert normalised.min() == 0.0 and normalised.max() == 1.0


def test_normalisation_of_a_degenerate_range_is_all_zero():
    assert normalise(np.ones((2, 2)), 1.0, 1.0).tolist() == [[0.0, 0.0], [0.0, 0.0]]


def test_normalisation_cannot_change_a_ranking_metric():
    """Why it is safe to normalise for display: AUROC is rank-based."""
    # Arrange
    rng = np.random.default_rng(0)
    scores = rng.normal(size=50)
    labels = (scores > scores.mean()).astype(int)

    # Act
    rescaled = normalise(scores, scores.min(), scores.max()).ravel()

    # Assert
    assert auroc(labels, rescaled) == pytest.approx(auroc(labels, scores))


def test_binary_mask_marks_exactly_the_pixels_at_or_above_the_threshold():
    # Arrange
    anomaly_map = np.array([[0.1, 0.5], [0.9, 0.5]], dtype=np.float32)

    # Act
    mask = binary_mask(anomaly_map, 0.5)

    # Assert
    assert mask.tolist() == [[0, 1], [1, 1]]
    assert mask.dtype == np.uint8


def test_overlay_returns_a_displayable_image_of_the_same_size():
    # Arrange
    image = np.full((CROP_SIZE, CROP_SIZE, 3), 100, dtype=np.uint8)
    anomaly_map = np.zeros((CROP_SIZE, CROP_SIZE), dtype=np.float32)
    anomaly_map[50:60, 50:60] = 1.0

    # Act
    blended = overlay(image, anomaly_map)

    # Assert
    assert blended.shape == image.shape and blended.dtype == np.uint8
    assert not np.array_equal(blended[55, 55], blended[5, 5])  # the hot region differs


def test_the_robust_score_averages_the_top_one_percent_of_pixels():
    # Arrange: 100 pixels, the top one is 10, the rest 0
    anomaly_map = np.zeros((10, 10))
    anomaly_map[0, 0] = 10.0

    # Act / Assert
    assert robust_image_score(anomaly_map) == pytest.approx(10.0)
    assert robust_image_score(anomaly_map, fraction=0.02) == pytest.approx(5.0)


def test_the_robust_score_needs_more_than_one_pixel_to_agree():
    """One hot pixel moves `max` all the way; it moves the robust score far less."""
    # Arrange
    anomaly_map = np.zeros((CROP_SIZE, CROP_SIZE))
    anomaly_map[5, 5] = 100.0

    # Act / Assert
    assert image_score(anomaly_map) == 100.0
    assert robust_image_score(anomaly_map) < 1.0


@pytest.mark.parametrize("fraction", [0.0, 1.5])
def test_the_robust_score_rejects_an_impossible_fraction(fraction):
    with pytest.raises(ValueError, match="fraction"):
        robust_image_score(np.ones((4, 4)), fraction)


def test_a_fixed_value_range_keeps_a_quiet_map_dark():
    """Without a fixed range, a near-flat map is stretched to full brightness."""
    # Arrange
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    quiet = np.full((8, 8), 0.1)
    quiet[0, 0] = 0.2

    # Act
    stretched = overlay(image, quiet, alpha=1.0)
    fixed = overlay(image, quiet, alpha=1.0, value_range=(0.0, 1.0))

    # Assert
    assert stretched[0, 0].sum() > fixed[0, 0].sum()

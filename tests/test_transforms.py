"""S1 preprocessing tests."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from data.transforms import (
    CROP_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    RESIZE_SHORT_SIDE,
    centre_crop,
    denormalise,
    empty_mask,
    preprocess_image,
    preprocess_mask,
    resize_short_side,
)


def test_preprocess_image_returns_chw_tensor_at_crop_size():
    # Arrange
    image = np.full((400, 600, 3), 128, dtype=np.uint8)

    # Act
    tensor = preprocess_image(image)

    # Assert
    assert tensor.shape == (3, CROP_SIZE, CROP_SIZE)
    assert tensor.dtype == np.float32


def test_preprocess_image_applies_imagenet_normalisation():
    # Arrange: a constant image, so every pixel must map to the same known value
    image = np.zeros((300, 300, 3), dtype=np.uint8)
    image[:, :] = (255, 255, 255)  # white in BGR is white in RGB

    # Act
    tensor = preprocess_image(image)

    # Assert
    expected = (1.0 - IMAGENET_MEAN) / IMAGENET_STD
    assert np.allclose(tensor[:, 0, 0], expected, atol=1e-5)


def test_preprocess_image_converts_bgr_to_rgb():
    # Arrange: pure blue in BGR must end up in the *third* (B) channel of RGB
    image = np.zeros((300, 300, 3), dtype=np.uint8)
    image[:, :, 0] = 255

    # Act
    tensor = preprocess_image(image)

    # Assert
    assert tensor[2, 0, 0] > tensor[0, 0, 0]


def test_preprocess_image_rejects_single_channel_input():
    with pytest.raises(ValueError, match="BGR"):
        preprocess_image(np.zeros((300, 300), dtype=np.uint8))


def test_resize_short_side_preserves_aspect_ratio():
    # Arrange
    image = np.zeros((400, 800, 3), dtype=np.uint8)

    # Act
    resized = resize_short_side(image, RESIZE_SHORT_SIDE, cv2.INTER_AREA)

    # Assert
    assert resized.shape[0] == RESIZE_SHORT_SIDE
    assert resized.shape[1] == 2 * RESIZE_SHORT_SIDE


def test_centre_crop_takes_the_middle_window():
    # Arrange: mark the centre so a mis-centred crop is detectable
    image = np.zeros((300, 300), dtype=np.uint8)
    image[150, 150] = 255

    # Act
    cropped = centre_crop(image, CROP_SIZE)

    # Assert: the marked centre pixel lands on the centre of the crop
    assert cropped.shape == (CROP_SIZE, CROP_SIZE)
    assert cropped[CROP_SIZE // 2, CROP_SIZE // 2] == 255
    assert cropped.sum() == 255


def test_centre_crop_rejects_too_small_input():
    with pytest.raises(ValueError, match="cannot crop"):
        centre_crop(np.zeros((100, 100), dtype=np.uint8), CROP_SIZE)


def test_preprocess_mask_is_binary_and_crop_sized():
    # Arrange
    mask = np.zeros((300, 300), dtype=np.uint8)
    cv2.circle(mask, (150, 150), 30, 255, -1)

    # Act
    processed = preprocess_mask(mask)

    # Assert
    assert processed.shape == (CROP_SIZE, CROP_SIZE)
    assert set(np.unique(processed)).issubset({0, 1})
    assert processed.sum() > 0


def test_preprocess_mask_introduces_no_intermediate_values():
    """Nearest-neighbour interpolation, not bilinear: no invented boundary pixels."""
    # Arrange
    mask = np.zeros((501, 377), dtype=np.uint8)
    cv2.circle(mask, (188, 250), 40, 255, -1)

    # Act
    processed = preprocess_mask(mask)

    # Assert
    assert np.isin(processed, (0, 1)).all()


def test_preprocess_mask_rejects_three_channel_input():
    with pytest.raises(ValueError, match="single-channel"):
        preprocess_mask(np.zeros((300, 300, 3), dtype=np.uint8))


def test_image_and_mask_share_the_same_geometry():
    """The alignment guarantee the localisation metrics depend on."""
    # Arrange: a bright square in the image, the same square in the mask
    image = np.zeros((400, 400, 3), dtype=np.uint8)
    cv2.rectangle(image, (180, 180), (220, 220), (255, 255, 255), -1)
    mask = np.zeros((400, 400), dtype=np.uint8)
    cv2.rectangle(mask, (180, 180), (220, 220), 255, -1)

    # Act
    processed_image = denormalise(preprocess_image(image))
    processed_mask = preprocess_mask(mask)

    # Assert: the bright region and the mask cover the same pixels. Not identical
    # pixel-for-pixel — the image is resized with INTER_AREA and the mask with
    # nearest neighbour, so they differ by at most the boundary row — hence IoU.
    bright = processed_image.mean(axis=2) > 200
    mask = processed_mask > 0
    intersection = int((bright & mask).sum())
    union = int((bright | mask).sum())
    assert intersection / union > 0.8
    # and they sit in the same place: no sub-pixel drift between the two paths
    bright_centre = np.argwhere(bright).mean(axis=0)
    mask_centre = np.argwhere(mask).mean(axis=0)
    assert np.abs(bright_centre - mask_centre).max() <= 1.0


def test_empty_mask_is_all_zero_at_crop_size():
    # Act
    mask = empty_mask()

    # Assert
    assert mask.shape == (CROP_SIZE, CROP_SIZE)
    assert mask.max() == 0


def test_denormalise_round_trips_a_constant_image():
    # Arrange
    image = np.full((300, 300, 3), 200, dtype=np.uint8)

    # Act
    restored = denormalise(preprocess_image(image))

    # Assert
    assert abs(int(restored[0, 0, 0]) - 200) <= 1

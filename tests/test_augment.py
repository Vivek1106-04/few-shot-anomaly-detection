"""S3 support-augmentation tests: seeded, shape-preserving, never black-bordered."""

from __future__ import annotations

import numpy as np
import pytest

from data.augment import Perturbation, augment, sample_perturbation


def gradient_image(height: int = 60, width: int = 80) -> np.ndarray:
    """A bright image with structure, so a warp is visible and a black border would be."""
    ramp = np.linspace(100, 250, width, dtype=np.float64)[None, :, None]
    return np.broadcast_to(ramp, (height, width, 3)).astype(np.uint8).copy()


def test_the_identity_perturbation_leaves_the_image_unchanged():
    # Arrange
    image = gradient_image()
    identity = Perturbation(angle=0.0, flip=False, shift=(0.0, 0.0), scale=1.0)

    # Act / Assert
    assert np.array_equal(identity.apply(image), image)


def test_a_flip_mirrors_the_image():
    # Arrange
    image = gradient_image()
    flip = Perturbation(angle=0.0, flip=True, shift=(0.0, 0.0), scale=1.0)

    # Act / Assert
    assert np.array_equal(flip.apply(image), image[:, ::-1])


def test_rotated_copies_keep_their_shape_and_have_no_black_corners():
    # Arrange
    image = gradient_image()
    rotation = Perturbation(angle=10.0, flip=False, shift=(0.05, -0.05), scale=0.95)

    # Act
    warped = rotation.apply(image)

    # Assert
    assert warped.shape == image.shape
    assert warped.min() >= 90  # reflected border: nothing near 0 was invented


def test_sampled_perturbations_stay_inside_the_policy():
    # Arrange
    rng = np.random.default_rng(0)

    # Act
    draws = [sample_perturbation(rng) for _ in range(200)]

    # Assert
    assert all(abs(p.angle) <= 10.0 for p in draws)
    assert all(0.95 <= p.scale <= 1.05 for p in draws)
    assert all(abs(p.shift[0]) <= 0.05 and abs(p.shift[1]) <= 0.05 for p in draws)
    assert {p.flip for p in draws} == {True, False}


def test_augment_returns_n_copies_per_image_and_excludes_the_originals():
    # Arrange
    images = [gradient_image(), gradient_image() // 2]

    # Act
    copies = augment(images, n_copies=3, seed=5)

    # Assert
    assert len(copies) == 6
    assert all(copy.shape == images[0].shape for copy in copies)


def test_augment_is_reproducible_from_the_seed():
    # Arrange
    images = [gradient_image()]

    # Act
    first = augment(images, 2, seed=9)
    second = augment(images, 2, seed=9)
    other = augment(images, 2, seed=10)

    # Assert
    assert all(np.array_equal(a, b) for a, b in zip(first, second))
    assert not all(np.array_equal(a, b) for a, b in zip(first, other))


def test_zero_copies_is_the_unaugmented_run():
    assert augment([gradient_image()], 0, seed=1) == []


def test_a_negative_copy_count_is_rejected():
    with pytest.raises(ValueError, match="n_copies"):
        augment([gradient_image()], -1, seed=1)

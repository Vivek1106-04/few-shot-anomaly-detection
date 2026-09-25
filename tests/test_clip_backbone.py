"""CLIP backbone tests (S2, second backbone). The contract: the declared channel
layout, unit-length text-aligned channels, an exact change of input normalisation,
and determinism. Needs torch, open_clip and the cached OpenAI weights."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("open_clip")

from data.transforms import IMAGENET_MEAN, IMAGENET_STD  # noqa: E402
from models.clip_backbone import (  # noqa: E402
    ANOMALOUS_STATES,
    CLIP_MEAN,
    CLIP_STD,
    LANGUAGE_DIM,
    NORMAL_STATES,
    TEMPLATES,
    CLIPBackbone,
    clip_input,
    object_name,
    prompts,
)


@pytest.fixture(scope="module")
def backbone() -> CLIPBackbone:
    """One shared instance: constructing it loads ~600 MB of weights."""
    return CLIPBackbone(batch_size=2)


def preprocessed(n: int = 3) -> np.ndarray:
    return np.random.default_rng(0).normal(size=(n, 3, 224, 224)).astype(np.float32)


def test_features_follow_the_declared_layout(backbone):
    # Act
    features = backbone.extract(preprocessed(3))  # 3 images, batch 2: exercises batching

    # Assert
    spec = backbone.spec
    assert features.shape == (3, spec.channels, spec.grid, spec.grid)
    assert spec.language_dim == LANGUAGE_DIM
    assert features.dtype == np.float32


def test_the_language_channels_are_unit_length_per_patch(backbone):
    # Act
    language = backbone.extract(preprocessed(1))[0, -LANGUAGE_DIM:]

    # Assert
    assert np.allclose(np.linalg.norm(language, axis=0), 1.0, atol=1e-4)


def test_a_single_unbatched_image_is_accepted(backbone):
    assert backbone.extract(preprocessed(1)[0]).shape[0] == 1


def test_extraction_is_deterministic(backbone):
    # Arrange
    images = preprocessed(1)

    # Act / Assert
    assert np.array_equal(backbone.extract(images), backbone.extract(images))


def test_a_wrongly_shaped_input_is_rejected(backbone):
    with pytest.raises(ValueError, match="preprocessed images"):
        backbone.extract(np.zeros((1, 1, 224, 224), dtype=np.float32))


def test_the_text_embeddings_are_two_unit_vectors_that_differ(backbone):
    # Act
    text = backbone.text_embeddings("bottle")

    # Assert
    assert text.shape == (2, LANGUAGE_DIM)
    assert np.allclose(np.linalg.norm(text, axis=1), 1.0, atol=1e-5)
    assert float(text[0] @ text[1]) < 0.999


def test_renormalisation_is_an_exact_change_of_variables():
    # Arrange: an image whose ImageNet-normalised form is known
    rgb = np.random.default_rng(1).uniform(size=(3, 4, 4)).astype(np.float32)
    imagenet = (rgb - IMAGENET_MEAN[:, None, None]) / IMAGENET_STD[:, None, None]

    # Act
    converted = clip_input(imagenet)

    # Assert
    expected = (rgb - CLIP_MEAN[:, None, None]) / CLIP_STD[:, None, None]
    assert np.allclose(converted, expected, atol=1e-5)


@pytest.mark.parametrize("category, name", [
    ("bottle", "bottle"), ("metal_nut", "metal nut"), ("pcb3", "printed circuit board"),
    ("macaroni1", "macaroni"), ("chewinggum", "chewing gum"), ("pipe_fryum", "pipe fryum"),
])
def test_category_folders_become_object_names(category, name):
    assert object_name(category) == name


def test_the_prompt_ensemble_is_every_state_in_every_template():
    # Act
    normal, anomalous = prompts("metal_nut")

    # Assert
    assert len(normal) == len(NORMAL_STATES) * len(TEMPLATES)
    assert len(anomalous) == len(ANOMALOUS_STATES) * len(TEMPLATES)
    assert "a photo of a damaged metal nut for visual inspection." in anomalous

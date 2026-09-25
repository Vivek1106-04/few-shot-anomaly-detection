"""Backbone tests (S2). The contract the methods rely on is the shape, the grid
alignment of the two taps, and that nothing about the network can change between
enrolment and inference — it is frozen and in eval mode."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from models.backbone import LAYER_CHANNELS, FrozenBackbone  # noqa: E402


@pytest.fixture(scope="module")
def backbone() -> FrozenBackbone:
    """One shared instance: constructing it loads 270 MB of weights."""
    return FrozenBackbone()


def preprocessed(n: int = 2) -> np.ndarray:
    """n S1-shaped inputs."""
    return np.random.default_rng(0).normal(size=(n, 3, 224, 224)).astype(np.float32)


def test_features_have_one_row_per_image_and_the_declared_channels(backbone):
    # Act
    features = backbone.extract(preprocessed(2))

    # Assert
    assert features.shape == (2, backbone.spec.channels, backbone.spec.grid, backbone.spec.grid)
    assert features.dtype == np.float32


def test_channel_count_is_the_sum_of_the_tapped_layers(backbone):
    # Assert: layer2 (512) + layer3 (1024)
    assert backbone.spec.channels == LAYER_CHANNELS["layer2"] + LAYER_CHANNELS["layer3"]
    assert backbone.spec.n_patches == backbone.spec.grid ** 2


def test_the_deeper_tap_is_resampled_onto_the_shallower_grid(backbone):
    """layer3 is 14x14 natively; a method must see it on layer2's 28x28 grid."""
    # Act
    features = backbone.extract(preprocessed(1))

    # Assert
    assert features.shape[-2:] == (28, 28)


def test_a_single_image_may_be_passed_without_a_batch_dimension(backbone):
    # Act
    features = backbone.extract(preprocessed(1)[0])

    # Assert
    assert features.shape[0] == 1


def test_extraction_is_deterministic(backbone):
    """Frozen and in eval mode: no dropout, no batch-norm updates between calls."""
    # Arrange
    images = preprocessed(2)

    # Act / Assert
    assert np.array_equal(backbone.extract(images), backbone.extract(images))


def test_batching_does_not_change_the_features(backbone):
    """The sweep batches; a batch-norm layer left in training mode would not agree."""
    # Arrange
    images = preprocessed(4)
    small = FrozenBackbone(batch_size=1)

    # Act
    batched = backbone.extract(images)
    one_at_a_time = small.extract(images)

    # Assert
    assert np.allclose(batched, one_at_a_time, atol=1e-5)


def test_no_parameter_carries_a_gradient(backbone):
    # Assert
    assert not any(parameter.requires_grad for parameter in backbone._model.parameters())


def test_rejects_an_unknown_layer():
    with pytest.raises(ValueError, match="unknown layer"):
        FrozenBackbone(layers=("layer9",))


def test_rejects_an_empty_layer_list():
    with pytest.raises(ValueError, match="at least one layer"):
        FrozenBackbone(layers=())


def test_rejects_an_unknown_model_name():
    with pytest.raises(ValueError, match="no model"):
        FrozenBackbone(name="not_a_network")


def test_rejects_input_that_is_not_three_channel(backbone):
    with pytest.raises(ValueError, match="expected"):
        backbone.extract(np.zeros((2, 1, 224, 224), dtype=np.float32))

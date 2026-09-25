"""WinCLIP-style method tests: the visual memory, the language score, and their fusion,
on synthetic features whose answer is known by construction."""

from __future__ import annotations

import numpy as np
import pytest

from methods.winclip import VisualMemory, WinCLIP, normalise_blocks, softmax_anomaly

LANGUAGE_DIM = 4
VISUAL = 6
GRID = 3


def text() -> np.ndarray:
    """[normal, anomalous] directions: the first and second language axes."""
    return np.eye(2, LANGUAGE_DIM, dtype=np.float32)


def features(k: int, anomalous_language_at: tuple[int, int] | None = None) -> np.ndarray:
    """k support-like images: identical visual tokens, language pointing at 'normal'."""
    rng = np.random.default_rng(0)
    visual = np.broadcast_to(rng.normal(size=(VISUAL, 1, 1)), (VISUAL, GRID, GRID))
    language = np.zeros((LANGUAGE_DIM, GRID, GRID))
    language[0] = 1.0
    if anomalous_language_at is not None:
        row, column = anomalous_language_at
        language[:, row, column] = 0.0
        language[1, row, column] = 1.0
    single = np.concatenate([visual, language]).astype(np.float32)
    return np.stack([single] * k)


def method(**overrides) -> WinCLIP:
    parameters = {"text": text(), "language_dim": LANGUAGE_DIM, "n_visual_blocks": 2}
    return WinCLIP(**{**parameters, **overrides})


def test_the_language_score_is_the_probability_of_the_anomalous_prompt():
    # Arrange
    language = np.array([[1.0, 0.0, 0, 0], [0.0, 1.0, 0, 0], [1.0, 1.0, 0, 0]])

    # Act
    probability = softmax_anomaly(language, text(), temperature=100.0)

    # Assert
    assert probability[0] < 1e-6
    assert probability[1] > 1 - 1e-6
    assert probability[2] == pytest.approx(0.5)


def test_block_normalisation_gives_each_block_equal_weight():
    # Arrange: second block 100x larger in magnitude
    patches = np.array([[3.0, 4.0, 300.0, 400.0]])

    # Act
    normalised = normalise_blocks(patches, 2)

    # Assert
    assert np.allclose(normalised, [[0.6, 0.8, 0.6, 0.8]] / np.sqrt(2))
    assert np.linalg.norm(normalised) == pytest.approx(1.0)


def test_enrolment_keeps_every_support_patch_as_a_unit_vector():
    # Act
    memory = method().enrol(features(2))

    # Assert
    assert memory.vectors.shape == (2 * GRID * GRID, VISUAL)
    assert np.allclose(np.linalg.norm(memory.vectors, axis=1), 1.0)
    assert memory.k == 2 and memory.grid == (GRID, GRID)


def test_a_single_image_can_be_enrolled_without_a_batch_axis():
    assert method().enrol(features(1)[0]).k == 1


def test_zero_shot_needs_no_support_and_finds_the_anomalous_patch():
    """language_weight = 1 is the k = 0 point: text alone, no reference images."""
    # Arrange
    query = features(1, anomalous_language_at=(1, 2))[0]

    # Act
    anomaly_map = method(language_weight=1.0).score(query, None)

    # Assert
    assert np.unravel_index(np.argmax(anomaly_map), anomaly_map.shape) == (1, 2)
    assert anomaly_map[1, 2] > 0.99


def test_the_visual_branch_scores_zero_on_a_query_identical_to_the_support():
    # Arrange
    support = features(2)

    # Act
    anomaly_map = method(language_weight=0.0).score(support[0], method().enrol(support))

    # Assert
    assert np.allclose(anomaly_map, 0.0, atol=1e-6)


def test_fusion_is_the_weighted_sum_of_the_two_branches():
    # Arrange
    support = features(2)
    query = features(1, anomalous_language_at=(0, 0))[0]
    reference = method().enrol(support)

    # Act
    visual = method(language_weight=0.0).score(query, reference)
    language = method(language_weight=1.0).score(query, reference)
    fused = method(language_weight=0.25).score(query, reference)

    # Assert
    assert np.allclose(fused, 0.75 * visual + 0.25 * language, atol=1e-6)


def test_the_parameters_omit_the_text_embeddings():
    assert "text" not in method().parameters
    assert method().parameters["language_weight"] == 0.5


def test_a_visual_score_without_a_reference_is_rejected():
    with pytest.raises(ValueError, match="support memory"):
        method(language_weight=0.5).score(features(1)[0], None)


def test_a_language_score_without_text_is_rejected():
    with pytest.raises(ValueError, match="text embeddings"):
        WinCLIP(text=None, language_weight=0.5)


def test_text_of_the_wrong_shape_is_rejected():
    with pytest.raises(ValueError, match="text must be"):
        WinCLIP(text=np.ones((3, 512)))


@pytest.mark.parametrize("weight", [-0.1, 1.1])
def test_an_out_of_range_language_weight_is_rejected(weight):
    with pytest.raises(ValueError, match="language_weight"):
        WinCLIP(text=text(), language_dim=LANGUAGE_DIM, language_weight=weight)


def test_features_without_visual_channels_are_rejected():
    with pytest.raises(ValueError, match="expected visual channels"):
        method().enrol(np.ones((1, LANGUAGE_DIM, GRID, GRID), dtype=np.float32))


def test_the_memory_is_a_frozen_record():
    memory = VisualMemory(np.ones((1, 2)), (1, 1), 1)
    with pytest.raises(AttributeError):
        memory.k = 2  # type: ignore[misc]

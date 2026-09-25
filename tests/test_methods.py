"""Method tests (S3/S4). Both methods are pure functions of feature arrays, so the
properties that matter are checked on synthetic features whose answer is known by
construction — no backbone, no dataset, no randomness beyond a fixed seed."""

from __future__ import annotations

import numpy as np
import pytest

from methods import build_method, known_methods
from methods.base import AnomalyMethod, as_patch_matrix, squared_distances
from methods.padim import PaDiM, regularise, select_channels
from methods.patchcore import PatchCore, average_pool, greedy_coreset

GRID = 4
CHANNELS = 6

# Parameters that make the two methods comparable on a 4x4 toy grid: PatchCore's
# defaults (a 10 % coreset of 64 patches, 3x3 pooling) are tuned for a 28x28 grid
# and are degenerate at this size. Both are exercised at their defaults by the
# method-specific tests further down, and by the real sweep.
TOY_PARAMETERS = {
    "patchcore": {"coreset_ratio": 1.0, "neighbourhood": 1},
    "padim": {"dim": CHANNELS},
    # Visual branch only (no text), two trailing "language" channels it must ignore.
    "winclip": {"language_dim": 2, "n_visual_blocks": 1, "language_weight": 0.0},
}


def toy_method(name: str, **overrides):
    """A method configured for the toy grid, with the given parameters changed."""
    return build_method(name, **{**TOY_PARAMETERS[name], **overrides})


def normal_features(k: int, seed: int = 0) -> np.ndarray:
    """k support images of smooth, low-variance "normal" features."""
    rng = np.random.default_rng(seed)
    base = rng.normal(size=(CHANNELS, GRID, GRID))
    return np.stack([base + 0.01 * rng.normal(size=base.shape) for _ in range(k)]).astype(
        np.float32
    )


def defective_query(support: np.ndarray, magnitude: float = 5.0) -> tuple[np.ndarray, tuple]:
    """A query equal to the support mean except at one position, which is far off."""
    query = support.mean(axis=0).copy()
    position = (GRID // 2, GRID // 2)
    query[:, position[0], position[1]] += magnitude
    return query, position


# --- shared interface -------------------------------------------------------


@pytest.mark.parametrize("name", known_methods())
def test_every_method_satisfies_the_interface(name):
    # Act
    method = toy_method(name)

    # Assert
    assert isinstance(method, AnomalyMethod)
    assert method.name == name
    assert isinstance(method.parameters, dict)


@pytest.mark.parametrize("name", known_methods())
def test_score_returns_one_value_per_patch_position(name):
    # Arrange
    support = normal_features(4)
    method = toy_method(name)

    # Act
    anomaly_map = method.score(support[0], method.enrol(support))

    # Assert
    assert anomaly_map.shape == (GRID, GRID)


@pytest.mark.parametrize("name", known_methods())
def test_the_planted_defect_is_the_highest_scoring_patch(name):
    # Arrange
    support = normal_features(4)
    query, position = defective_query(support)
    method = toy_method(name)

    # Act
    anomaly_map = method.score(query, method.enrol(support))

    # Assert
    assert np.unravel_index(np.argmax(anomaly_map), anomaly_map.shape) == position


@pytest.mark.parametrize("name", known_methods())
def test_both_methods_work_at_k_equals_one(name):
    """The few-shot floor: k=1 must produce a finite map, not a division by zero."""
    # Arrange
    support = normal_features(1)
    query, _ = defective_query(support)

    # Act
    method = toy_method(name)
    anomaly_map = method.score(query, method.enrol(support))

    # Assert
    assert np.isfinite(anomaly_map).all()


@pytest.mark.parametrize("name", known_methods())
def test_enrolment_is_deterministic_for_a_fixed_seed(name):
    # Arrange
    support = normal_features(4)
    query, _ = defective_query(support)

    # Act
    first = toy_method(name, seed=7)
    second = toy_method(name, seed=7)
    map_one = first.score(query, first.enrol(support))
    map_two = second.score(query, second.enrol(support))

    # Assert
    assert np.array_equal(map_one, map_two)


def test_build_method_rejects_an_unknown_name():
    with pytest.raises(ValueError, match="unknown method"):
        build_method("magic")


# --- shared helpers ---------------------------------------------------------


def test_as_patch_matrix_keeps_channels_together():
    # Arrange: one image, channel c is constant c everywhere
    features = np.stack([np.full((GRID, GRID), c) for c in range(CHANNELS)])[None]

    # Act
    patches = as_patch_matrix(features)

    # Assert
    assert patches.shape == (GRID * GRID, CHANNELS)
    assert np.array_equal(patches[0], np.arange(CHANNELS))


def test_as_patch_matrix_accepts_a_single_unbatched_image():
    # Act
    patches = as_patch_matrix(np.zeros((CHANNELS, GRID, GRID)))

    # Assert
    assert patches.shape == (GRID * GRID, CHANNELS)


def test_as_patch_matrix_rejects_wrong_rank():
    with pytest.raises(ValueError, match="expected"):
        as_patch_matrix(np.zeros((3, 3)))


def test_squared_distances_matches_the_explicit_form():
    # Arrange
    rng = np.random.default_rng(0)
    queries, bank = rng.normal(size=(5, 3)), rng.normal(size=(4, 3))

    # Act
    computed = squared_distances(queries, bank)
    expected = ((queries[:, None, :] - bank[None, :, :]) ** 2).sum(axis=-1)

    # Assert
    assert np.allclose(computed, expected, atol=1e-5)


def test_squared_distances_are_never_negative():
    """The expanded form can go below zero in floating point on coincident points."""
    # Arrange
    point = np.full((1, 64), 1e3, dtype=np.float32)

    # Act / Assert
    assert squared_distances(point, point).min() >= 0.0


def test_squared_distances_rejects_dimension_mismatch():
    with pytest.raises(ValueError, match="dimension mismatch"):
        squared_distances(np.zeros((2, 3)), np.zeros((2, 4)))


# --- PatchCore --------------------------------------------------------------


def test_average_pool_of_a_constant_map_is_unchanged():
    """Edge padding, not zero padding: a constant field must stay constant."""
    # Arrange
    features = np.full((1, 2, GRID, GRID), 3.0, dtype=np.float32)

    # Act / Assert
    assert np.allclose(average_pool(features, 3), 3.0)


def test_average_pool_spreads_an_isolated_spike_over_its_neighbourhood():
    # Arrange
    features = np.zeros((1, 1, 5, 5), dtype=np.float32)
    features[0, 0, 2, 2] = 9.0

    # Act
    pooled = average_pool(features, 3)

    # Assert: the spike's mass is shared by the 9 cells of its neighbourhood
    assert pooled[0, 0, 2, 2] == pytest.approx(1.0)
    assert pooled[0, 0, 1, 2] == pytest.approx(1.0)
    assert pooled[0, 0, 0, 0] == pytest.approx(0.0)


def test_average_pool_of_size_one_is_the_identity():
    # Arrange
    features = np.arange(2 * 3 * GRID * GRID, dtype=np.float32).reshape(2, 3, GRID, GRID)

    # Act / Assert
    assert np.array_equal(average_pool(features, 1), features)


def test_average_pool_rejects_an_even_neighbourhood():
    with pytest.raises(ValueError, match="odd"):
        average_pool(np.zeros((1, 1, GRID, GRID)), 2)


def test_average_pool_rejects_a_map_that_is_not_a_feature_grid():
    with pytest.raises(ValueError, match="expected"):
        average_pool(np.zeros((GRID, GRID)), 3)


def test_coreset_selects_the_spread_out_points_not_a_random_sample():
    # Arrange: three tight clumps; a covering subset must take one from each
    rng = np.random.default_rng(0)
    centres = np.array([[0.0, 0.0], [50.0, 0.0], [0.0, 50.0]])
    points = np.repeat(centres, 20, axis=0) + rng.normal(0, 0.1, size=(60, 2))

    # Act
    indices = greedy_coreset(points, 3, np.random.default_rng(1), projection_dim=2)
    chosen = points[indices]

    # Assert: one point near each centre
    for centre in centres:
        assert np.abs(chosen - centre).sum(axis=1).min() < 1.0


def test_coreset_returns_every_point_when_the_ratio_keeps_all():
    # Arrange
    points = np.random.default_rng(0).normal(size=(10, 4))

    # Act / Assert
    assert len(greedy_coreset(points, 10, np.random.default_rng(0), 4)) == 10


def test_coreset_keeps_at_least_one_point():
    # Arrange: a ratio small enough to round to zero patches
    support = normal_features(1)

    # Act
    bank = PatchCore(coreset_ratio=1e-6).enrol(support)

    # Assert
    assert bank.size >= 1


def test_memory_bank_records_the_reduction_it_performed():
    # Arrange
    support = normal_features(8)

    # Act
    bank = PatchCore(coreset_ratio=0.5).enrol(support)

    # Assert
    assert bank.n_patches_seen == 8 * GRID * GRID
    assert bank.size == pytest.approx(0.5 * bank.n_patches_seen, abs=1)


def test_patchcore_scores_a_memorised_patch_at_almost_zero():
    """A query patch that is in the bank is by definition normal."""
    # Arrange
    support = normal_features(4)
    method = PatchCore(coreset_ratio=1.0)

    # Act
    anomaly_map = method.score(support[0], method.enrol(support))

    # Assert
    assert anomaly_map.max() < 0.1


def test_patchcore_rejects_an_invalid_coreset_ratio():
    with pytest.raises(ValueError, match="coreset_ratio"):
        PatchCore(coreset_ratio=0.0)


# --- PaDiM ------------------------------------------------------------------


def test_select_channels_is_deterministic_and_within_range():
    # Act
    first = select_channels(20, 5, seed=3)
    second = select_channels(20, 5, seed=3)

    # Assert
    assert np.array_equal(first, second)
    assert len(set(first.tolist())) == 5
    assert first.max() < 20


def test_select_channels_cannot_ask_for_more_than_exist():
    assert len(select_channels(4, 10, seed=0)) == 4


def test_regularisation_makes_the_k_equals_one_covariance_invertible():
    """At k=1 the sample covariance is exactly zero and has no inverse at all."""
    # Arrange
    covariance = np.zeros((1, 3, 3))

    # Act
    regularised = regularise(covariance, shrinkage=0.1, epsilon=1e-4)

    # Assert
    assert np.linalg.matrix_rank(regularised[0]) == 3


def test_regularisation_shrinks_towards_the_scaled_identity():
    # Arrange: a covariance with all its variance in one direction
    covariance = np.diag([3.0, 0.0, 0.0])[None]

    # Act
    regularised = regularise(covariance, shrinkage=0.5, epsilon=0.0)[0]

    # Assert: half the trace is redistributed evenly (trace/d = 1 per dimension)
    assert regularised[0, 0] == pytest.approx(0.5 * 3.0 + 0.5 * 1.0)
    assert regularised[1, 1] == pytest.approx(0.5 * 1.0)


def test_padim_reports_rank_deficiency_when_k_is_below_the_dimension():
    # Arrange
    support = normal_features(4)

    # Act
    model = PaDiM(dim=CHANNELS).enrol(support)

    # Assert: 4 samples cannot fill a 6-dimensional covariance
    assert model.is_rank_deficient


def test_padim_is_not_rank_deficient_once_k_exceeds_the_dimension():
    # Arrange
    support = normal_features(5)

    # Act
    model = PaDiM(dim=2).enrol(support)

    # Assert
    assert not model.is_rank_deficient


def test_padim_scores_the_support_mean_lower_than_a_shifted_query():
    # Arrange
    support = normal_features(6)
    method = PaDiM(dim=CHANNELS)
    model = method.enrol(support)

    # Act
    normal_score = method.score(support.mean(axis=0), model).max()
    shifted_score = method.score(support.mean(axis=0) + 3.0, model).max()

    # Assert
    assert shifted_score > normal_score


def test_padim_rejects_a_query_on_a_different_grid():
    # Arrange
    model = PaDiM(dim=CHANNELS).enrol(normal_features(2))

    # Act / Assert
    with pytest.raises(ValueError, match="grid mismatch"):
        PaDiM(dim=CHANNELS).score(np.zeros((CHANNELS, GRID + 1, GRID)), model)


def test_padim_accepts_a_single_support_image_without_a_batch_dimension():
    """k = 1 is the point of the project; passing it unbatched must still work."""
    # Arrange
    single = normal_features(1)[0]

    # Act
    model = PaDiM(dim=CHANNELS).enrol(single)

    # Assert
    assert model.k == 1


def test_padim_rejects_support_of_the_wrong_rank():
    with pytest.raises(ValueError, match="expected"):
        PaDiM().enrol(np.zeros((GRID, GRID)))


def test_padim_rejects_an_out_of_range_shrinkage():
    with pytest.raises(ValueError, match="shrinkage"):
        PaDiM(shrinkage=1.5)


def test_padim_rejects_a_non_positive_epsilon():
    with pytest.raises(ValueError, match="epsilon"):
        PaDiM(epsilon=0.0)

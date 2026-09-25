"""Few-shot support-set sampler tests (S9 and the S11 stress variants)."""

from __future__ import annotations

from pathlib import Path

import pytest

from data.mvtec import MVTecAD
from data.sampler import (
    FULL,
    draw_support,
    draw_support_contaminated,
    draw_support_with_gap,
    id_block_group,
    resolve_k,
)


@pytest.fixture
def normals(mvtec_root: Path):
    return MVTecAD(mvtec_root).train_samples("widget")


@pytest.fixture
def anomalies(mvtec_root: Path):
    return [s for s in MVTecAD(mvtec_root).test_samples("widget") if s.label == 1]


def test_draws_exactly_k_distinct_images(normals):
    # Act
    support = draw_support(normals, k=4, draw=0, seed=1234)

    # Assert
    assert len(support) == 4
    assert len({s.path for s in support.samples}) == 4


def test_same_coordinates_give_the_same_support_set(normals):
    # Act
    first = draw_support(normals, k=4, draw=2, seed=1234)
    second = draw_support(normals, k=4, draw=2, seed=1234)

    # Assert: reproducibility is the whole point of seeding the sampler
    assert [s.path for s in first.samples] == [s.path for s in second.samples]


def test_different_draws_give_different_support_sets(normals):
    # Act
    draws = [draw_support(normals, k=2, draw=d, seed=1234) for d in range(5)]

    # Assert: N draws must actually vary, or the reported std is meaningless
    distinct = {tuple(s.path for s in d.samples) for d in draws}
    assert len(distinct) > 1


def test_different_seeds_give_different_support_sets(normals):
    # Act
    first = draw_support(normals, k=3, draw=0, seed=1)
    second = draw_support(normals, k=3, draw=0, seed=2)

    # Assert
    assert [s.path for s in first.samples] != [s.path for s in second.samples]


def test_draw_is_independent_of_k_ordering(normals):
    """Draw 3 of k=4 is the same set whether or not k=1 ran first."""
    # Act
    before = draw_support(normals, k=4, draw=3, seed=7)
    _ = [draw_support(normals, k=1, draw=d, seed=7) for d in range(4)]
    after = draw_support(normals, k=4, draw=3, seed=7)

    # Assert
    assert [s.path for s in before.samples] == [s.path for s in after.samples]


def test_full_uses_the_entire_train_split(normals):
    # Act
    support = draw_support(normals, k=FULL, draw=0, seed=1234)

    # Assert: the ceiling run each few-shot curve is normalised against
    assert len(support) == len(normals)


def test_k_larger_than_the_pool_is_rejected(normals):
    with pytest.raises(ValueError, match="cannot draw"):
        draw_support(normals, k=len(normals) + 1, draw=0, seed=1234)


def test_empty_pool_is_rejected():
    with pytest.raises(ValueError, match="empty normal pool"):
        draw_support([], k=1, draw=0, seed=1234)


def test_resolve_k_rejects_non_positive():
    with pytest.raises(ValueError, match="k must be"):
        resolve_k(0, 10)


def test_provenance_records_everything_needed_to_reproduce(normals):
    # Act
    support = draw_support(normals, k=2, draw=1, seed=99)
    provenance = support.provenance

    # Assert
    assert provenance["k"] == 2
    assert provenance["draw"] == 1
    assert provenance["mode"] == "uniform"
    assert len(provenance["image_ids"]) == 2


def test_support_set_is_immutable(normals):
    # Act
    support = draw_support(normals, k=2, draw=0, seed=1)

    # Assert: a result file must describe the support set that was actually used
    with pytest.raises(AttributeError):
        support.k = 8


def test_coverage_gap_excludes_the_held_out_group(normals):
    # Arrange: the fixture has 12 train images -> blocks of 5 give three groups
    def group(sample):
        return id_block_group(sample, block_size=5)

    # Act
    support = draw_support_with_gap(
        normals, k=3, draw=0, seed=1234, group_of=group, held_out_group="block0"
    )

    # Assert
    assert support.held_out_group == "block0"
    assert all(group(s) != "block0" for s in support.samples)


def test_coverage_gap_picks_a_group_when_none_is_named(normals):
    # Act
    support = draw_support_with_gap(
        normals, k=3, draw=0, seed=1234, group_of=lambda s: id_block_group(s, 5)
    )

    # Assert
    assert support.held_out_group is not None
    assert support.mode == "coverage_gap"


def test_coverage_gap_needs_more_than_one_group(normals):
    with pytest.raises(ValueError, match="at least two groups"):
        draw_support_with_gap(normals, k=2, draw=0, seed=1, group_of=lambda s: "only")


def test_contaminated_support_contains_the_defective_images(normals, anomalies):
    # Act
    support = draw_support_contaminated(normals, anomalies, k=4, draw=0, seed=1234)

    # Assert
    assert len(support) == 4
    assert len(support.contaminated) == 1
    assert support.contaminated[0].label == 1
    assert sum(s.label for s in support.samples) == 1


def test_contamination_cannot_exceed_k(normals, anomalies):
    with pytest.raises(ValueError, match="cannot contaminate"):
        draw_support_contaminated(normals, anomalies, k=1, draw=0, seed=1, n_contaminated=2)


def test_contamination_requires_anomalies(normals):
    with pytest.raises(ValueError, match="no anomalous images"):
        draw_support_contaminated(normals, [], k=2, draw=0, seed=1)


def test_contamination_count_must_be_positive(normals, anomalies):
    with pytest.raises(ValueError, match="n_contaminated"):
        draw_support_contaminated(normals, anomalies, k=2, draw=0, seed=1, n_contaminated=0)


def test_id_block_group_blocks_are_contiguous(normals):
    # Act
    groups = [id_block_group(s, block_size=5) for s in normals]

    # Assert: ids 000-004 -> block0, 005-009 -> block1, ...
    assert groups[:5] == ["block0"] * 5
    assert groups[5:10] == ["block1"] * 5


def test_coverage_gap_rejects_an_empty_pool():
    with pytest.raises(ValueError, match="empty normal pool"):
        draw_support_with_gap([], k=1, draw=0, seed=1, group_of=id_block_group)

"""S11a/S11b — the stress-test runs: what the support set is, and what to measure.

The samplers for both stress tests were built in Task 4 (`data/sampler.py`); this
file is the part that turns a drawn support set into a *measurement* (week03 §S11):

  * **Coverage gap (S11a, Gap G6).** One group of normals — a block of consecutive
    file ids, the acquisition-order proxy for "one mode of normal variation" — is
    withheld from the support set. The decision threshold is fitted on validation
    normals from the *covered* groups only, and the measurement is the false-alarm
    rate on the withheld group's normals at that threshold. If the group is a real
    mode of variation, its false-alarm rate rises above the covered normals'.
  * **Contamination (S11b, Gap G8).** n of the k "normal" references are defective
    test images. They are removed from the query set, and the measurement is the
    recall on test images of the *same defect type* — the ones the method has now
    been told are normal — against the recall on every other defect type.

Uniform runs pass through unchanged: the same code path, no extra measurement.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from config.schema import RunConfig
from data.mvtec import Sample
from data.sampler import (
    SupportSet,
    draw_support,
    draw_support_contaminated,
    draw_support_with_gap,
    id_block_group,
)

MAX_HELD_OUT = 20  # withheld-group normals scored per coverage-gap run


def draw_for_mode(
    normals: Sequence[Sample], test: Sequence[Sample], k: int | str, draw: int,
    config: RunConfig, category: str,
) -> SupportSet:
    """The support set for this run's `support_mode`."""
    if config.support_mode == "coverage_gap":
        return draw_support_with_gap(normals, k, draw, config.seed, id_block_group,
                                     category=category)
    if config.support_mode == "contaminated":
        anomalies = [sample for sample in test if sample.label == 1]
        return draw_support_contaminated(normals, anomalies, k, draw, config.seed,
                                         config.n_contaminated, category)
    return draw_support(normals, k, draw, config.seed, category)


def query_samples(test: Sequence[Sample], support: SupportSet) -> list[Sample]:
    """The test split minus any image used as a (contaminating) reference."""
    used = {sample.image_id for sample in support.contaminated}
    return [sample for sample in test if sample.image_id not in used]


def validation_pool(normals: Sequence[Sample], support: SupportSet) -> list[Sample]:
    """Normals the threshold may be fitted on: never the withheld group."""
    if support.held_out_group is None:
        return list(normals)
    return [s for s in normals if id_block_group(s) != support.held_out_group]


def held_out_normals(normals: Sequence[Sample], support: SupportSet) -> list[Sample]:
    """The withheld group's normals, capped at `MAX_HELD_OUT` (first ids first)."""
    if support.held_out_group is None:
        return []
    group = [s for s in normals if id_block_group(s) == support.held_out_group]
    return group[:MAX_HELD_OUT]


def contamination_metrics(
    support: SupportSet, queries: Sequence[Sample], rejected: np.ndarray
) -> dict[str, float]:
    """Recall on the contaminant's defect type(s) against recall on the other types."""
    rejected = np.asarray(rejected, dtype=bool)
    poisoned = {sample.defect_type for sample in support.contaminated}
    same = np.array([q.label == 1 and q.defect_type in poisoned for q in queries])
    other = np.array([q.label == 1 and q.defect_type not in poisoned for q in queries])
    return {
        "same_type_recall": float(rejected[same].mean()) if same.any() else float("nan"),
        "other_type_recall": float(rejected[other].mean()) if other.any() else float("nan"),
        "n_same_type": int(same.sum()),
    }


def coverage_metrics(held_out_scores: np.ndarray, threshold: float) -> dict[str, float]:
    """False-alarm rate on the withheld group's normals at the fitted threshold."""
    scores = np.asarray(held_out_scores, dtype=np.float64)
    if scores.size == 0:
        return {"held_out_fpr": float("nan"), "n_held_out": 0}
    return {"held_out_fpr": float((scores > threshold).mean()), "n_held_out": int(scores.size)}

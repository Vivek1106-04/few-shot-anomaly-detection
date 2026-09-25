"""S7 — the decision stage: image score -> PASS / REJECT.

Week03 §S7 fixes the rule this file implements: the threshold is set on **normal
validation scores only**, at the line's operating point, and never tuned on test
defects. A one-class system that picks its threshold by looking at defects has
quietly stopped being one-class, and the recall it then reports is not a recall any
deployment could achieve.

Where the validation normals come from. MVTec's train split is normal-only and far
larger than any few-shot support set, so the images *not* drawn into the support
set are held-out normals the method has never seen. A seeded subset of them is the
validation set: at k = 4 on `bottle` that is up to `N_VALIDATION` of the other 205
normals. At k = full nothing is held out, and the decision is reported as
undefined (NaN) rather than being fitted on the support set itself.

The threshold rule. With n validation scores and a false-alarm budget b, the
threshold is the ceil((1 - b) n)-th smallest score and an image is rejected when it
scores *strictly above* it, so at most floor(b n) validation normals are rejected:
the empirical validation FPR is inside the budget by construction. Whether that
holds on the test normals too is the measurement — the threshold did not see them.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from data.mvtec import Sample

N_VALIDATION = 20  # held-out normals per run: enough for a 0.10 quantile, cheap to score
PIXEL_QUANTILE = 0.995  # the defect-mask threshold: top 0.5 % of normal pixels
TARGET_RECALL = 0.95  # SPEC §2 operating point: recall >= 0.95 at FPR <= the budget
QUANTILE = "quantile"
CONFORMAL = "conformal"
RULES = (QUANTILE, CONFORMAL)


def fit_threshold(normal_scores: np.ndarray, fpr_budget: float, rule: str = QUANTILE) -> float:
    """A score threshold fitted on normal validation scores.

    `quantile` — the ceil((1 - b) n)-th smallest score: rejects at most b of the
    *validation* normals. For a new normal, though, the chance of exceeding the r-th
    of n exchangeable scores is (n + 1 - r) / (n + 1): 3/21 = 0.143 at n = 20, b = 0.1,
    so this rule overshoots the budget on unseen data (measured in the week06 report).
    `conformal` — the ceil((1 - b)(n + 1))-th smallest (split-conformal rank): the
    expected FPR on a new normal is then <= b, which is the guarantee S7 is for.
    With too few normals for that rank, the largest normal is used.
    """
    if not 0.0 < fpr_budget <= 1.0:
        raise ValueError(f"fpr_budget must be in (0, 1], got {fpr_budget}")
    if rule not in RULES:
        raise ValueError(f"unknown threshold rule {rule!r}; known: {RULES}")
    scores = np.sort(np.asarray(normal_scores, dtype=np.float64).ravel())
    if scores.size == 0:
        raise ValueError("a threshold needs at least one normal validation score")
    size = scores.size + (1 if rule == CONFORMAL else 0)
    rank = min(max(math.ceil((1.0 - fpr_budget) * size), 1), scores.size)
    return float(scores[rank - 1])


@dataclass(frozen=True)
class Decision:
    """A fitted S7 stage: the image threshold, and the pixel threshold for S6b masks."""

    threshold: float
    fpr_budget: float
    n_validation: int
    pixel_threshold: float
    low: float  # smallest normal pixel score: the floor of the display range

    def verdict(self, score: float) -> str:
        return "REJECT" if score > self.threshold else "PASS"

    def as_dict(self) -> dict[str, float | int]:
        return {
            "threshold": self.threshold,
            "fpr_budget": self.fpr_budget,
            "n_validation": self.n_validation,
            "pixel_threshold": self.pixel_threshold,
            "low": self.low,
        }


def fit_decision(normal_maps: np.ndarray, fpr_budget: float, rule: str = QUANTILE) -> Decision:
    """Fit S7 from the post-processed maps of the validation normals."""
    maps = np.asarray(normal_maps, dtype=np.float64)
    if maps.ndim != 3 or maps.shape[0] == 0:
        raise ValueError(f"need a non-empty (n, H, W) stack of validation maps, got {maps.shape}")
    image_scores = maps.reshape(maps.shape[0], -1).max(axis=1)  # the S6 score
    return Decision(
        threshold=fit_threshold(image_scores, fpr_budget, rule),
        fpr_budget=float(fpr_budget),
        n_validation=int(maps.shape[0]),
        pixel_threshold=float(np.quantile(maps, PIXEL_QUANTILE)),
        low=float(maps.min()),
    )


@dataclass(frozen=True)
class OperatingPoint:
    """What the fitted threshold does on the test split."""

    recall: float
    fpr: float
    meets_target: bool

    @classmethod
    def undefined(cls) -> OperatingPoint:
        return cls(recall=math.nan, fpr=math.nan, meets_target=False)


def operating_point(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    fpr_budget: float,
    target_recall: float = TARGET_RECALL,
) -> OperatingPoint:
    """Recall on defective and false-alarm rate on normal test images, at `threshold`."""
    labels = np.asarray(labels).ravel()
    rejected = np.asarray(scores, dtype=np.float64).ravel() > threshold
    recall = float(rejected[labels == 1].mean())
    fpr = float(rejected[labels == 0].mean())
    return OperatingPoint(recall, fpr, recall >= target_recall and fpr <= fpr_budget)


def validation_normals(
    pool: Sequence[Sample], support: Sequence[Sample], n_max: int, seed: int
) -> tuple[Sample, ...]:
    """Up to `n_max` train normals outside the support set, drawn with `seed`."""
    used = {sample.image_id for sample in support}
    remaining = [sample for sample in pool if sample.image_id not in used]
    if len(remaining) <= n_max:
        return tuple(remaining)
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(len(remaining), size=n_max, replace=False))
    return tuple(remaining[int(i)] for i in indices)

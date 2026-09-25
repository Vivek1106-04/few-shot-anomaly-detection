"""S10 — the metric set, one implementation shared by every method.

Week03 §S10 makes this a first-class component: because every method is scored by
this file and nothing else, a difference between two methods cannot come from a
difference in how they were measured (Gap G1). Implemented directly on numpy —
no scikit-learn — so the definitions used are visible and testable rather than
delegated, which matters for PRO, where published implementations differ.

  I-AUROC      image-level ranking
  P-AUROC      pixel-level ranking
  PRO          per-region overlap, AUC integrated to FPR 0.30 and normalised
  pixel-AP     average precision, honest under the pixel-class imbalance
  F1-max       best achievable F1 over thresholds
  recall@FPR   recall at the deployable operating point (FPR <= 0.10)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

PRO_FPR_LIMIT = 0.30  # the integration limit used by MVTec AD and every paper since
DEFAULT_FPR_BUDGET = 0.10  # the line's operating point (week01 §2: recall >= 0.95 here)
PRO_THRESHOLD_STEPS = 200


def _as_float(scores: np.ndarray) -> np.ndarray:
    return np.asarray(scores, dtype=np.float64).ravel()


def _as_binary(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels).ravel()
    unique = np.unique(labels)
    if not np.isin(unique, (0, 1)).all():
        raise ValueError(f"labels must be 0/1, got values {unique[:5]}")
    return labels.astype(np.int8)


def auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Area under the ROC curve, via the rank-sum (Mann-Whitney U) identity.

    Ties are handled by averaging their ranks, which is what makes this agree with
    the trapezoidal ROC integral rather than being optimistic on tied scores — and
    ties are common at k=1, where many patches share the same nearest neighbour.
    """
    labels = _as_binary(labels)
    scores = _as_float(scores)
    if labels.size != scores.size:
        raise ValueError(f"size mismatch: {labels.size} labels vs {scores.size} scores")
    n_pos = int(labels.sum())
    n_neg = labels.size - n_pos
    if n_pos == 0 or n_neg == 0:
        raise ValueError("AUROC needs both classes present")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(scores.size, dtype=np.float64)
    ranks[order] = np.arange(1, scores.size + 1, dtype=np.float64)
    # average the ranks inside each tied group
    sorted_scores = scores[order]
    start = 0
    for end in range(1, sorted_scores.size + 1):
        if end == sorted_scores.size or sorted_scores[end] != sorted_scores[start]:
            if end - start > 1:
                ranks[order[start:end]] = ranks[order[start:end]].mean()
            start = end
    return float((ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    """Average precision: sum over thresholds of (recall step) * precision.

    The step form, not the interpolated one, so a degenerate score map cannot be
    flattered by interpolation.
    """
    labels = _as_binary(labels)
    scores = _as_float(scores)
    n_pos = int(labels.sum())
    if n_pos == 0:
        raise ValueError("average precision needs at least one positive")
    order = np.argsort(-scores, kind="mergesort")
    hits = labels[order].astype(np.float64)
    true_positives = np.cumsum(hits)
    precision = true_positives / np.arange(1, hits.size + 1)
    return float((precision * hits).sum() / n_pos)


def roc_curve(labels: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(fpr, tpr, threshold) at every distinct score, highest threshold first."""
    labels = _as_binary(labels)
    scores = _as_float(scores)
    order = np.argsort(-scores, kind="mergesort")
    sorted_labels = labels[order]
    sorted_scores = scores[order]
    distinct = np.r_[np.flatnonzero(np.diff(sorted_scores)), sorted_scores.size - 1]
    true_positives = np.cumsum(sorted_labels)[distinct]
    false_positives = np.cumsum(1 - sorted_labels)[distinct]
    n_pos = max(int(labels.sum()), 1)
    n_neg = max(int(labels.size - labels.sum()), 1)
    return false_positives / n_neg, true_positives / n_pos, sorted_scores[distinct]


def f1_max(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    """Best F1 over all thresholds, and the threshold that achieves it."""
    labels = _as_binary(labels)
    scores = _as_float(scores)
    order = np.argsort(-scores, kind="mergesort")
    sorted_labels = labels[order].astype(np.float64)
    true_positives = np.cumsum(sorted_labels)
    predicted = np.arange(1, sorted_labels.size + 1, dtype=np.float64)
    n_pos = max(float(labels.sum()), 1.0)
    precision = true_positives / predicted
    recall = true_positives / n_pos
    with np.errstate(invalid="ignore", divide="ignore"):
        f1 = np.where(precision + recall > 0, 2 * precision * recall / (precision + recall), 0.0)
    best = int(np.argmax(f1))
    return float(f1[best]), float(scores[order][best])


def recall_at_fpr(
    labels: np.ndarray, scores: np.ndarray, fpr_budget: float = DEFAULT_FPR_BUDGET
) -> tuple[float, float]:
    """Recall at the highest threshold whose FPR still fits the budget.

    AUROC is threshold-free; a production line is not. This is the number the
    success criteria in SPEC.md §2 are written against (recall >= 0.95 at FPR <= 0.10),
    and the threshold it returns is the τ that S7 would deploy.
    """
    if not 0.0 < fpr_budget <= 1.0:
        raise ValueError(f"fpr_budget must be in (0, 1], got {fpr_budget}")
    fpr, tpr, thresholds = roc_curve(labels, scores)
    allowed = fpr <= fpr_budget
    if not allowed.any():
        return 0.0, float(thresholds[0])
    index = int(np.flatnonzero(allowed)[-1])
    return float(tpr[index]), float(thresholds[index])


def pro_score(
    masks: np.ndarray, maps: np.ndarray, fpr_limit: float = PRO_FPR_LIMIT
) -> float:
    """Per-region overlap (PRO), integrated to `fpr_limit` and normalised to [0,1].

    At each threshold, every connected ground-truth defect region contributes the
    fraction of itself that is covered, and those fractions are averaged *per region*
    — so a 20-px scratch counts as much as a large stain, which is exactly what
    P-AUROC's pixel average hides (Gap G3). The curve is integrated against the
    false-positive rate over normal pixels and divided by `fpr_limit`, so a perfect
    localiser scores 1.0.
    """
    masks = np.asarray(masks)
    maps = np.asarray(maps, dtype=np.float64)
    if masks.shape != maps.shape:
        raise ValueError(f"shape mismatch: masks {masks.shape} vs maps {maps.shape}")
    if masks.ndim == 2:
        masks, maps = masks[None], maps[None]

    # One entry per connected defect region: the scores of that region's pixels.
    region_scores: list[np.ndarray] = []
    for image_index, mask in enumerate(masks):
        count, labelled = cv2.connectedComponents((mask > 0).astype(np.uint8))
        for label in range(1, count):
            region_scores.append(maps[image_index][labelled == label])
    if not region_scores:
        raise ValueError("PRO needs at least one defect region")

    normal_pixels = maps[masks == 0]
    if normal_pixels.size == 0:
        raise ValueError("PRO needs normal pixels to measure the false-positive rate")

    thresholds = np.quantile(maps, np.linspace(0.0, 1.0, PRO_THRESHOLD_STEPS))[::-1]

    fprs, pros = [], []
    for threshold in thresholds:
        fpr = float((normal_pixels >= threshold).mean())
        if fpr > fpr_limit:
            break
        overlaps = [float((scores >= threshold).mean()) for scores in region_scores]
        fprs.append(fpr)
        pros.append(float(np.mean(overlaps)))
    if not fprs:
        # Not even the highest threshold keeps the false-positive rate inside the
        # limit — the map carries no usable localisation signal at this budget.
        return 0.0
    # The curve is a step function of the threshold, so the stretch between the
    # last achievable FPR and the integration limit holds the last overlap value.
    if fprs[-1] < fpr_limit:
        fprs.append(fpr_limit)
        pros.append(pros[-1])
    return float(np.trapezoid(pros, fprs) / fpr_limit)


@dataclass(frozen=True)
class Result:
    """Every metric for one (method, category, k, draw) run."""

    image_auroc: float
    pixel_auroc: float
    pro: float
    pixel_ap: float
    image_f1_max: float
    recall_at_fpr: float
    threshold: float
    # Filled in by the runner once S6a's robust score and S7's threshold exist; NaN
    # where they are undefined (no held-out normals at k = full).
    image_auroc_robust: float = math.nan
    decision_recall: float = math.nan
    decision_fpr: float = math.nan

    def as_dict(self) -> dict[str, float]:
        return {
            "I-AUROC": self.image_auroc,
            "P-AUROC": self.pixel_auroc,
            "PRO": self.pro,
            "pixel-AP": self.pixel_ap,
            "F1-max": self.image_f1_max,
            f"recall@FPR<={DEFAULT_FPR_BUDGET}": self.recall_at_fpr,
            "threshold": self.threshold,
            "I-AUROC(top1%)": self.image_auroc_robust,
            "recall@tau": self.decision_recall,
            "FPR@tau": self.decision_fpr,
        }


def evaluate(
    image_labels: np.ndarray,
    image_scores: np.ndarray,
    masks: np.ndarray,
    maps: np.ndarray,
    fpr_budget: float = DEFAULT_FPR_BUDGET,
) -> Result:
    """Compute the whole metric set for one run."""
    pixel_labels = (np.asarray(masks) > 0).astype(np.int8).ravel()
    pixel_scores = np.asarray(maps, dtype=np.float64).ravel()
    recall, threshold = recall_at_fpr(image_labels, image_scores, fpr_budget)
    return Result(
        image_auroc=auroc(image_labels, image_scores),
        pixel_auroc=auroc(pixel_labels, pixel_scores),
        pro=pro_score(masks, maps),
        pixel_ap=average_precision(pixel_labels, pixel_scores),
        image_f1_max=f1_max(image_labels, image_scores)[0],
        recall_at_fpr=recall,
        threshold=threshold,
    )


def aggregate(results: list[Result]) -> dict[str, tuple[float, float]]:
    """Mean and standard deviation over the N draws of one (method, category, k).

    Returning the spread, not just the mean, is the point of Gap G2: a 1-shot number
    without its variance across draws is not reproducible information.
    """
    if not results:
        raise ValueError("no results to aggregate")
    keys = results[0].as_dict().keys()
    stacked = {key: np.array([r.as_dict()[key] for r in results]) for key in keys}
    return {key: (float(values.mean()), float(values.std(ddof=0)))
            for key, values in stacked.items()}

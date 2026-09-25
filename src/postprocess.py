"""S5/S6 — anomaly-map post-processing and the image-level score.

Every method's S4 output is a small patch grid (28x28 for a 224x224 input). Turning
that into the two things the project promises — a pixel-level anomaly map and one
image-level score — happens **here and only here**, for every method identically.
That is deliberate: post-processing choices (how much smoothing, how the map is
reduced to a scalar) move the metrics by more than the difference between two
methods, so if each method brought its own, the comparison would measure the
post-processing rather than the method (Gap G1).

  S5  upsample to input resolution -> Gaussian smoothing -> optional normalisation
  S6  image score = maximum of the smoothed map (robust variant: top-1 % mean)
  S6b binary mask and overlay for the qualitative outputs
"""

from __future__ import annotations

import cv2
import numpy as np

from data.transforms import CROP_SIZE

SMOOTH_SIGMA = 4.0  # shared with the week04 floor baseline, so the two are comparable
SMOOTH_KSIZE = (0, 0)  # OpenCV derives the kernel size from sigma
OVERLAY_ALPHA = 0.5
TOP_FRACTION = 0.01  # S6a robust score: mean of the top 1 % of pixels


def upsample_map(patch_map: np.ndarray, size: int = CROP_SIZE) -> np.ndarray:
    """Patch grid -> pixel grid, bilinear.

    Bilinear and not nearest: the score is a continuous quantity sampled on a
    coarse grid, and blocky edges from nearest-neighbour upsampling would both look
    like structure that is not there and add spurious high-frequency boundaries to
    every localisation metric.
    """
    patch_map = np.asarray(patch_map, dtype=np.float32)
    if patch_map.ndim != 2:
        raise ValueError(f"expected a 2-D patch map, got shape {patch_map.shape}")
    return cv2.resize(patch_map, (size, size), interpolation=cv2.INTER_LINEAR)


def smooth(anomaly_map: np.ndarray, sigma: float = SMOOTH_SIGMA) -> np.ndarray:
    """Gaussian smoothing of the upsampled map.

    A single patch responding strongly is usually noise; a defect covers a
    neighbourhood. Smoothing is what makes the PRO curve reward contiguous regions
    instead of isolated pixels, and the sigma is a shared constant rather than a
    per-method knob for the reason given in this module's docstring.
    """
    if sigma <= 0:
        return np.asarray(anomaly_map, dtype=np.float32)
    return cv2.GaussianBlur(np.asarray(anomaly_map, dtype=np.float32), SMOOTH_KSIZE, sigma)


def to_pixel_map(patch_map: np.ndarray, size: int = CROP_SIZE, sigma: float = SMOOTH_SIGMA
                 ) -> np.ndarray:
    """The whole of S5: upsample, then smooth."""
    return smooth(upsample_map(patch_map, size), sigma)


def image_score(anomaly_map: np.ndarray) -> float:
    """S6 — one scalar per image: the maximum of the smoothed map.

    An image is anomalous if *any* part of it is, so the maximum is the definition
    that matches the task; a mean would let a large normal area outvote a small
    defect, which is exactly the small-defect blindness Gap G3 is about.
    """
    return float(np.asarray(anomaly_map).max())


def robust_image_score(anomaly_map: np.ndarray, fraction: float = TOP_FRACTION) -> float:
    """S6a robust variant: the mean of the top `fraction` of pixels (week03 §S6).

    `max` is a single-pixel statistic, and at k = 1 one noisy patch can decide an
    image. Averaging the top 1 % keeps the "any part of it" semantics — a 1 % region
    is ~500 pixels at 224 x 224, smaller than most defects — while needing more than
    one pixel to agree. Reported next to the max score, not instead of it.
    """
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")
    values = np.asarray(anomaly_map, dtype=np.float64).ravel()
    count = max(int(round(fraction * values.size)), 1)
    return float(np.partition(values, values.size - count)[-count:].mean())


def normalise(anomaly_map: np.ndarray, low: float, high: float) -> np.ndarray:
    """Map [low, high] to [0, 1] for display and for cross-category thresholds.

    Applied only to the qualitative outputs and to the S7 decision stage, never
    before the metrics: AUROC and PRO are rank-based, so a monotone rescaling
    cannot change them, and rescaling before scoring would only create the
    impression that it might.
    """
    anomaly_map = np.asarray(anomaly_map, dtype=np.float32)
    span = float(high) - float(low)
    if span <= 0:
        return np.zeros_like(anomaly_map)
    return np.clip((anomaly_map - low) / span, 0.0, 1.0)


def binary_mask(anomaly_map: np.ndarray, threshold: float) -> np.ndarray:
    """S6b — the defect mask a downstream consumer would act on."""
    return (np.asarray(anomaly_map) >= threshold).astype(np.uint8)


def overlay(image: np.ndarray, anomaly_map: np.ndarray, alpha: float = OVERLAY_ALPHA,
            value_range: tuple[float, float] | None = None) -> np.ndarray:
    """Heat-map overlay on the de-normalised RGB image, for the qualitative grids.

    By default the colours span the map's own range; `value_range` fixes the span
    instead (the demo passes the S7 normal range, so a clean part stays dark).
    """
    image = np.asarray(image, dtype=np.uint8)
    low, high = value_range or (float(np.min(anomaly_map)), float(np.max(anomaly_map)))
    normalised = normalise(anomaly_map, low, high)
    heat = cv2.applyColorMap((normalised * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    heat_rgb = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(image, 1.0 - alpha, heat_rgb, alpha, 0.0)

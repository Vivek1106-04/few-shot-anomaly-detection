"""The method interface — S3 (reference model) and S4 (scoring), and nothing else.

This is the architectural keystone for Gap G1. Every method is exactly two
operations on *features*, never on images:

    enrol(support_features)          -> reference model R        (S3)
    score(query_features, R)         -> patch-level anomaly map   (S4)

Consequences that the rest of the project depends on:

  * A method cannot touch pre-processing, sampling, post-processing or metrics, so
    a difference between two methods can only come from those two functions.
  * Methods are pure numpy over arrays the backbone already produced, so they are
    unit-testable on synthetic features, with no weights and no dataset.
  * S5/S6 (upsampling, smoothing, image score) live in `postprocess` and are
    applied identically to every method's output, so post-processing cannot
    advantage one method over another.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class AnomalyMethod(Protocol):
    """What the experiment runner is allowed to assume about a method."""

    name: str

    def enrol(self, support: np.ndarray) -> object:
        """Build the reference model from (k, C, grid, grid) support features."""

    def score(self, query: np.ndarray, reference: object) -> np.ndarray:
        """Patch-level anomaly map (grid, grid) for one (C, grid, grid) query."""


def as_patch_matrix(features: np.ndarray) -> np.ndarray:
    """(N, C, H, W) or (C, H, W) features -> (N*H*W, C) patch rows.

    Every method works on patch rows; doing the reshape in one place keeps the
    channel/position ordering identical across methods, which is the sort of
    silent inconsistency that would otherwise make a comparison meaningless.
    """
    features = np.asarray(features, dtype=np.float32)
    if features.ndim == 3:
        features = features[None]
    if features.ndim != 4:
        raise ValueError(f"expected (N, C, H, W) or (C, H, W) features, got {features.shape}")
    n, channels, height, width = features.shape
    return features.transpose(0, 2, 3, 1).reshape(n * height * width, channels)


def grid_shape(features: np.ndarray) -> tuple[int, int]:
    """The (H, W) patch grid of a feature array."""
    features = np.asarray(features)
    return (features.shape[-2], features.shape[-1])


def squared_distances(queries: np.ndarray, bank: np.ndarray) -> np.ndarray:
    """Pairwise squared Euclidean distances, (n_queries, n_bank).

    Expanded as |a|^2 + |b|^2 - 2ab so the work lands in one matrix product; the
    nearest-neighbour search is the dominant cost of PatchCore at inference, and
    the naive broadcast form is both slower and (n_q, n_b, C) in memory.
    Distances are clipped at zero: the expansion can go slightly negative in
    floating point when a query coincides with a bank vector.
    """
    queries = np.asarray(queries, dtype=np.float32)
    bank = np.asarray(bank, dtype=np.float32)
    if queries.shape[1] != bank.shape[1]:
        raise ValueError(f"dimension mismatch: {queries.shape[1]} vs {bank.shape[1]}")
    cross = queries @ bank.T
    squared = (queries ** 2).sum(axis=1)[:, None] + (bank ** 2).sum(axis=1)[None, :] - 2 * cross
    return np.maximum(squared, 0.0)


def build_method(name: str, **parameters: object) -> AnomalyMethod:
    """Construct a method by name; unknown names fail loudly."""
    from .padim import PaDiM
    from .patchcore import PatchCore
    from .winclip import WinCLIP

    registry = {PatchCore.name: PatchCore, PaDiM.name: PaDiM, WinCLIP.name: WinCLIP}
    if name not in registry:
        raise ValueError(f"unknown method {name!r}; known: {sorted(registry)}")
    return registry[name](**parameters)


def known_methods() -> tuple[str, ...]:
    """Method names the runner can build, for validation and for the report."""
    from .padim import PaDiM
    from .patchcore import PatchCore
    from .winclip import WinCLIP

    return (PatchCore.name, PaDiM.name, WinCLIP.name)

"""PatchCore (Roth et al., CVPR 2022) — S3 memory bank, S4 nearest-neighbour score.

Why this is the method to implement first in a *few-shot* project: PatchCore stores
exemplars instead of estimating parameters. Nothing in enrolment divides by k, so
there is no statistic that becomes undefined when k = 1 — it degrades gracefully
where a parametric method degenerates (contrast `padim.py`).

Three parts, in the paper's order:

  1. **Local neighbourhood aggregation** — each patch is replaced by the average of
     its 3x3 neighbourhood. This widens the receptive field without going deeper
     into the network, which would have cost resolution and added ImageNet class
     bias.
  2. **Greedy coreset subsampling** — keep a fraction of the patches chosen to
     cover the bank, not at random. The search cost at inference is proportional to
     the bank size, and at k = full the bank is otherwise hundreds of thousands of
     vectors.
  3. **Nearest-neighbour scoring** — a patch is as anomalous as it is far from the
     closest normal patch ever seen.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .base import as_patch_matrix, grid_shape, squared_distances

DEFAULT_CORESET_RATIO = 0.1
DEFAULT_NEIGHBOURHOOD = 3
DEFAULT_PROJECTION_DIM = 128  # Johnson-Lindenstrauss dimension for coreset selection
MIN_BANK = 1


@dataclass(frozen=True)
class MemoryBank:
    """The reference model R: normal patch exemplars, plus what produced them."""

    vectors: np.ndarray  # (m, C) retained patch features
    n_patches_seen: int  # before subsampling — the reduction is reported
    coreset_ratio: float

    @property
    def size(self) -> int:
        return int(self.vectors.shape[0])


def average_pool(features: np.ndarray, size: int = DEFAULT_NEIGHBOURHOOD) -> np.ndarray:
    """Average each patch with its `size` x `size` neighbourhood, edge-padded.

    Written as a sum of shifted views rather than with a sliding-window view: the
    window form materialises size^2 copies of the feature map, which at k = full is
    several gigabytes for no gain.
    """
    features = np.asarray(features, dtype=np.float32)
    if features.ndim == 3:
        features = features[None]
    if features.ndim != 4:
        raise ValueError(f"expected (N, C, H, W) or (C, H, W) features, got {features.shape}")
    if size < 1 or size % 2 == 0:
        raise ValueError(f"neighbourhood size must be a positive odd number, got {size}")
    if size == 1:
        return features
    pad = size // 2
    height, width = features.shape[-2:]
    padded = np.pad(features, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode="edge")
    total = np.zeros_like(features)
    for row in range(size):
        for column in range(size):
            total += padded[..., row: row + height, column: column + width]
    return total / float(size * size)


def greedy_coreset(
    points: np.ndarray, n_select: int, rng: np.random.Generator, projection_dim: int
) -> np.ndarray:
    """Indices of a greedy k-centre coreset: repeatedly take the farthest point.

    The selection runs on a random projection of the features, as in the paper: a
    Johnson-Lindenstrauss projection preserves the distances the greedy rule
    compares, and cuts the cost of the selection loop by an order of magnitude.
    The *retained vectors* are the full-dimensional originals — only the choosing
    is approximate.
    """
    n_points = points.shape[0]
    n_select = int(np.clip(n_select, MIN_BANK, n_points))
    if n_select == n_points:
        return np.arange(n_points)

    dimension = min(projection_dim, points.shape[1])
    projection = rng.normal(0.0, 1.0 / np.sqrt(dimension), size=(points.shape[1], dimension))
    projected = (points @ projection).astype(np.float32)

    selected = np.empty(n_select, dtype=np.int64)
    selected[0] = int(rng.integers(n_points))
    distance = squared_distances(projected, projected[selected[0]][None]).ravel()
    for index in range(1, n_select):
        chosen = int(np.argmax(distance))
        selected[index] = chosen
        distance = np.minimum(
            distance, squared_distances(projected, projected[chosen][None]).ravel()
        )
    return selected


class PatchCore:
    """Memory-bank anomaly detection over frozen patch features."""

    name = "patchcore"

    def __init__(
        self,
        coreset_ratio: float = DEFAULT_CORESET_RATIO,
        neighbourhood: int = DEFAULT_NEIGHBOURHOOD,
        projection_dim: int = DEFAULT_PROJECTION_DIM,
        seed: int = 0,
    ) -> None:
        if not 0.0 < coreset_ratio <= 1.0:
            raise ValueError(f"coreset_ratio must be in (0, 1], got {coreset_ratio}")
        self.coreset_ratio = float(coreset_ratio)
        self.neighbourhood = int(neighbourhood)
        self.projection_dim = int(projection_dim)
        self.seed = int(seed)

    @property
    def parameters(self) -> dict[str, object]:
        """Written into the result file next to the metrics."""
        return {
            "coreset_ratio": self.coreset_ratio,
            "neighbourhood": self.neighbourhood,
            "projection_dim": self.projection_dim,
            "seed": self.seed,
        }

    def enrol(self, support: np.ndarray) -> MemoryBank:
        """S3 — pooled support patches, subsampled to a coreset."""
        pooled = average_pool(support, self.neighbourhood)
        patches = as_patch_matrix(pooled)
        rng = np.random.default_rng(self.seed)
        n_select = int(round(self.coreset_ratio * patches.shape[0]))
        indices = greedy_coreset(patches, n_select, rng, self.projection_dim)
        return MemoryBank(
            vectors=patches[np.sort(indices)],
            n_patches_seen=int(patches.shape[0]),
            coreset_ratio=self.coreset_ratio,
        )

    def score(self, query: np.ndarray, reference: MemoryBank) -> np.ndarray:
        """S4 — distance from each query patch to its nearest normal exemplar."""
        pooled = average_pool(query, self.neighbourhood)
        patches = as_patch_matrix(pooled)
        nearest = squared_distances(patches, reference.vectors).min(axis=1)
        return np.sqrt(nearest).reshape(grid_shape(query))

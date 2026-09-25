"""PaDiM (Defard et al., ICPR 2021) — S3 per-position Gaussian, S4 Mahalanobis score.

PaDiM models each of the grid's positions independently: one multivariate Gaussian
over the k support vectors at that position, and a query patch is scored by its
Mahalanobis distance from that position's Gaussian.

**This method is in the project because of how it fails.** Fitting a d-dimensional
covariance from k samples is rank-deficient whenever k <= d, which in a few-shot
setting is always: at k = 1 the sample covariance is exactly zero. The honest
response is to regularise explicitly and *report the regularisation*, not to pick a
d small enough to hide the problem:

    S_reg = (1 - shrinkage) * S + shrinkage * (trace(S) / d) * I

with a floor epsilon added to the diagonal so the matrix is invertible even when S
is the zero matrix (k = 1). The shrinkage target is the scaled identity, i.e. "when
in doubt, assume the dimensions are uncorrelated with equal variance" — the
standard Ledoit-Wolf target, with a fixed coefficient rather than an estimated one,
because estimating the coefficient from k = 1 sample is no better founded than
choosing it.

PaDiM's second structural assumption is that **position (i, j) means the same thing
in every image**, since each position carries its own Gaussian. That is the
alignment assumption the week04 floor baseline already broke on `screw`, and this
implementation is what turns that observation into a measurement across k.

Dimensionality reduction is a random subset of the feature channels, as in the
paper: it reports random selection to be as good as PCA here, and unlike PCA it
needs no fitting, which matters when there are k = 1 samples to fit on.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .base import grid_shape

DEFAULT_DIM = 100  # channels kept per position; with k <= 16 a larger d buys nothing
DEFAULT_SHRINKAGE = 0.1
EPSILON = 1e-4  # diagonal floor: makes the k = 1 covariance invertible at all


@dataclass(frozen=True)
class GaussianModel:
    """The reference model R: one Gaussian per patch position."""

    mean: np.ndarray  # (P, d)
    precision: np.ndarray  # (P, d, d) inverse of the regularised covariance
    channels: np.ndarray  # (d,) indices of the retained feature channels
    grid: tuple[int, int]
    k: int  # support size the Gaussians were fitted from
    shrinkage: float

    @property
    def is_rank_deficient(self) -> bool:
        """True when the sample covariance could not have full rank (k <= d).

        Reported rather than avoided: it is the point of including PaDiM.
        """
        return self.k <= self.mean.shape[1]


def select_channels(n_channels: int, dim: int, seed: int) -> np.ndarray:
    """Deterministic random subset of feature channels, sorted."""
    dim = int(min(dim, n_channels))
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_channels, size=dim, replace=False))


def regularise(covariance: np.ndarray, shrinkage: float, epsilon: float) -> np.ndarray:
    """Shrink each (d, d) covariance towards the scaled identity, then floor it."""
    dimension = covariance.shape[-1]
    identity = np.eye(dimension, dtype=covariance.dtype)
    scale = np.trace(covariance, axis1=-2, axis2=-1)[:, None, None] / dimension
    shrunk = (1.0 - shrinkage) * covariance + shrinkage * scale * identity
    return shrunk + epsilon * identity


class PaDiM:
    """Per-position Gaussian modelling of normal patch features."""

    name = "padim"

    def __init__(
        self,
        dim: int = DEFAULT_DIM,
        shrinkage: float = DEFAULT_SHRINKAGE,
        epsilon: float = EPSILON,
        seed: int = 0,
    ) -> None:
        if not 0.0 <= shrinkage <= 1.0:
            raise ValueError(f"shrinkage must be in [0, 1], got {shrinkage}")
        if epsilon <= 0.0:
            raise ValueError(f"epsilon must be positive, got {epsilon}")
        self.dim = int(dim)
        self.shrinkage = float(shrinkage)
        self.epsilon = float(epsilon)
        self.seed = int(seed)

    @property
    def parameters(self) -> dict[str, object]:
        """Written into the result file next to the metrics."""
        return {
            "dim": self.dim,
            "shrinkage": self.shrinkage,
            "epsilon": self.epsilon,
            "seed": self.seed,
        }

    def enrol(self, support: np.ndarray) -> GaussianModel:
        """S3 — mean and regularised covariance per position, from k support images."""
        support = np.asarray(support, dtype=np.float32)
        if support.ndim == 3:
            support = support[None]
        if support.ndim != 4:
            raise ValueError(f"expected (k, C, H, W) support features, got {support.shape}")

        k, n_channels, height, width = support.shape
        channels = select_channels(n_channels, self.dim, self.seed)
        # (k, C, H, W) -> (P, k, d): every position gets its own sample matrix.
        samples = support[:, channels].reshape(k, len(channels), height * width)
        samples = samples.transpose(2, 0, 1).astype(np.float64)

        mean = samples.mean(axis=1)  # (P, d)
        centred = samples - mean[:, None, :]
        denominator = max(k - 1, 1)  # k = 1: the sample covariance is zero either way
        covariance = np.einsum("pkd,pke->pde", centred, centred) / denominator
        precision = np.linalg.inv(regularise(covariance, self.shrinkage, self.epsilon))
        return GaussianModel(
            mean=mean,
            precision=precision,
            channels=channels,
            grid=(height, width),
            k=k,
            shrinkage=self.shrinkage,
        )

    def score(self, query: np.ndarray, reference: GaussianModel) -> np.ndarray:
        """S4 — Mahalanobis distance from each position's Gaussian."""
        query = np.asarray(query, dtype=np.float64)
        height, width = grid_shape(query)
        if (height, width) != reference.grid:
            raise ValueError(f"grid mismatch: query {(height, width)} vs model {reference.grid}")
        patches = query[reference.channels].reshape(len(reference.channels), height * width).T
        difference = patches - reference.mean
        squared = np.einsum("pd,pde,pe->p", difference, reference.precision, difference)
        return np.sqrt(np.maximum(squared, 0.0)).reshape(height, width).astype(np.float32)

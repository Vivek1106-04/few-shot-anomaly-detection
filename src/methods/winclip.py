"""WinCLIP-style few-shot method (Jeong et al., CVPR 2023) — the third S3/S4 variant.

Week03 §S3(c): the few-shot-specific method, chosen here over the RegAD branch
because it needs no training on auxiliary categories — RegAD's registration network
is trained on the *other* MVTec categories, which would put gradient training back
into a pipeline whose defining rule is that nothing is trained. The price is the
extra data assumption this method carries and the results table declares: CLIP's
web-scale image-text pretraining.

Two scores per patch, fused, both in [0, 1]:

  * **Language (zero-shot).** Each patch's text-aligned embedding is compared with
    the [normal, anomalous] prompt-ensemble embeddings; the score is the softmax
    probability of "anomalous". This needs *no support image at all*, so it is
    defined at k = 0 — the one point on the k-axis no other method in the project
    can occupy.
  * **Visual (few-shot).** A memory of the support set's visual patch tokens; a
    query patch scores (1 - max cosine similarity) / 2. This is PatchCore's
    nearest-neighbour idea in cosine form, on CLIP tokens, without a coreset: at
    k <= 16 on a 14 x 14 grid the whole bank is at most 3 136 vectors.

    fused = (1 - w) * visual + w * language,     w = `language_weight` (0.5 default)

What is simplified relative to the paper, stated plainly: WinCLIP extracts features
from multi-scale sliding *windows* (hundreds of masked forward passes per image);
this implementation reads dense per-patch embeddings from one forward pass instead
(the MaskCLIP value-path construction, see `models/clip_backbone.py`). That trades
some of the paper's localisation quality for a ~100x cheaper inference, which is the
side of the trade a line with a latency budget has to take.

Feature layout expected (from `CLIPBackbone`): (C, H, W) with the trailing
`language_dim` channels text-aligned and unit-length, and the rest split into
`n_visual_blocks` equal blocks of visual tokens, each normalised separately so no
one layer dominates the cosine similarity by having larger activations.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .base import as_patch_matrix, grid_shape

DEFAULT_LANGUAGE_DIM = 512
DEFAULT_VISUAL_BLOCKS = 2
DEFAULT_LANGUAGE_WEIGHT = 0.5
DEFAULT_TEMPERATURE = 100.0  # CLIP's own logit scale
NORM_FLOOR = 1e-8


@dataclass(frozen=True)
class VisualMemory:
    """The reference model R: unit-length visual patch tokens of the support set."""

    vectors: np.ndarray  # (k * H * W, C_visual), unit rows
    grid: tuple[int, int]
    k: int


def unit_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, NORM_FLOOR)


def normalise_blocks(patches: np.ndarray, n_blocks: int) -> np.ndarray:
    """Unit-normalise each of `n_blocks` equal channel blocks, then rescale to unit length."""
    blocks = np.split(patches, n_blocks, axis=1)
    return np.concatenate([unit_rows(block) for block in blocks], axis=1) / np.sqrt(n_blocks)


def softmax_anomaly(language: np.ndarray, text: np.ndarray, temperature: float) -> np.ndarray:
    """P(anomalous) per patch row, from cosine similarities to [normal, anomalous] text."""
    logits = temperature * unit_rows(language) @ text.T  # (P, 2)
    logits -= logits.max(axis=1, keepdims=True)
    exponentials = np.exp(logits)
    return exponentials[:, 1] / exponentials.sum(axis=1)


class WinCLIP:
    """Language + visual-memory anomaly scoring over CLIP patch features."""

    name = "winclip"

    def __init__(
        self,
        text: np.ndarray | None = None,
        language_weight: float = DEFAULT_LANGUAGE_WEIGHT,
        language_dim: int = DEFAULT_LANGUAGE_DIM,
        n_visual_blocks: int = DEFAULT_VISUAL_BLOCKS,
        temperature: float = DEFAULT_TEMPERATURE,
        seed: int = 0,
    ) -> None:
        if not 0.0 <= language_weight <= 1.0:
            raise ValueError(f"language_weight must be in [0, 1], got {language_weight}")
        if language_weight > 0.0 and text is None:
            raise ValueError("a language score needs the [normal, anomalous] text embeddings")
        if text is not None and np.asarray(text).shape != (2, language_dim):
            raise ValueError(f"text must be (2, {language_dim}), got {np.asarray(text).shape}")
        self.text = None if text is None else unit_rows(np.asarray(text, dtype=np.float32))
        self.language_weight = float(language_weight)
        self.language_dim = int(language_dim)
        self.n_visual_blocks = int(n_visual_blocks)
        self.temperature = float(temperature)
        self.seed = int(seed)  # accepted for the runner's uniform call; nothing is random

    @property
    def parameters(self) -> dict[str, object]:
        """Written into the result file next to the metrics (the text is not: it is
        a deterministic function of the category name and the frozen CLIP weights)."""
        return {
            "language_weight": self.language_weight,
            "language_dim": self.language_dim,
            "n_visual_blocks": self.n_visual_blocks,
            "temperature": self.temperature,
            "seed": self.seed,
        }

    def _split(self, patches: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if patches.shape[1] <= self.language_dim:
            raise ValueError(f"features have {patches.shape[1]} channels; expected visual "
                             f"channels before the {self.language_dim} language channels")
        return patches[:, :-self.language_dim], patches[:, -self.language_dim:]

    def enrol(self, support: np.ndarray) -> VisualMemory:
        """S3 — the support set's visual patch tokens, normalised per block."""
        support = np.asarray(support, dtype=np.float32)
        visual, _ = self._split(as_patch_matrix(support))
        k = 1 if support.ndim == 3 else support.shape[0]
        return VisualMemory(normalise_blocks(visual, self.n_visual_blocks),
                            grid_shape(support), k)

    def score(self, query: np.ndarray, reference: VisualMemory | None) -> np.ndarray:
        """S4 — fused visual-memory and language score per patch.

        `reference` may be None only for the pure zero-shot configuration
        (language_weight = 1), which is the k = 0 point of the curve.
        """
        visual, language = self._split(as_patch_matrix(query))
        fused = np.zeros(visual.shape[0], dtype=np.float64)
        if self.language_weight < 1.0:
            if reference is None:
                raise ValueError("a visual score needs an enrolled support memory")
            similarity = normalise_blocks(visual, self.n_visual_blocks) @ reference.vectors.T
            fused += (1.0 - self.language_weight) * (1.0 - similarity.max(axis=1)) / 2.0
        if self.language_weight > 0.0:
            fused += self.language_weight * softmax_anomaly(language, self.text, self.temperature)
        return fused.reshape(grid_shape(query)).astype(np.float32)

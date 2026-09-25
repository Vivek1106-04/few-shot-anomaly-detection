"""S2 — frozen feature extraction.

One backbone, never fine-tuned, shared by every method (week03 §S2). Two design
decisions are load-bearing and both are dictated by the few-shot setting:

  * **Frozen, ImageNet-pretrained.** With k as low as 1, there is nothing to train
    on; the method's whole job is to describe normality in a feature space it did
    not choose. Freezing also makes the comparison in Gap G1 fair — every method
    sees bit-identical features.
  * **Mid-level taps (layer2 + layer3), not the last layer.** layer1 is close to
    raw texture and carries no semantics; layer4 has a 7x7 grid and an ImageNet
    classification bias, which localises defects too coarsely. layer2 (28x28) and
    layer3 (14x14) are the pair PaDiM and PatchCore both use.

layer3 is bilinearly upsampled to the layer2 grid before concatenation, so one
position in the returned map is one position in every method's model, and the
anomaly map that S5 upsamples has a single well-defined geometry (28x28 for a
224x224 input, i.e. one patch per 8x8 pixel block).

Torch appears in the S2 files (this one and `clip_backbone.py`) and nowhere else: methods (S3/S4) take
feature arrays, so they stay numpy-only and testable without the 270 MB weights.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DEFAULT_BACKBONE = "wide_resnet50_2"
DEFAULT_LAYERS = ("layer2", "layer3")
# Channel counts of the WideResNet-50-2 taps, used to state the contract without
# running the network (and asserted against the real tensors at extraction time).
LAYER_CHANNELS = {"layer1": 256, "layer2": 512, "layer3": 1024, "layer4": 2048}
BATCH_SIZE = 8


@dataclass(frozen=True)
class FeatureSpec:
    """The shape contract of one configured backbone."""

    channels: int
    grid: int  # feature maps are grid x grid for a 224x224 input
    language_dim: int = 0  # trailing text-aligned channels (CLIP only; see clip_backbone)

    @property
    def n_patches(self) -> int:
        return self.grid * self.grid


class FrozenBackbone:
    """WideResNet-50-2 in eval mode with forward hooks on the tapped layers.

    `extract` maps a batch of S1-preprocessed images, (N, 3, 224, 224), to patch
    features (N, C, grid, grid) as float32, with gradients disabled throughout.
    """

    def __init__(
        self,
        name: str = DEFAULT_BACKBONE,
        layers: tuple[str, ...] = DEFAULT_LAYERS,
        device: str = "cpu",
        batch_size: int = BATCH_SIZE,
    ) -> None:
        import torch  # imported lazily: the methods and their tests need no torch
        import torchvision

        if not layers:
            raise ValueError("at least one layer must be tapped")
        unknown = [layer for layer in layers if layer not in LAYER_CHANNELS]
        if unknown:
            raise ValueError(f"unknown layer(s) {unknown}; known: {sorted(LAYER_CHANNELS)}")

        self.name = name
        self.layers = tuple(layers)
        self.device = torch.device(device)
        self.batch_size = batch_size

        builder = getattr(torchvision.models, name, None)
        if builder is None:
            raise ValueError(f"torchvision has no model {name!r}")
        self._torch = torch
        self._model = builder(weights="IMAGENET1K_V1").to(self.device).eval()
        self._model.requires_grad_(False)

        self._taps: dict[str, object] = {}
        for layer in self.layers:
            getattr(self._model, layer).register_forward_hook(self._make_hook(layer))

    def _make_hook(self, layer: str):
        def hook(_module, _inputs, output) -> None:
            self._taps[layer] = output.detach()

        return hook

    @property
    def spec(self) -> FeatureSpec:
        """Channels after concatenation, and the grid every tap is aligned to."""
        channels = sum(LAYER_CHANNELS[layer] for layer in self.layers)
        return FeatureSpec(channels=channels, grid=28)

    def extract(self, images: np.ndarray) -> np.ndarray:
        """S1 tensors (N, 3, 224, 224) -> patch features (N, C, grid, grid)."""
        torch = self._torch
        images = np.asarray(images, dtype=np.float32)
        if images.ndim == 3:
            images = images[None]
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(f"expected (N, 3, H, W) preprocessed images, got {images.shape}")

        outputs = []
        with torch.no_grad():
            for start in range(0, len(images), self.batch_size):
                batch = torch.from_numpy(images[start: start + self.batch_size]).to(self.device)
                self._taps.clear()
                self._model(batch)
                outputs.append(self._align_and_concat().cpu().numpy())
        return np.concatenate(outputs, axis=0).astype(np.float32)

    def _align_and_concat(self):
        """Resample every tap to the first tap's grid and concatenate on channels.

        Bilinear, because the deeper tap is being *upsampled*: nearest neighbour
        would quantise the finer grid's positions onto the coarser one and put a
        visible 2x2 blockiness into every anomaly map produced downstream.
        """
        torch = self._torch
        taps = [self._taps[layer] for layer in self.layers]
        target = taps[0].shape[-2:]
        aligned = [
            tap if tap.shape[-2:] == target
            else torch.nn.functional.interpolate(tap, size=target, mode="bilinear",
                                                 align_corners=False)
            for tap in taps
        ]
        return torch.cat(aligned, dim=1)

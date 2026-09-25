"""S2, second backbone — frozen CLIP ViT-B/16, for the few-shot-specific method.

`backbone: clip_vit_b16` in the run configuration selects this file instead of
`backbone.py`; nothing else in the pipeline changes (week03 §S2: "backbone choice
is a configuration flag, not a code path"). It exists because the third method
(WinCLIP-style, `methods/winclip.py`) needs what a CLIP model has and an ImageNet
classifier does not: an image space aligned with a text space, so that "a photo of a
damaged bottle" can be scored against a patch with no defective example ever shown.

One feature array per image, (2048, 14, 14), in two declared parts:

  * channels [0, 1536)    **visual** — patch tokens after blocks 6 and 9 of 12,
                          the mid-depth pair, for the same reason WRN-50 is tapped at
                          layer2 + layer3: early enough to keep local detail, late
                          enough to be semantic. PatchCore/PaDiM run on these too.
  * channels [1536, 2048) **language** — one text-aligned 512-d embedding per
                          patch, unit length, computed from the last block's value
                          path (the MaskCLIP construction, Zhou et al., ECCV 2022):
                          the final attention's query-key mixing is what smears
                          CLIP's patch tokens into one global summary, and dropping
                          it keeps each position's embedding local while still
                          landing it in the joint image-text space.

Input contract is unchanged: S1's ImageNet-normalised tensors. The ImageNet
normalisation is undone and CLIP's own applied *inside* this extractor, which is an
exact affine change of variables — S1 stays a single transform for every backbone.

The grid is 14 x 14 (one token per 16 x 16 pixels), twice as coarse as WRN-50's
28 x 28. That is a real cost for small defects and is reported, not hidden.

Extra data assumption, declared as week03 §S3(c) requires: web-scale image-text
pretraining (OpenAI CLIP, 400 M pairs). No category image is used in training.
"""

from __future__ import annotations

import re

import numpy as np

from data.transforms import IMAGENET_MEAN, IMAGENET_STD
from models.backbone import FeatureSpec

CLIP_NAME = "clip_vit_b16"
OPEN_CLIP_MODEL = "ViT-B-16-quickgelu"  # quick-GELU: the activation the OpenAI weights use
OPEN_CLIP_WEIGHTS = "openai"
CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
VISUAL_BLOCKS = (5, 8)  # zero-based: the outputs of blocks 6 and 9
WIDTH = 768
LANGUAGE_DIM = 512
GRID = 14
BATCH_SIZE = 16

# WinCLIP's compositional prompt ensemble (Jeong et al., CVPR 2023): state words x
# templates, averaged per class. Averaging over many phrasings is what makes a
# zero-shot score stable; any single prompt is noticeably sensitive to wording.
NORMAL_STATES = ("{}", "flawless {}", "perfect {}", "unblemished {}", "{} without flaw",
                 "{} without defect", "{} without damage")
ANOMALOUS_STATES = ("damaged {}", "broken {}", "{} with flaw", "{} with defect",
                    "{} with damage")
TEMPLATES = (
    "a cropped photo of the {}.", "a close-up photo of a {}.", "a close-up photo of the {}.",
    "a bright photo of a {}.", "a bright photo of the {}.", "a dark photo of the {}.",
    "a dark photo of a {}.", "a jpeg corrupted photo of the {}.", "a blurry photo of the {}.",
    "a photo of the {}.", "a photo of a {}.", "a photo of a small {}.",
    "a photo of the large {}.", "a photo of the {} for visual inspection.",
    "a photo of a {} for visual inspection.", "a photo of the {} for anomaly detection.",
    "a photo of a {} for anomaly detection.",
)
# Category folder names that are not the English name of the object.
OBJECT_NAMES = {"pcb": "printed circuit board", "chewinggum": "chewing gum",
                "metal_nut": "metal nut", "pipe_fryum": "pipe fryum"}


def object_name(category: str) -> str:
    """'pcb3' -> 'printed circuit board', 'metal_nut' -> 'metal nut', 'bottle' -> 'bottle'."""
    base = re.sub(r"\d+$", "", category)
    return OBJECT_NAMES.get(base, base.replace("_", " "))


def prompts(category: str) -> tuple[list[str], list[str]]:
    """The full (normal, anomalous) prompt lists for one category."""
    name = object_name(category)
    expand = lambda states: [t.format(s.format(name)) for s in states for t in TEMPLATES]  # noqa: E731
    return expand(NORMAL_STATES), expand(ANOMALOUS_STATES)


def clip_input(images: np.ndarray) -> np.ndarray:
    """S1 (ImageNet-normalised) tensors -> CLIP-normalised tensors, exactly."""
    images = np.asarray(images, dtype=np.float32)
    mean_in, std_in = IMAGENET_MEAN[:, None, None], IMAGENET_STD[:, None, None]
    mean_out, std_out = CLIP_MEAN[:, None, None], CLIP_STD[:, None, None]
    return ((images * std_in + mean_in) - mean_out) / std_out


class CLIPBackbone:
    """Frozen OpenAI CLIP ViT-B/16: visual patch tokens plus text-aligned patch embeddings."""

    name = CLIP_NAME

    def __init__(self, device: str = "cpu", batch_size: int = BATCH_SIZE) -> None:
        import open_clip  # lazily: only runs that ask for CLIP pay for it
        import torch

        self._torch = torch
        self.device = torch.device(device)
        self.batch_size = batch_size
        model, _, _ = open_clip.create_model_and_transforms(
            OPEN_CLIP_MODEL, pretrained=OPEN_CLIP_WEIGHTS)
        self._model = model.to(self.device).eval()
        self._model.requires_grad_(False)
        self._tokenizer = open_clip.get_tokenizer(OPEN_CLIP_MODEL)

    @property
    def spec(self) -> FeatureSpec:
        return FeatureSpec(channels=WIDTH * len(VISUAL_BLOCKS) + LANGUAGE_DIM, grid=GRID,
                           language_dim=LANGUAGE_DIM)

    def extract(self, images: np.ndarray) -> np.ndarray:
        """S1 tensors (N, 3, 224, 224) -> features (N, 2048, 14, 14), float32."""
        torch = self._torch
        images = np.asarray(images, dtype=np.float32)
        if images.ndim == 3:
            images = images[None]
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(f"expected (N, 3, H, W) preprocessed images, got {images.shape}")
        outputs = []
        with torch.no_grad():
            for start in range(0, len(images), self.batch_size):
                batch = torch.from_numpy(clip_input(images[start: start + self.batch_size]))
                outputs.append(self._forward(batch.to(self.device)).cpu().numpy())
        return np.concatenate(outputs, axis=0).astype(np.float32)

    def _forward(self, batch):
        torch = self._torch
        visual = self._model.visual
        tokens = visual._embeds(batch)
        blocks = visual.transformer.resblocks
        taps = []
        for index, block in enumerate(blocks[:-1]):
            tokens = block(tokens)
            if index in VISUAL_BLOCKS:
                taps.append(tokens[:, 1:])  # drop the class token
        language = self._value_path(blocks[-1], tokens)[:, 1:]
        grid = int(round((tokens.shape[1] - 1) ** 0.5))
        parts = [*taps, language]
        stacked = torch.cat(parts, dim=-1)  # (N, grid*grid, C)
        return stacked.transpose(1, 2).reshape(stacked.shape[0], -1, grid, grid)

    def _value_path(self, block, tokens):
        """MaskCLIP dense embeddings: value projection -> out_proj -> ln_post -> proj."""
        torch = self._torch
        attention = block.attn
        width = attention.embed_dim
        normed = block.ln_1(tokens)
        value = torch.nn.functional.linear(normed, attention.in_proj_weight[2 * width:],
                                           attention.in_proj_bias[2 * width:])
        embedded = self._model.visual.ln_post(attention.out_proj(value)) @ self._model.visual.proj
        return torch.nn.functional.normalize(embedded, dim=-1)

    def text_embeddings(self, category: str) -> np.ndarray:
        """(2, 512) unit vectors: the [normal, anomalous] prompt-ensemble means."""
        torch = self._torch
        rows = []
        with torch.no_grad():
            for texts in prompts(category):
                encoded = self._model.encode_text(self._tokenizer(texts).to(self.device))
                mean = torch.nn.functional.normalize(encoded, dim=-1).mean(dim=0)
                rows.append(torch.nn.functional.normalize(mean, dim=0).cpu().numpy())
        return np.stack(rows).astype(np.float32)

"""S8 — qualitative outputs: what one image looks like coming out of the pipeline.

A metric table says a method localises well; it does not say *what it marks*. This
produces the grid the report shows per category:

    input | ground truth | anomaly map | overlay | binary decision

The binary mask needs a threshold, and it comes from the S7 decision stage
(`decision.py`): fitted on **held-out train normals** — normals outside the support
set, never a test image — so no defect, and no image that is later scored, ever
informs the decision. That is the one-class rule the project is defined by.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from data.mvtec import MVTecAD, Sample  # noqa: E402
from data.sampler import draw_support  # noqa: E402
from data.transforms import denormalise  # noqa: E402
from decision import N_VALIDATION, fit_decision, validation_normals  # noqa: E402
from experiment import TEXT_METHODS, build_extractor  # noqa: E402
from config.schema import RunConfig  # noqa: E402
from methods import build_method  # noqa: E402
from postprocess import binary_mask, overlay, to_pixel_map  # noqa: E402

FPR_BUDGET = 0.10
FIG_DPI = 150


def pixel_maps(extractor, method, reference, dataset: MVTecAD,
               samples: Sequence[Sample]) -> np.ndarray:
    """Post-processed anomaly maps for these samples, via the shared S5 path."""
    images = np.stack([dataset.load_image(sample) for sample in samples])
    features = extractor.extract(images)
    return np.stack([to_pixel_map(method.score(feature, reference)) for feature in features])


def build_panels(
    dataset: MVTecAD, sample: Sample, anomaly_map: np.ndarray, threshold: float
) -> list[tuple[str, np.ndarray, str | None]]:
    """The five panels of one row of the grid."""
    image = denormalise(dataset.load_image(sample))
    return [
        (f"{sample.category} / {sample.defect_type}", image, None),
        ("ground truth", dataset.load_mask(sample) * 255, "gray"),
        ("anomaly map", anomaly_map, "inferno"),
        ("overlay", overlay(image, anomaly_map), None),
        ("decision @ S7 pixel tau", binary_mask(anomaly_map, threshold) * 255, "gray"),
    ]


def category_row(
    dataset: MVTecAD, extractor, category: str, method_name: str, k: int, seed: int,
    parameters: dict | None = None,
) -> list[tuple[str, np.ndarray, str | None]]:
    """Enrol on a seeded support set, then post-process one defective test image."""
    support = draw_support(dataset.train_samples(category), k, 0, seed, category)
    parameters = {**(parameters or {}), "seed": support.seed % 2**31}
    if method_name in TEXT_METHODS:
        parameters.setdefault("text", extractor.text_embeddings(category))
    method = build_method(method_name, **parameters)
    images = np.stack([dataset.load_image(sample) for sample in support.samples])
    reference = method.enrol(extractor.extract(images))

    validation = validation_normals(dataset.train_samples(category), support.samples,
                                    N_VALIDATION, support.seed)
    decision = fit_decision(pixel_maps(extractor, method, reference, dataset, validation),
                            FPR_BUDGET)
    defective = next(s for s in dataset.test_samples(category) if s.mask_path is not None)
    anomaly_map = pixel_maps(extractor, method, reference, dataset, [defective])[0]
    return build_panels(dataset, defective, anomaly_map, decision.pixel_threshold)


def qualitative_grid(
    dataset: MVTecAD, extractor, categories: Sequence[str], method_name: str, k: int,
    seed: int, path: Path, parameters: dict | None = None,
) -> Path:
    """One row per category: the first defective test image, fully post-processed."""
    rows = [
        category_row(dataset, extractor, category, method_name, k, seed, parameters)
        for category in categories
    ]
    figure, axes = plt.subplots(len(rows), 5, figsize=(14, 2.9 * len(rows)))
    axes = np.atleast_2d(axes)
    for row_index, panels in enumerate(rows):
        for column, (title, data, cmap) in enumerate(panels):
            axis = axes[row_index][column]
            axis.imshow(data, cmap=cmap)
            axis.set_title(title, fontsize=8)
            axis.axis("off")
    figure.suptitle(f"{method_name}, k = {k}", fontsize=11)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(figure)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="data/mvtec_anomaly_detection")
    parser.add_argument("--categories", nargs="*", default=["bottle", "grid", "screw"])
    parser.add_argument("--method", default="patchcore")
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--backbone", default="wide_resnet50_2")
    parser.add_argument("--output", default="output/qualitative.png")
    arguments = parser.parse_args()

    dataset = MVTecAD(arguments.root)
    extractor = build_extractor(RunConfig(backbone=arguments.backbone))
    path = qualitative_grid(
        dataset, extractor, arguments.categories, arguments.method,
        arguments.k, arguments.seed, Path(arguments.output),
    )
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()

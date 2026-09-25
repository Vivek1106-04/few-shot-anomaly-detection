"""S11c — the failure catalogue: every error at the deployed threshold, binned by cause.

A metric says how often a method is wrong; the catalogue says *why*, with the image
to prove it (week03 §S11, Gap G8). Each test image is judged at the S7 threshold
fitted on held-out normals, and every error lands in exactly one bin decided by a
rule that reads only the ground truth and the anomaly map:

  missed defects (false negatives)
    sub-stride defect        the whole defect covers fewer than `SUB_STRIDE_PATCHES`
                             feature patches, so it cannot own a patch of its own —
                             the resolution floor named in week03 §S2
    weak defect              larger than that, yet scored inside the normal range:
                             the defect looks normal in this feature space
  false alarms (false positives)
    border artefact          the map's peak sits within `BORDER` px of the frame edge,
                             where the S1 crop and the zero-padded convolutions meet
    unmodelled normal variation  anywhere else: normal appearance the k references
                             did not cover — the coverage failure G6 predicts
  right verdict, wrong place (true positive, flagged separately)
    mislocalised             rejected, but the map's peak lies outside the defect
                             (grown by `TOLERANCE` px): the verdict is right for a
                             reason an operator could not verify

The rules are deliberately simple and stated here in full, so a reader can check a
bin assignment against the pictures rather than trust a classifier.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from config.schema import RunConfig  # noqa: E402
from data.mvtec import MVTecAD, Sample  # noqa: E402
from data.transforms import denormalise  # noqa: E402
from experiment import (  # noqa: E402
    FeatureCache,
    build_dataset,
    build_extractor,
    build_run_method,
    fit_threshold_stage,
    score_maps,
)
from stress import draw_for_mode, query_samples  # noqa: E402

PATCH_PIXELS = 8 * 8  # one WRN-50 layer2 patch at 224 x 224
SUB_STRIDE_PATCHES = 4
BORDER = 16  # px
TOLERANCE = 8  # px the ground truth is grown by before asking "is the peak inside?"
EXAMPLES_PER_BIN = 3
FIG_DPI = 130

MISSED = ("sub-stride defect", "weak defect")
FALSE_ALARMS = ("border artefact", "unmodelled normal variation")
MISLOCALISED = "mislocalised"
BINS = (*MISSED, *FALSE_ALARMS, MISLOCALISED)


def peak(anomaly_map: np.ndarray) -> tuple[int, int]:
    """(row, column) of the map's maximum."""
    index = np.unravel_index(int(np.argmax(anomaly_map)), anomaly_map.shape)
    return int(index[0]), int(index[1])


def classify(label: int, score: float, anomaly_map: np.ndarray, mask: np.ndarray,
             threshold: float) -> str | None:
    """The bin of one test image at `threshold`, or None when it is handled correctly."""
    rejected = score > threshold
    row, column = peak(anomaly_map)
    if label == 1 and not rejected:
        area = int((np.asarray(mask) > 0).sum())
        return MISSED[0] if area < SUB_STRIDE_PATCHES * PATCH_PIXELS else MISSED[1]
    if label == 0 and rejected:
        height, width = anomaly_map.shape
        at_border = min(row, column, height - 1 - row, width - 1 - column) < BORDER
        return FALSE_ALARMS[0] if at_border else FALSE_ALARMS[1]
    if label == 1 and rejected:
        kernel = np.ones((2 * TOLERANCE + 1, 2 * TOLERANCE + 1), dtype=np.uint8)
        grown = cv2.dilate((np.asarray(mask) > 0).astype(np.uint8), kernel)
        return None if grown[row, column] else MISLOCALISED
    return None


@dataclass(frozen=True)
class Entry:
    """One catalogued error, with what is needed to draw it."""

    category: str
    sample: Sample
    bin: str
    score: float
    threshold: float
    anomaly_map: np.ndarray


def collect(cache: FeatureCache, method_name: str, category: str, k: int,
            config: RunConfig) -> tuple[list[Entry], int]:
    """Enrol draw 0, fit S7, and catalogue every error; also returns the test size."""
    dataset = cache.dataset
    normals = dataset.train_samples(category)
    test = dataset.test_samples(category)
    support = draw_for_mode(normals, test, k, 0, config, category)
    method = build_run_method(cache, method_name, category, support.seed, config, None)
    reference = method.enrol(cache.features(support.samples))
    decision = fit_threshold_stage(cache, method, reference, normals, support, config)
    if decision is None:
        raise ValueError("the catalogue needs held-out normals; use k below the pool size")
    queries = query_samples(test, support)
    maps, _ = score_maps(method, reference, cache.features(queries))
    entries = []
    for sample, anomaly_map in zip(queries, maps):
        score = float(anomaly_map.max())
        found = classify(sample.label, score, anomaly_map, dataset.load_mask(sample),
                         decision.threshold)
        if found is not None:
            entries.append(Entry(category, sample, found, score, decision.threshold,
                                 anomaly_map))
    return entries, len(queries)


def count_table(entries: Sequence[Entry], sizes: dict[str, int]) -> str:
    """Markdown: one row per category, one column per bin, plus the error rate."""
    counts = Counter((entry.category, entry.bin) for entry in entries)
    lines = ["| Category | test images | " + " | ".join(BINS) + " | error rate |",
             "|---|---|" + "---|" * (len(BINS) + 1)]
    for category in sorted(sizes):
        cells = [str(counts[(category, name)]) for name in BINS]
        errors = sum(counts[(category, name)] for name in (*MISSED, *FALSE_ALARMS))
        rate = errors / sizes[category] if sizes[category] else 0.0
        lines.append(f"| {category} | {sizes[category]} | " + " | ".join(cells)
                     + f" | {rate:.3f} |")
    totals = [str(sum(counts[(c, name)] for c in sizes)) for name in BINS]
    lines.append(f"| **all** | {sum(sizes.values())} | " + " | ".join(totals) + " | |")
    return "\n".join(lines)


def examples(entries: Sequence[Entry], per_bin: int = EXAMPLES_PER_BIN) -> list[Entry]:
    """The highest-margin examples of each bin: the clearest evidence, not the borderline."""
    chosen: list[Entry] = []
    for name in BINS:
        in_bin = [entry for entry in entries if entry.bin == name]
        in_bin.sort(key=lambda entry: -abs(entry.score - entry.threshold))
        chosen.extend(in_bin[:per_bin])
    return chosen


def example_figure(dataset: MVTecAD, chosen: Sequence[Entry], path: Path) -> Path:
    """One row per example: input, ground truth, map with its peak marked."""
    rows = max(len(chosen), 1)
    figure, axes = plt.subplots(rows, 3, figsize=(8.4, 2.7 * rows))
    axes = np.atleast_2d(axes)
    for row, entry in enumerate(chosen):
        image = denormalise(dataset.load_image(entry.sample))
        peak_row, peak_column = peak(entry.anomaly_map)
        panels = [(image, None, f"{entry.bin}\n{entry.category}/{entry.sample.defect_type}"),
                  (dataset.load_mask(entry.sample) * 255, "gray", "ground truth"),
                  (entry.anomaly_map, "inferno",
                   f"score {entry.score:.2f} vs tau {entry.threshold:.2f}")]
        for column, (data, cmap, title) in enumerate(panels):
            axes[row][column].imshow(data, cmap=cmap)
            axes[row][column].set_title(title, fontsize=7)
            axes[row][column].axis("off")
        axes[row][2].plot(peak_column, peak_row, "c+", markersize=12, markeredgewidth=2)
    for axis in axes.ravel()[len(chosen) * 3:]:
        axis.set_visible(False)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(figure)
    return path


def build_catalogue(dataset: MVTecAD, extractor, categories: Sequence[str], method: str,
                    k: int, config: RunConfig, output: Path) -> dict[str, object]:
    """Catalogue every category, write the table, the figure and a JSON of the bins."""
    entries: list[Entry] = []
    sizes: dict[str, int] = {}
    for category in categories:
        cache = FeatureCache(dataset=dataset, extractor=extractor)
        found, size = collect(cache, method, category, k, config)
        entries.extend(found)
        sizes[category] = size
    output.mkdir(parents=True, exist_ok=True)
    table = count_table(entries, sizes)
    (output / "failure_catalogue.md").write_text(
        f"## Failure catalogue: {method}, k = {k}, draw 0, S7 threshold\n\n{table}\n",
        encoding="utf-8")
    example_figure(dataset, examples(entries), output / "failure_catalogue.png")
    payload = {"method": method, "k": k,
               "entries": [{"category": e.category, "image_id": e.sample.image_id,
                            "bin": e.bin, "score": e.score, "threshold": e.threshold}
                           for e in entries],
               "test_images": sizes}
    (output / "failure_catalogue.json").write_text(json.dumps(payload, indent=2),
                                                   encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data/mvtec_anomaly_detection")
    parser.add_argument("--dataset", default="mvtec_ad")
    parser.add_argument("--backbone", default="wide_resnet50_2")
    parser.add_argument("--categories", nargs="*", default=None)
    parser.add_argument("--method", default="patchcore")
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--output", default="output/task6")
    arguments = parser.parse_args()

    config = RunConfig(dataset=arguments.dataset, data_root=arguments.data_root,
                       backbone=arguments.backbone)
    dataset = build_dataset(config)
    categories = arguments.categories or dataset.categories
    payload = build_catalogue(dataset, build_extractor(config), categories, arguments.method,
                              arguments.k, config, Path(arguments.output))
    print(f"Catalogued {len(payload['entries'])} errors over {len(categories)} categories "
          f"-> {arguments.output}/failure_catalogue.md")


if __name__ == "__main__":
    main()

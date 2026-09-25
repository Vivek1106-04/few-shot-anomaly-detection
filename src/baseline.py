"""End-to-end smoke run: the trivial floor baseline, measured by the real harness.

This is **not** one of the three methods of week03 §S3 — it uses no backbone and no
feature space. It exists for two reasons:

  1. It exercises S1 -> (stand-in for S2-S4) -> S5 -> S6 -> S10 on real MVTec images,
     so the data loader, the seeded sampler and the metric code are shown working
     together before any method is added.
  2. It gives the **floor** every real method must beat. A k-vs-accuracy curve is only
     meaningful against a floor: a method that scores 0.62 I-AUROC at k=1 has learned
     something only if raw pixel differencing scores less.

Method: average the k support images, score a query by the blurred absolute pixel
difference from that average. Same S5 post-processing (upsample/smooth) and the same
S6 aggregation (max of the map) the real methods will use.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from data.mvtec import MVTecAD, Sample
from data.sampler import draw_support
from data.transforms import CROP_SIZE, denormalise
from metrics import Result, aggregate, evaluate

SMOOTH_SIGMA = 4.0  # the S5 smoothing constant, shared with the real methods
SMOOTH_KSIZE = (0, 0)  # derived from sigma by OpenCV


def enrol(dataset: MVTecAD, support: tuple[Sample, ...]) -> np.ndarray:
    """"Reference model" R: the per-pixel mean of the support images, (3, 224, 224)."""
    stacked = np.stack([dataset.load_image(sample) for sample in support])
    return stacked.mean(axis=0)


def score(dataset: MVTecAD, query: Sample, reference: np.ndarray) -> np.ndarray:
    """Anomaly map for one query: smoothed mean absolute difference from R."""
    difference = np.abs(dataset.load_image(query) - reference).mean(axis=0)
    return cv2.GaussianBlur(difference, SMOOTH_KSIZE, SMOOTH_SIGMA)


def run_draw(dataset: MVTecAD, category: str, k: int, draw: int, seed: int) -> tuple[Result, float]:
    """One (category, k, draw): enrol, score every test image, evaluate."""
    support = draw_support(dataset.train_samples(category), k, draw, seed)
    reference = enrol(dataset, support.samples)

    queries = dataset.test_samples(category)
    maps, masks, labels = [], [], []
    elapsed = 0.0
    for query in queries:
        started = time.perf_counter()
        anomaly_map = score(dataset, query, reference)
        elapsed += time.perf_counter() - started  # inference only: no mask loading
        maps.append(anomaly_map)
        masks.append(dataset.load_mask(query))
        labels.append(query.label)
    latency_ms = 1000.0 * elapsed / len(queries)

    maps_array = np.stack(maps)
    image_scores = maps_array.reshape(len(queries), -1).max(axis=1)  # S6a
    return evaluate(np.array(labels), image_scores, np.stack(masks), maps_array), latency_ms


def qualitative_figure(
    dataset: MVTecAD, category: str, k: int, seed: int, path: Path
) -> None:
    """Input / ground truth / anomaly map / overlay for one defective image."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    support = draw_support(dataset.train_samples(category), k, 0, seed)
    reference = enrol(dataset, support.samples)
    query = next(s for s in dataset.test_samples(category) if s.mask_path)
    anomaly_map = score(dataset, query, reference)
    image = denormalise(dataset.load_image(query))

    panels = [
        (f"{category} / {query.defect_type}", image, None),
        ("ground truth", dataset.load_mask(query) * 255, "gray"),
        (f"anomaly map (k={k})", anomaly_map, "inferno"),
    ]
    plt.figure(figsize=(11, 3.6))
    for index, (title, data, cmap) in enumerate(panels, start=1):
        plt.subplot(1, 4, index)
        plt.imshow(data, cmap=cmap)
        plt.title(title, fontsize=9)
        plt.axis("off")
    plt.subplot(1, 4, 4)
    plt.imshow(image)
    plt.imshow(anomaly_map, cmap="inferno", alpha=0.5)
    plt.title("overlay", fontsize=9)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="data/mvtec_anomaly_detection")
    parser.add_argument("--categories", nargs="*", default=["bottle", "grid", "screw"])
    parser.add_argument("--k", nargs="*", type=int, default=[1, 4, 16])
    parser.add_argument("--draws", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output", default="output")
    arguments = parser.parse_args()

    dataset = MVTecAD(arguments.root)
    output_dir = Path(arguments.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Floor baseline: mean support image, absolute pixel difference.")
    print(f"seed={arguments.seed}  draws per k={arguments.draws}\n")
    print(f"{'category':>10} {'k':>4} {'I-AUROC':>16} {'P-AUROC':>16} {'PRO':>16} {'ms/img':>8}")

    summary: dict[str, dict[str, dict[str, tuple[float, float]]]] = {}
    for category in arguments.categories:
        summary[category] = {}
        for k in arguments.k:
            results, latencies = [], []
            for draw in range(arguments.draws):
                result, latency = run_draw(dataset, category, k, draw, arguments.seed)
                results.append(result)
                latencies.append(latency)
            stats = aggregate(results)
            summary[category][str(k)] = stats
            print(f"{category:>10} {k:>4} "
                  f"{stats['I-AUROC'][0]:>8.3f}±{stats['I-AUROC'][1]:<7.3f}"
                  f"{stats['P-AUROC'][0]:>8.3f}±{stats['P-AUROC'][1]:<7.3f}"
                  f"{stats['PRO'][0]:>8.3f}±{stats['PRO'][1]:<7.3f}"
                  f"{np.mean(latencies):>8.1f}")

    (output_dir / "baseline_results.json").write_text(
        json.dumps({"seed": arguments.seed, "draws": arguments.draws, "results": summary},
                   indent=2),
        encoding="utf-8",
    )
    qualitative_figure(
        dataset, arguments.categories[0], max(arguments.k), arguments.seed,
        output_dir / "baseline_qualitative.png",
    )
    print(f"\nWrote {output_dir/'baseline_results.json'} and "
          f"{output_dir/'baseline_qualitative.png'}")


if __name__ == "__main__":
    main()

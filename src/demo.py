"""The single-image demonstration path: one image in, a verdict and a heat-map out.

This is Level 0 of the week03 flowchart run once, the way a line would run it:

    enrol (S1-S3, once per product)  ->  cached reference R + fitted threshold (S7)
    inspect (S1, S2, S4-S7, per part) ->  score, PASS/REJECT, heat-map, defect boxes

Enrolment draws k normal references from the product's train split (or takes the
files given with --support), fits the reference model, and fits the S7 threshold on
held-out normals (or the files given with --normals). The result is cached on disk
keyed by (dataset, category, method, backbone, k, seed, threshold rule) — week03 §S3's "R is
written to disk" — so a second inspection skips enrolment entirely.

The cache is a pickle this program wrote itself into its own output directory; do
not point --cache at files from anywhere else.

    ../venv/bin/python src/demo.py --category bottle \\
        --image data/mvtec_anomaly_detection/bottle/test/broken_large/000.png
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from config.schema import RunConfig  # noqa: E402
from data.mvtec import MVTecAD  # noqa: E402
from data.sampler import draw_support  # noqa: E402
from data.transforms import denormalise, preprocess_image  # noqa: E402
from decision import CONFORMAL, N_VALIDATION, Decision, fit_decision, validation_normals  # noqa: E402,E501
from experiment import TEXT_METHODS, build_extractor  # noqa: E402
from methods import build_method  # noqa: E402
from postprocess import binary_mask, image_score, overlay, to_pixel_map  # noqa: E402

FPR_BUDGET = 0.10
MIN_BOX_AREA = 32  # px: connected regions smaller than half a patch are not boxed
FIG_DPI = 130
SEED = 1234


@dataclass(frozen=True)
class Enrolled:
    """Everything inspection needs, and what produced it."""

    method_name: str
    method_parameters: dict[str, object]
    reference: object
    decision: Decision
    support_ids: tuple[str, ...]


@dataclass(frozen=True)
class Inspection:
    """The per-image output (week03 §S8): score, verdict, map, mask and boxes."""

    score: float
    verdict: str
    anomaly_map: np.ndarray
    mask: np.ndarray
    boxes: tuple[tuple[int, int, int, int], ...]  # (x, y, width, height)
    latency_ms: float

    def as_dict(self) -> dict[str, object]:
        return {"score": self.score, "verdict": self.verdict,
                "boxes": [list(box) for box in self.boxes], "latency_ms": self.latency_ms}


def read_image(path: str | Path) -> np.ndarray:
    """BGR uint8, failing loudly on a path that is not a readable image."""
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise OSError(f"could not read image: {path}")
    return image


def make_method(extractor, method_name: str, category: str, seed: int):
    parameters: dict[str, object] = {"seed": seed}
    if method_name in TEXT_METHODS:
        parameters["text"] = extractor.text_embeddings(category)
    return build_method(method_name, **parameters)


def maps_for(extractor, method, reference, images: list[np.ndarray]) -> np.ndarray:
    """S1 -> S2 -> S4 -> S5 for raw BGR images."""
    features = extractor.extract(np.stack([preprocess_image(image) for image in images]))
    return np.stack([to_pixel_map(method.score(feature, reference)) for feature in features])


def enrol_from_files(extractor, method_name: str, category: str, support: list[Path],
                     normals: list[Path]) -> Enrolled:
    """Enrol on the given support images; fit S7 on the given held-out normals."""
    if not support or not normals:
        raise ValueError("enrolling from files needs both --support and --normals images")
    method = make_method(extractor, method_name, category, SEED)
    images = [read_image(path) for path in support]
    reference = method.enrol(extractor.extract(np.stack([preprocess_image(i) for i in images])))
    maps = maps_for(extractor, method, reference, [read_image(path) for path in normals])
    return Enrolled(method_name, method.parameters, reference,
                    fit_decision(maps, FPR_BUDGET, CONFORMAL), tuple(str(path) for path in support))


def enrol_from_dataset(extractor, dataset: MVTecAD, method_name: str, category: str,
                       k: int) -> Enrolled:
    """Enrol on a seeded draw of k train normals; fit S7 on other train normals."""
    pool = dataset.train_samples(category)
    support = draw_support(pool, k, 0, SEED, category)
    method = make_method(extractor, method_name, category, support.seed % 2**31)
    images = [dataset.load_raw(sample) for sample in support.samples]
    reference = method.enrol(extractor.extract(np.stack([preprocess_image(i) for i in images])))
    validation = validation_normals(pool, support.samples, N_VALIDATION, support.seed)
    if not validation:
        raise ValueError(f"k={k} leaves no held-out normals to fit the threshold on")
    maps = maps_for(extractor, method, reference,
                    [dataset.load_raw(sample) for sample in validation])
    return Enrolled(method_name, method.parameters, reference,
                    fit_decision(maps, FPR_BUDGET, CONFORMAL),
                    tuple(sample.image_id for sample in support.samples))


def cached(path: Path, build) -> tuple[Enrolled, bool]:
    """Load an enrolment from `path`, or build and store it. Returns (enrolled, hit)."""
    if path.is_file():
        with open(path, "rb") as handle:
            return pickle.load(handle), True
    enrolled = build()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as handle:
        pickle.dump(enrolled, handle)
    return enrolled, False


def boxes_of(mask: np.ndarray) -> tuple[tuple[int, int, int, int], ...]:
    """Bounding boxes of the mask's connected regions, largest first."""
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
    found = [tuple(int(v) for v in stats[i, :4]) for i in range(1, count)
             if stats[i, cv2.CC_STAT_AREA] >= MIN_BOX_AREA]
    return tuple(sorted(found, key=lambda box: -box[2] * box[3]))


def inspect(extractor, enrolled: Enrolled, image: np.ndarray, category: str) -> Inspection:
    """Score one raw BGR image against an enrolment."""
    method = make_method(extractor, enrolled.method_name, category,
                         int(enrolled.method_parameters.get("seed", SEED)))
    started = time.perf_counter()
    anomaly_map = maps_for(extractor, method, enrolled.reference, [image])[0]
    score = image_score(anomaly_map)
    latency_ms = 1000.0 * (time.perf_counter() - started)
    mask = binary_mask(anomaly_map, enrolled.decision.pixel_threshold)
    verdict = enrolled.decision.verdict(score)
    return Inspection(score, verdict, anomaly_map, mask,
                      boxes_of(mask) if verdict == "REJECT" else (), latency_ms)


def render(image: np.ndarray, result: Inspection, decision: Decision, path: Path) -> Path:
    """input | calibrated heat-map overlay | defect mask with boxes."""
    rgb = np.ascontiguousarray(denormalise(preprocess_image(image)))
    # Colour range from the normal floor to the larger of pixel tau and this map's peak:
    # a clean part stays dark, and a defect's peak is the brightest thing on screen.
    top = max(decision.pixel_threshold, float(result.anomaly_map.max()))
    heat = overlay(rgb, result.anomaly_map, value_range=(decision.low, top))
    boxed = cv2.cvtColor(result.mask * 255, cv2.COLOR_GRAY2RGB)
    for x, y, width, height in result.boxes:
        cv2.rectangle(boxed, (x, y), (x + width, y + height), (0, 200, 255), 2)
        cv2.rectangle(rgb, (x, y), (x + width, y + height), (0, 200, 255), 2)
    figure, axes = plt.subplots(1, 3, figsize=(10, 3.6))
    panels = [(rgb, "input (S1 crop)"), (heat, "heat-map, normal floor = dark"),
              (boxed, f"defect mask @ pixel tau ({len(result.boxes)} regions)")]
    for axis, (data, title) in zip(axes, panels):
        axis.imshow(data)
        axis.set_title(title, fontsize=8)
        axis.axis("off")
    colour = "tab:red" if result.verdict == "REJECT" else "tab:green"
    figure.suptitle(f"{result.verdict}   score {result.score:.3f}   tau {decision.threshold:.3f}",
                    color=colour, fontsize=12, fontweight="bold")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(figure)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image", required=True, nargs="+")
    parser.add_argument("--category", required=True)
    parser.add_argument("--method", default="patchcore")
    parser.add_argument("--backbone", default="wide_resnet50_2")
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--data-root", default="data/mvtec_anomaly_detection")
    parser.add_argument("--support", nargs="*", default=None, help="own reference images")
    parser.add_argument("--normals", nargs="*", default=None, help="own held-out normals")
    parser.add_argument("--cache", default="output/demo/cache")
    parser.add_argument("--output", default="output/demo")
    arguments = parser.parse_args()

    extractor = build_extractor(RunConfig(backbone=arguments.backbone))
    if arguments.support:
        key = f"files-{arguments.category}-{arguments.method}-{arguments.backbone}-" \
              f"{len(arguments.support)}-{CONFORMAL}"
        build = lambda: enrol_from_files(  # noqa: E731
            extractor, arguments.method, arguments.category,
            [Path(p) for p in arguments.support], [Path(p) for p in arguments.normals or []])
    else:
        key = f"mvtec-{arguments.category}-{arguments.method}-{arguments.backbone}-" \
              f"k{arguments.k}-seed{SEED}-{CONFORMAL}"
        build = lambda: enrol_from_dataset(  # noqa: E731
            extractor, MVTecAD(arguments.data_root), arguments.method, arguments.category,
            arguments.k)
    enrolled, hit = cached(Path(arguments.cache) / f"{key}.pkl", build)
    print(f"enrolment {'loaded from cache' if hit else 'fitted'}: {key}")

    for image_path in arguments.image:
        image = read_image(image_path)
        result = inspect(extractor, enrolled, image, arguments.category)
        name = f"{Path(image_path).parent.name}_{Path(image_path).stem}_{result.verdict}.png"
        figure = render(image, result, enrolled.decision, Path(arguments.output) / name)
        print(json.dumps({"image": str(image_path), **result.as_dict(),
                          "threshold": enrolled.decision.threshold, "figure": str(figure)}))


if __name__ == "__main__":
    main()

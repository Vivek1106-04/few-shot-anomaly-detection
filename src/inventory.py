"""Dataset collection check: inventory an extracted MVTec AD tree and verify it.

Run once after download/extraction. It answers the three questions the project
actually depends on, rather than trusting the archive:

  1. Is every category present, with a normal-only train split?
  2. Does every defective test image have a ground-truth mask, and does that mask
     survive S1 preprocessing still aligned with its image?
  3. What are the image sizes, so the resize/crop in S1 is known not to be
     cropping the part out of frame?

Writes `output/dataset_inventory.md` and a preprocessing sanity figure.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from data.mvtec import MVTecAD, Sample
from data.transforms import CROP_SIZE, denormalise

SANITY_CATEGORIES = 3  # categories drawn into the preprocessing figure
MASK_OVERLAY_COLOUR = (255, 0, 0)
MASK_OVERLAY_ALPHA = 0.45


def image_size(dataset: MVTecAD, sample: Sample) -> tuple[int, int]:
    raw = dataset.load_raw(sample)
    return raw.shape[1], raw.shape[0]  # (width, height)


def check_category(dataset: MVTecAD, category: str) -> dict[str, object]:
    """Counts plus the two integrity checks, for one category."""
    counts = dataset.inventory(category)
    test = dataset.test_samples(category)
    anomalous = [s for s in test if s.label == 1]
    missing_masks = [s.image_id for s in anomalous if s.mask_path is None]

    # Alignment check: a defective image and its mask must agree in size before
    # preprocessing, and the preprocessed mask must still mark defect pixels after.
    misaligned: list[str] = []
    empty_after_crop: list[str] = []
    for sample in anomalous[:: max(1, len(anomalous) // 10)]:  # ~10 spot checks
        if sample.mask_path is None:
            continue
        raw = dataset.load_raw(sample)
        raw_mask = cv2.imread(str(sample.mask_path), cv2.IMREAD_GRAYSCALE)
        if raw_mask.shape[:2] != raw.shape[:2]:
            misaligned.append(sample.image_id)
        if dataset.load_mask(sample).sum() == 0:
            empty_after_crop.append(sample.image_id)

    width, height = image_size(dataset, dataset.train_samples(category)[0])
    return {
        "category": category,
        **counts,
        "image_size": f"{width}x{height}",
        "missing_masks": missing_masks,
        "misaligned_masks": misaligned,
        "empty_after_crop": empty_after_crop,
    }


def overlay_mask(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Paint the ground-truth mask over the preprocessed image for the figure."""
    painted = rgb.copy()
    colour = np.array(MASK_OVERLAY_COLOUR, dtype=np.float32)
    selection = mask > 0
    painted[selection] = (
        (1 - MASK_OVERLAY_ALPHA) * painted[selection] + MASK_OVERLAY_ALPHA * colour
    ).astype(np.uint8)
    return painted


def sanity_figure(dataset: MVTecAD, categories: list[str], path: Path) -> None:
    """Raw image / preprocessed image / preprocessed mask overlay, per category.

    This is the visual half of the alignment check: if S1 handled the mask with a
    different interpolation or a different crop than the image, the overlay would
    visibly slide off the defect.
    """
    rows = [c for c in categories if dataset.defect_types(c)][:SANITY_CATEGORIES]
    plt.figure(figsize=(10, 3.4 * len(rows)))
    for row, category in enumerate(rows):
        defective = next(s for s in dataset.test_samples(category) if s.mask_path)
        raw = cv2.cvtColor(dataset.load_raw(defective), cv2.COLOR_BGR2RGB)
        processed = denormalise(dataset.load_image(defective))
        mask = dataset.load_mask(defective)
        panels = [
            (f"{category}: raw {raw.shape[1]}x{raw.shape[0]}", raw),
            (f"S1 preprocessed {CROP_SIZE}x{CROP_SIZE}", processed),
            (f"mask overlay ({int(mask.sum())} px)", overlay_mask(processed, mask)),
        ]
        for column, (title, image) in enumerate(panels):
            plt.subplot(len(rows), 3, row * 3 + column + 1)
            plt.imshow(image)
            plt.title(title, fontsize=9)
            plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def write_report(rows: list[dict[str, object]], path: Path) -> None:
    """Markdown inventory table, one row per category plus the totals."""
    header = ("| Category | Train (normal) | Test normal | Test defective | Defect types "
              "| Masks | Image size |\n|---|---|---|---|---|---|---|\n")
    lines = [
        f"| {r['category']} | {r['train_normal']} | {r['test_normal']} | "
        f"{r['test_anomalous']} | {r['defect_types']} | "
        f"{r['masks_present']}/{r['test_anomalous']} | {r['image_size']} |"
        for r in rows
    ]
    totals = {key: sum(int(r[key]) for r in rows)
              for key in ("train_normal", "test_normal", "test_anomalous", "defect_types")}
    lines.append(
        f"| **total ({len(rows)})** | **{totals['train_normal']}** | "
        f"**{totals['test_normal']}** | **{totals['test_anomalous']}** | "
        f"**{totals['defect_types']}** | | |"
    )
    problems = [
        f"- {r['category']}: {name} -> {value}"
        for r in rows
        for name, value in (
            ("missing masks", r["missing_masks"]),
            ("misaligned masks", r["misaligned_masks"]),
            ("mask empty after crop", r["empty_after_crop"]),
        )
        if value
    ]
    body = header + "\n".join(lines) + "\n\n## Integrity checks\n\n"
    body += "No problems found.\n" if not problems else "\n".join(problems) + "\n"
    path.write_text("# MVTec AD inventory\n\n" + body, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="data/mvtec_anomaly_detection")
    parser.add_argument("--output", default="output")
    arguments = parser.parse_args()

    dataset = MVTecAD(arguments.root)
    output_dir = Path(arguments.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Root       : {dataset.root}")
    print(f"Categories : {len(dataset.categories)} -> {', '.join(dataset.categories)}")
    print(f"\n{'category':>12} {'train':>7} {'test ok':>8} {'test bad':>9} "
          f"{'types':>6} {'masks':>7} {'size':>12}")

    rows = []
    for category in dataset.categories:
        row = check_category(dataset, category)
        rows.append(row)
        print(f"{row['category']:>12} {row['train_normal']:>7} {row['test_normal']:>8} "
              f"{row['test_anomalous']:>9} {row['defect_types']:>6} "
              f"{row['masks_present']:>7} {row['image_size']:>12}")

    problems = sum(len(r[key]) for r in rows
                   for key in ("missing_masks", "misaligned_masks", "empty_after_crop"))
    print(f"\nTotal images: "
          f"{sum(int(r['train_normal']) + int(r['test_normal']) + int(r['test_anomalous']) for r in rows)}")
    print(f"Integrity problems: {problems}")

    write_report(rows, output_dir / "dataset_inventory.md")
    sanity_figure(dataset, dataset.categories, output_dir / "preprocessing_check.png")
    print(f"Wrote {output_dir/'dataset_inventory.md'} and {output_dir/'preprocessing_check.png'}")


if __name__ == "__main__":
    main()

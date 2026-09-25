"""Shared fixtures: a miniature MVTec-shaped dataset built on disk.

Building the fixture instead of pointing the tests at the real download keeps the
test suite runnable on a machine without the 5 GB archive, and lets the tests state
the expected counts exactly.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

IMAGE_SIZE = 300  # bigger than the 256 resize so the S1 crop path is exercised
N_TRAIN = 12
N_TEST_NORMAL = 4
N_TEST_DEFECT = 3
DEFECT_TYPES = ("scratch", "dent")


def _write_image(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.full((IMAGE_SIZE, IMAGE_SIZE, 3), value, dtype=np.uint8)
    cv2.imwrite(str(path), image)


def _write_mask(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
    cv2.rectangle(mask, (140, 140), (170, 170), 255, -1)  # centred: survives the crop
    cv2.imwrite(str(path), mask)


@pytest.fixture(scope="session")
def mvtec_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Two categories in the official layout, with masks for every defective image."""
    root = tmp_path_factory.mktemp("mvtec_ad")
    for category in ("widget", "gasket"):
        for index in range(N_TRAIN):
            _write_image(root / category / "train" / "good" / f"{index:03d}.png", 120)
        for index in range(N_TEST_NORMAL):
            _write_image(root / category / "test" / "good" / f"{index:03d}.png", 118)
        for defect in DEFECT_TYPES:
            for index in range(N_TEST_DEFECT):
                _write_image(root / category / "test" / defect / f"{index:03d}.png", 90)
                _write_mask(root / category / "ground_truth" / defect / f"{index:03d}_mask.png")
    return root


@pytest.fixture(scope="session")
def visa_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Two VisA categories in the official layout; only `pcb9` ships an annotation file."""
    root = tmp_path_factory.mktemp("visa")
    rows = ["object,split,label,image,mask"]
    for category in ("pcb9", "candle9"):
        images = root / category / "Data" / "Images"
        for index in range(N_TRAIN):
            rel = f"{category}/Data/Images/Normal/{index:04d}.JPG"
            _write_image(root / rel, 120)
            rows.append(f"{category},train,normal,{rel},")
        for index in range(N_TEST_NORMAL):
            rel = f"{category}/Data/Images/Normal/{100 + index:04d}.JPG"
            _write_image(root / rel, 118)
            rows.append(f"{category},test,normal,{rel},")
        annotations = ["image,label,mask"]
        for index in range(N_TEST_DEFECT):
            rel = f"{category}/Data/Images/Anomaly/{index:03d}.JPG"
            mask = f"{category}/Data/Masks/Anomaly/{index:03d}.png"
            _write_image(root / rel, 90)
            _write_mask(root / mask)
            rows.append(f"{category},test,anomaly,{rel},{mask}")
            label = "bent" if index == 0 else '"melt, scratch"'
            annotations.append(f"{rel},{label},{mask}")
        assert images.is_dir()
        if category == "pcb9":
            (root / category / "image_anno.csv").write_text("\n".join(annotations) + "\n")
    (root / "split_csv").mkdir()
    (root / "split_csv" / "1cls.csv").write_text("\n".join(rows) + "\n")
    return root

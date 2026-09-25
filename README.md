# Few-Shot Unsupervised Visual Anomaly Detection for Industrial Defect Inspection

Computer Vision Laboratory — Part B Project | 2023BCS0175

Detect and localise manufacturing defects from **normal images only** (unsupervised,
one-class) using **only a handful of reference images per product** (few-shot, k = 1…16).
Output is an image-level anomaly score plus a pixel-level defect heat-map.

## Layout

```
project/
  README.md          this file
  src/
    config/          run configuration (YAML) + schema with validation
    data/            transforms.py (S1), mvtec.py, visa.py, sampler.py (S9/S11),
                     augment.py (S3 support augmentation)
    models/          backbone.py (S2) — frozen WideResNet-50-2, layer2+layer3 taps
                     clip_backbone.py (S2) — frozen CLIP ViT-B/16, visual + text-aligned taps
    methods/         base.py (the S3/S4 interface), patchcore.py, padim.py, winclip.py
    postprocess.py   S5/S6 — upsample, smooth, image score (max and top-1 %), mask, overlay
    decision.py      S7 — threshold fitted on held-out normals, PASS / REJECT
    experiment.py    S9 — the sweep: category x method x k x draw, with provenance
    stress.py        S11a/b — coverage-gap and contaminated support sets, and what they measure
    failures.py      S11c — every error at the S7 threshold, binned by cause, with images
    profiling.py     S10 cost — peak memory and reference-model size
    report.py        S8 — k-vs-accuracy curves and result tables from result files
    tables.py        S8 — decision, cost, ceiling and stress tables
    qualitative.py   S8 — input / ground truth / map / overlay / decision grids
    demo.py          single-image demonstration: image in, verdict + heat-map + boxes out
    metrics.py       S10 — I-AUROC, P-AUROC, PRO, pixel-AP, F1-max, recall@FPR
    inventory.py     dataset integrity check -> output/dataset_inventory.md
    baseline.py      floor baseline: end-to-end smoke run on real images
  tests/             pytest suite (runs without the datasets; backbone tests need the weights)
  data/              datasets (MVTec AD, VisA) — not committed
  output/            figures, heat-maps, result tables (Task 6 in output/task6/)
```

## Setup

The scripts expect the virtual environment one level up, at `../venv` (shared with the
weekly labs in the original layout):

```bash
git clone https://github.com/Vivek1106-04/few-shot-anomaly-detection.git project
python3.12 -m venv venv && venv/bin/pip install -r project/requirements.txt
cd project
```

Datasets are not in the repository (12 GB). Extract them under `data/`:

- MVTec AD -> `data/mvtec_anomaly_detection/` (from mvtec.com, MVTec AD download page)
- VisA -> `data/visa/` (`VisA_20220922.tar` from the amazon-visual-anomaly S3 bucket)

Model weights (WRN-50-2 from torchvision, CLIP ViT-B/16 via open_clip) download on first
use. `output/` holds every result file and figure produced by the experiments.

## Running

```bash
../venv/bin/python -m pytest --cov=src --cov-report=term-missing   # test suite
../venv/bin/python src/inventory.py --root data/mvtec_anomaly_detection
../venv/bin/python src/baseline.py --root data/mvtec_anomaly_detection

# the sweep (S9): every category, both methods, k = 1..16, 3 seeded draws each
../venv/bin/python src/experiment.py --methods patchcore padim --k 1 2 4 8 16 --draws 3 \
    --output output/experiment_fewshot.json

# figures and tables from a result file (several files are merged, for a split sweep)
../venv/bin/python src/report.py --results output/experiment_fewshot.json
../venv/bin/python src/qualitative.py --categories bottle grid screw --method patchcore --k 4

# Task 6/7 sweeps
../venv/bin/python src/experiment.py --backbone clip_vit_b16 --methods winclip --tag clip \
    --k 1 4 16 --draws 3 --output output/task6/clip.json          # the few-shot method
../venv/bin/python src/experiment.py --mode contaminated --k 4 --tag contaminated  # S11b
../venv/bin/python src/experiment.py --dataset visa --data-root data/visa         # VisA
../venv/bin/python src/failures.py --method patchcore --k 4 --output output/task6  # S11c

# the demonstration: enrol once (cached), then inspect any number of images
../venv/bin/python src/demo.py --category bottle --k 4 \
    --image data/mvtec_anomaly_detection/bottle/test/broken_large/000.png
# ...or enrol on your own product photos
../venv/bin/python src/demo.py --category widget --support ref1.png ref2.png \
    --normals ok1.png ok2.png ok3.png ok4.png --image part.png
```

## Stack

Python 3, OpenCV, NumPy, matplotlib, PyYAML, pytest, PyTorch + torchvision (the frozen
S2 backbone), open_clip (the CLIP backbone). Torch is confined to `src/models/`: the
methods take feature arrays, so they are plain numpy and their tests need neither
weights nor a GPU.
Metrics are implemented directly on numpy rather
than via scikit-learn, so the definitions used — PRO in particular — are visible and
testable. Extends the environment used for the weekly labs (`../venv`).

## Datasets

- **MVTec AD** — 15 categories, ~5 000 images, pixel-level masks. Primary benchmark.
- **VisA** — 12 categories, ~10 800 images, smaller defects. Secondary/generalisation.

## Metrics

Image AUROC, pixel AUROC, PRO, F1-max, recall @ fixed FPR, inference latency.

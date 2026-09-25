# Project Specification

**Few-Shot Unsupervised Visual Anomaly Detection for Industrial Defect Inspection**

| | |
|---|---|
| Course | Computer Vision Laboratory — Part B Project |
| Roll No. | 2023BCS0175 |
| Started | Week of 14 August 2026 |

This document holds the project's **scope and standing definitions** — the parts that do
not change week to week.

**Weekly tasks are assigned by the instructor**, one at a time. They are not planned in
advance here. Each assigned task gets its own report (submitted to the LMS, not kept here), and its row is
added to the log in §5 once assigned.

The problem statement, motivation, application and objectives are defined in
the week 1 report and are
not repeated here.

---

## 1. Scope

**In scope.** One-class (normal-images-only) anomaly detection with few-shot support
sets (*k* = 1…16), producing an image-level anomaly score and a pixel-level anomaly map;
evaluation on MVTec AD (primary) and VisA (secondary); comparison of PaDiM, PatchCore
and one few-shot-specific method under one protocol.

**Out of scope.** Defect classification into named types; training a backbone from
scratch; PLC/line-control integration; 3-D or multi-spectral input; shipping a
production system (latency is measured, not deployed).

**Central intended result.** The ***k*-vs-accuracy curve** — how detection and
localisation accuracy degrade as the number of normal reference images shrinks.

---

## 2. Success Criteria

Stated up front as falsifiable targets. Where a target is missed, the report states by
how much and why, rather than moving the target.

| Metric | Target (few-shot, *k* = 4) |
|---|---|
| Image AUROC | ≥ 0.90 mean over categories |
| Pixel AUROC | ≥ 0.93 |
| PRO (per-region overlap) | ≥ 0.85 |
| Recall @ FPR ≤ 0.10 | ≥ 0.95 |
| Inference latency | ≤ 100 ms/image, single GPU |

---

## 3. Repository Layout

```
project/
  SPEC.md            this file — scope, success criteria, weekly log
  README.md          index of weekly reports
  src/               code
  data/              MVTec AD, VisA — not committed
  output/            figures, heat-maps, result tables
```

## 4. Environment

Python 3, OpenCV, PyTorch, torchvision, open_clip, NumPy, matplotlib — extends the
virtual environment already used for the weekly labs (`../venv`), pinned in
`requirements.txt`. (scikit-learn, listed here originally, was never needed: the
metrics are implemented directly on NumPy.)

---

## 5. Weekly Log

One row per task as the instructor assigns it.

| Week | Task as assigned | Report | Status |
|---|---|---|---|
| 1 | Problem definition & project write-up — problem statement, motivation/application, objectives | week01 | Complete |
| 2 | Literature survey (8–10 papers, comparison table) & research gap identification | week02 | Complete |
| 3 | Flowchart / system architecture & explanation of the proposed methodology | week03 | Complete |
| 4 | Implementation Part 1 — dataset collection, pre-processing, ~25% of the methodology, remaining-work list | week04 | Complete |
| 5 | Implementation Part 2 — continue the methodology to ~50%, integrate and test the modules, document results, list remaining work | week05 | Complete |
| 6 | Implementation Part 3 — complete ~75% of the work, integrate into a working prototype, demonstrate with outputs, list the remaining 25% | week06 | Complete |
| 7 | Implementation Part 4 — complete the entire methodology, integrate and test all modules, final report with source code | week07 | Complete |

---

## 6. Submission Protocol

Each week's report is uploaded to the LMS **separately** as its own PDF. The compiled
set becomes the final report for documentation. Final evaluation is the PPT plus the
project demonstration; the weekly submissions carry the evaluation weight.

"""S9 — the experiment runner: the loops that produce the k-vs-accuracy curve.

The sweep is dataset x category x method x k x draw. Everything it needs already
exists: support sets come from the seeded sampler, scoring from a method behind the
S3/S4 interface, post-processing from S5/S6, the threshold from S7, numbers from
S10. This file adds only the loops, the feature cache and the result file.

Two properties it is built to guarantee:

  * **Every number is traceable.** A record carries the method's parameters, the
    support set's provenance (the exact image ids), the per-draw seed, the fitted
    decision threshold, and the configuration the sweep ran under.
  * **Every method sees identical features.** Features are extracted once per image
    and cached, so PaDiM and PatchCore at the same (category, k, draw) are scoring
    the same arrays, not two independent runs of the backbone.

Latency is reported as extraction + scoring, with extraction timed on uncached
images only, so the cache speeds the sweep up without flattering the measurement.
Peak memory is measured on a separate, untimed call (draw 0 only), because tracing
allocations slows the call being traced.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Protocol

import numpy as np

from config.schema import RunConfig, load_config
from data.augment import augment
from data.mvtec import MVTecAD, Sample
from data.sampler import resolve_k
from data.transforms import preprocess_image
from decision import Decision, OperatingPoint, fit_decision, operating_point, validation_normals
from methods import build_method
from metrics import Result, aggregate, auroc, evaluate
from postprocess import image_score, robust_image_score, to_pixel_map
from profiling import peak_memory_mb, reference_size_mb
from stress import (
    contamination_metrics,
    coverage_metrics,
    draw_for_mode,
    held_out_normals,
    query_samples,
    validation_pool,
)

MILLISECONDS = 1000.0
TEXT_METHODS = ("winclip",)  # methods that take the category's prompt embeddings


class FeatureExtractor(Protocol):
    """What the runner needs from S2 — and nothing more, so tests can fake it."""

    def extract(self, images: np.ndarray) -> np.ndarray: ...


@dataclass
class FeatureCache:
    """Per-image features, extracted once and kept as float16.

    float16 halves the footprint of a category's test split (a 224x224 image yields
    1536 x 28 x 28 values) at a precision that is far below the distances any metric
    resolves. Cleared between categories, since nothing is shared across them.
    """

    dataset: MVTecAD
    extractor: FeatureExtractor
    store: dict[str, np.ndarray] = field(default_factory=dict)
    texts: dict[str, np.ndarray] = field(default_factory=dict)
    extract_ms: float = 0.0
    n_extracted: int = 0

    def features(self, samples: Sequence[Sample]) -> np.ndarray:
        """(N, C, H, W) float32 features for these samples, in the order given."""
        missing = [s for s in samples if s.image_id not in self.store]
        if missing:
            images = np.stack([self.dataset.load_image(s) for s in missing])
            started = time.perf_counter()
            extracted = self.extractor.extract(images)
            self.extract_ms += MILLISECONDS * (time.perf_counter() - started)
            self.n_extracted += len(missing)
            for sample, feature in zip(missing, extracted):
                self.store[sample.image_id] = feature.astype(np.float16)
        return np.stack([self.store[s.image_id].astype(np.float32) for s in samples])

    def support_features(self, samples: Sequence[Sample], n_copies: int, seed: int
                         ) -> np.ndarray:
        """The support features, plus `n_copies` S3-augmented copies of each image.

        Augmented copies are extracted fresh (they are never queried twice) and are
        not counted in the per-image latency, which is an inference-path figure.
        """
        originals = self.features(samples)
        if n_copies == 0:
            return originals
        raw = [self.dataset.load_raw(sample) for sample in samples]
        copies = np.stack([preprocess_image(image) for image in augment(raw, n_copies, seed)])
        return np.concatenate([originals, self.extractor.extract(copies)])

    def text(self, category: str) -> np.ndarray:
        """The category's [normal, anomalous] prompt embeddings (CLIP extractors only)."""
        if category not in self.texts:
            embed = getattr(self.extractor, "text_embeddings", None)
            if embed is None:
                raise ValueError("this method needs a text-aligned backbone (clip_vit_b16)")
            self.texts[category] = embed(category)
        return self.texts[category]

    @property
    def ms_per_image(self) -> float:
        """Mean backbone cost per image, over images that were actually extracted."""
        return self.extract_ms / self.n_extracted if self.n_extracted else 0.0

    def clear(self) -> None:
        self.store.clear()


@dataclass(frozen=True)
class RunRecord:
    """One (method, category, k, draw): its metrics, its costs and its provenance."""

    method: str
    category: str
    k: int
    draw: int
    result: Result
    enrol_ms: float
    score_ms_per_image: float
    extract_ms_per_image: float
    method_parameters: dict[str, object]
    provenance: dict[str, object]
    decision: dict[str, object] | None = None
    memory: dict[str, float] | None = None
    stress: dict[str, object] = field(default_factory=dict)

    @property
    def latency_ms(self) -> float:
        """What an inference costs end to end: backbone plus method."""
        return self.extract_ms_per_image + self.score_ms_per_image

    def as_dict(self) -> dict[str, object]:
        return {
            "method": self.method,
            "category": self.category,
            "k": self.k,
            "draw": self.draw,
            "metrics": self.result.as_dict(),
            "enrol_ms": self.enrol_ms,
            "score_ms_per_image": self.score_ms_per_image,
            "extract_ms_per_image": self.extract_ms_per_image,
            "latency_ms": self.latency_ms,
            "method_parameters": self.method_parameters,
            "support": self.provenance,
            "decision": self.decision,
            "memory": self.memory,
            "stress": self.stress,
        }


def score_maps(method, reference, features: np.ndarray) -> tuple[np.ndarray, float]:
    """S4 + S5 for a stack of query features: pixel maps, and the seconds it took."""
    maps, seconds = [], 0.0
    for feature in features:
        started = time.perf_counter()
        maps.append(to_pixel_map(method.score(feature, reference)))
        seconds += time.perf_counter() - started
    return np.stack(maps), seconds


def measure_memory(method, support: np.ndarray, query: np.ndarray) -> dict[str, float]:
    """Peak working memory of enrolment and of one query, and the size of R."""
    reference, enrol_peak = peak_memory_mb(lambda: method.enrol(support))
    _, score_peak = peak_memory_mb(lambda: method.score(query, reference))
    return {"enrol_peak_mb": enrol_peak, "score_peak_mb": score_peak,
            "reference_mb": reference_size_mb(reference)}


def build_run_method(cache: FeatureCache, method_name: str, category: str, seed: int,
                     config: RunConfig, overrides: dict[str, object] | None):
    """The method, with config parameters, per-draw seed and (if needed) prompt text."""
    parameters = {**config.method_parameters.get(method_name, {}), **(overrides or {})}
    # The per-draw seed reaches the method too: PatchCore's coreset is a random
    # choice, and it must be reproducible from the result file like everything else.
    parameters.setdefault("seed", seed % (2**31))
    if method_name in TEXT_METHODS:
        parameters.setdefault("text", cache.text(category))
    return build_method(method_name, **parameters)


def fit_threshold_stage(cache: FeatureCache, method, reference, normals: Sequence[Sample],
                        support, config: RunConfig) -> Decision | None:
    """S7 on held-out normals; None when no normal is left over (k = full)."""
    pool = validation_pool(normals, support)
    validation = validation_normals(pool, support.samples, config.n_validation, support.seed)
    if not validation:
        return None
    maps, _ = score_maps(method, reference, cache.features(validation))
    return fit_decision(maps, config.fpr_budget, config.threshold_rule)


def run_single(
    cache: FeatureCache,
    method_name: str,
    category: str,
    k: int | str,
    draw: int,
    config: RunConfig,
    method_parameters: dict[str, object] | None = None,
) -> RunRecord:
    """Enrol on one drawn support set, score the test split, fit S7, evaluate."""
    dataset = cache.dataset
    normals = dataset.train_samples(category)
    test = dataset.test_samples(category)
    support = draw_for_mode(normals, test, k, draw, config, category)
    method = build_run_method(cache, method_name, category, support.seed, config,
                              method_parameters)

    support_features = cache.support_features(support.samples, config.augment, support.seed)
    started = time.perf_counter()
    reference = method.enrol(support_features)
    enrol_ms = MILLISECONDS * (time.perf_counter() - started)

    queries = query_samples(test, support)
    query_features = cache.features(queries)
    maps, score_seconds = score_maps(method, reference, query_features)
    labels = np.array([sample.label for sample in queries])
    masks = np.stack([dataset.load_mask(sample) for sample in queries])
    scores = np.array([image_score(single) for single in maps])
    result = evaluate(labels, scores, masks, maps, config.fpr_budget)
    result = replace(result, image_auroc_robust=auroc(
        labels, np.array([robust_image_score(single) for single in maps])))

    decision = fit_threshold_stage(cache, method, reference, normals, support, config)
    point = (OperatingPoint.undefined() if decision is None
             else operating_point(labels, scores, decision.threshold, config.fpr_budget))
    result = replace(result, decision_recall=point.recall, decision_fpr=point.fpr)

    stress = stress_measurements(cache, method, reference, normals, support, queries, scores,
                                 decision)
    memory = measure_memory(method, support_features, query_features[0]) if draw == 0 else None
    return RunRecord(
        method=f"{method_name}+{config.tag}" if config.tag else method_name,
        category=category,
        k=len(support.samples),
        draw=draw,
        result=result,
        enrol_ms=enrol_ms,
        score_ms_per_image=MILLISECONDS * score_seconds / len(queries),
        extract_ms_per_image=cache.ms_per_image,
        method_parameters={**method.parameters, "augment": config.augment},
        provenance=support.provenance,
        decision=None if decision is None else {**decision.as_dict(),
                                                 "meets_target": point.meets_target},
        memory=memory,
        stress=stress,
    )


def stress_measurements(cache: FeatureCache, method, reference, normals, support, queries,
                        scores: np.ndarray, decision: Decision | None) -> dict[str, object]:
    """S11a/b measurements for this run's support mode; empty for a uniform run."""
    if decision is None or support.mode == "uniform":
        return {}
    if support.mode == "contaminated":
        return contamination_metrics(support, queries, scores > decision.threshold)
    held_out = held_out_normals(normals, support)
    held_scores = (np.array([], dtype=np.float64) if not held_out else np.array([
        image_score(single)
        for single in score_maps(method, reference, cache.features(held_out))[0]
    ]))
    return {"held_out_group": support.held_out_group,
            **coverage_metrics(held_scores, decision.threshold)}


def run_sweep(
    dataset: MVTecAD,
    extractor: FeatureExtractor,
    config: RunConfig,
    verbose: bool = True,
) -> list[RunRecord]:
    """The full loop. Features are cached per category and freed before the next."""
    categories = list(config.categories) if config.categories else dataset.categories
    records: list[RunRecord] = []
    for category in categories:
        cache = FeatureCache(dataset=dataset, extractor=extractor)
        pool = len(dataset.train_samples(category))
        for method_name in config.methods:
            for k in config.k_values:
                size = resolve_k(k, pool)
                if size > pool:
                    raise ValueError(f"k={k} exceeds the {pool}-image train pool of {category}")
                draws = 1 if size == pool else config.n_draws  # the ceiling run is unique
                for draw in range(draws):
                    record = run_single(cache, method_name, category, k, draw, config)
                    records.append(record)
                if verbose:
                    _print_row(records[-draws:], records[-1].method, category, size)
        cache.clear()
    return records


def _print_row(records: list[RunRecord], method: str, category: str, k: int) -> None:
    stats = aggregate([record.result for record in records])
    latency = float(np.mean([record.latency_ms for record in records]))
    print(
        f"{method:>16} {category:>12} {k:>5} "
        f"{stats['I-AUROC'][0]:>7.3f}+-{stats['I-AUROC'][1]:<6.3f}"
        f"{stats['P-AUROC'][0]:>7.3f}+-{stats['P-AUROC'][1]:<6.3f}"
        f"{stats['PRO'][0]:>7.3f}+-{stats['PRO'][1]:<6.3f}"
        f"{stats['recall@tau'][0]:>7.3f}{latency:>8.1f}",
        flush=True,
    )


def _mean_of(values: list[float]) -> float:
    finite = [value for value in values if value is not None and np.isfinite(value)]
    return float(np.mean(finite)) if finite else float("nan")


def summarise(records: Sequence[RunRecord]) -> dict[str, dict[str, dict[str, object]]]:
    """method -> category -> k -> {metric: [mean, std]}, plus costs and stress means."""
    summary: dict[str, dict[str, dict[str, object]]] = {}
    for record in records:
        by_category = summary.setdefault(record.method, {}).setdefault(record.category, {})
        by_category.setdefault(str(record.k), {"records": []})["records"].append(record)
    for methods in summary.values():
        for categories in methods.values():
            for key, holder in categories.items():
                group: list[RunRecord] = holder.pop("records")
                holder.update(aggregate([record.result for record in group]))
                holder["latency_ms"] = float(np.mean([r.latency_ms for r in group]))
                holder["enrol_ms"] = float(np.mean([r.enrol_ms for r in group]))
                holder["n_draws"] = len(group)
                measured = [r.memory for r in group if r.memory is not None]
                if measured:
                    holder["memory"] = {name: _mean_of([m[name] for m in measured])
                                        for name in measured[0]}
                numeric = sorted({name for r in group for name, value in r.stress.items()
                                  if isinstance(value, (int, float))})
                if numeric:
                    holder["stress"] = {name: _mean_of([r.stress.get(name) for r in group])
                                        for name in numeric}
    return summary


def write_results(path: Path, config: RunConfig, records: Sequence[RunRecord]) -> None:
    """One file holding the numbers, the configuration and every support set."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": config.as_dict(),
        "summary": summarise(records),
        "runs": [record.as_dict() for record in records],
    }
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def build_extractor(config: RunConfig) -> FeatureExtractor:
    """The S2 backbone named by the configuration."""
    if config.backbone == "clip_vit_b16":
        from models.clip_backbone import CLIPBackbone

        return CLIPBackbone()
    from models.backbone import FrozenBackbone

    return FrozenBackbone(name=config.backbone)


def build_dataset(config: RunConfig) -> MVTecAD:
    """The benchmark named by the configuration, behind one loader interface."""
    if config.dataset == "visa":
        from data.visa import VisA

        return VisA(config.data_root)
    return MVTecAD(config.data_root)


def parse_parameters(pairs: Sequence[str]) -> dict[str, dict[str, object]]:
    """['winclip.language_weight=1.0'] -> {'winclip': {'language_weight': 1.0}}."""
    parsed: dict[str, dict[str, object]] = {}
    for pair in pairs:
        name, _, value = pair.partition("=")
        method, _, parameter = name.partition(".")
        if not value or not parameter:
            raise ValueError(f"expected method.parameter=value, got {pair!r}")
        parsed.setdefault(method, {})[parameter] = json.loads(value)
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="src/config/default.yaml")
    parser.add_argument("--dataset", default=None, choices=("mvtec_ad", "visa"))
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--backbone", default=None)
    parser.add_argument("--categories", nargs="*", default=None)
    parser.add_argument("--methods", nargs="*", default=None)
    parser.add_argument("--k", nargs="*", default=None, help="ints, or 'full'")
    parser.add_argument("--draws", type=int, default=None)
    parser.add_argument("--mode", default=None, dest="support_mode")
    parser.add_argument("--augment", type=int, default=None)
    parser.add_argument("--param", nargs="*", default=None,
                        help="method parameters, e.g. winclip.language_weight=1.0")
    parser.add_argument("--threshold-rule", default=None, choices=("quantile", "conformal"))
    parser.add_argument("--tag", default=None, help="variant label appended to method names")
    parser.add_argument("--output", default=None)
    arguments = parser.parse_args()

    config = load_config(arguments.config)
    overrides: dict[str, object] = {}
    for name in ("dataset", "data_root", "backbone", "support_mode", "augment", "tag",
                 "threshold_rule"):
        if getattr(arguments, name) is not None:
            overrides[name] = getattr(arguments, name)
    if arguments.categories is not None:
        overrides["categories"] = tuple(arguments.categories)
    if arguments.methods is not None:
        overrides["methods"] = tuple(arguments.methods)
    if arguments.k is not None:
        overrides["k_values"] = tuple(int(k) if k != "full" else "full" for k in arguments.k)
    if arguments.draws is not None:
        overrides["n_draws"] = arguments.draws
    if arguments.param is not None:
        overrides["method_parameters"] = parse_parameters(arguments.param)
    config = RunConfig(**{**config.as_dict(), **overrides})

    dataset = build_dataset(config)
    extractor = build_extractor(config)
    print(f"{'method':>16} {'category':>12} {'k':>5} {'I-AUROC':>14}{'P-AUROC':>14}"
          f"{'PRO':>14}{'rec@tau':>7}{'ms/img':>8}", flush=True)
    records = run_sweep(dataset, extractor, config)

    output = Path(arguments.output or Path(config.output_dir) / "experiment_results.json")
    write_results(output, config, records)
    print(f"\nWrote {output} ({len(records)} runs)")


if __name__ == "__main__":
    main()

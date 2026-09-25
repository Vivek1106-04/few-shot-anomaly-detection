"""Run configuration — the one file that drives the S9 harness.

Every field that can change a number is here and nowhere else, and the loaded
config is written verbatim into each result file (week03 §S9), so a run is
reproducible from its output alone.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

KNOWN_METHODS = ("padim", "patchcore", "winclip")
KNOWN_BACKBONES = ("wide_resnet50_2", "clip_vit_b16")
KNOWN_MODES = ("uniform", "coverage_gap", "contaminated")
KNOWN_DATASETS = ("mvtec_ad", "visa")


@dataclass(frozen=True)
class RunConfig:
    """A full experiment sweep: methods x categories x k x draws."""

    dataset: str = "mvtec_ad"
    data_root: str = "data/mvtec_anomaly_detection"
    categories: tuple[str, ...] = ()  # empty means every category in the dataset
    methods: tuple[str, ...] = ("patchcore",)
    backbone: str = "wide_resnet50_2"
    k_values: tuple[int | str, ...] = (1, 2, 4, 8, 16, "full")
    n_draws: int = 5
    seed: int = 1234
    support_mode: str = "uniform"
    n_contaminated: int = 1
    fpr_budget: float = 0.10
    n_validation: int = 20  # held-out train normals the S7 threshold is fitted on
    threshold_rule: str = "quantile"  # S7 rank rule: quantile | conformal (decision.py)
    augment: int = 0  # S3 augmented copies per support image (0 = off)
    method_parameters: dict[str, dict[str, object]] = field(default_factory=dict)
    tag: str = ""  # variant label: records are named "<method>+<tag>" so result files merge
    output_dir: str = "output"
    extras: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.dataset not in KNOWN_DATASETS:
            raise ValueError(f"unknown dataset {self.dataset!r}; known: {KNOWN_DATASETS}")
        if self.threshold_rule not in ("quantile", "conformal"):
            raise ValueError(f"unknown threshold_rule {self.threshold_rule!r}")
        if self.n_validation < 1:
            raise ValueError(f"n_validation must be >= 1, got {self.n_validation}")
        if self.augment < 0:
            raise ValueError(f"augment must be >= 0, got {self.augment}")
        for method in self.method_parameters:
            if method not in KNOWN_METHODS:
                raise ValueError(f"method_parameters names unknown method {method!r}")
        for method in self.methods:
            if method not in KNOWN_METHODS:
                raise ValueError(f"unknown method {method!r}; known: {KNOWN_METHODS}")
        if self.backbone not in KNOWN_BACKBONES:
            raise ValueError(f"unknown backbone {self.backbone!r}; known: {KNOWN_BACKBONES}")
        if self.support_mode not in KNOWN_MODES:
            raise ValueError(f"unknown support_mode {self.support_mode!r}; known: {KNOWN_MODES}")
        if self.n_draws < 1:
            raise ValueError(f"n_draws must be >= 1, got {self.n_draws}")
        if not 0.0 < self.fpr_budget <= 1.0:
            raise ValueError(f"fpr_budget must be in (0, 1], got {self.fpr_budget}")
        for k in self.k_values:
            if k != "full" and (not isinstance(k, int) or k < 1):
                raise ValueError(f"k values must be positive ints or 'full', got {k!r}")

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def load_config(path: str | Path) -> RunConfig:
    """Read a YAML config, rejecting unknown keys instead of ignoring them.

    A typo in a config key is otherwise the quietest way to publish a number that
    was produced by a different setting than the one written in the report.
    """
    with open(path, encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"config must be a mapping, got {type(raw).__name__}")
    known = {f for f in RunConfig.__dataclass_fields__}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"unknown config keys: {sorted(unknown)}")
    for key in ("categories", "methods", "k_values"):
        if key in raw and raw[key] is not None:
            raw[key] = tuple(raw[key])
    return RunConfig(**raw)

"""S8 — the Task 6 result tables, read from a result file's summary like `report.py`.

Four tables, one per question the new modules answer:

  * **decision_table** — S7: what the threshold fitted on held-out normals does on
    the test split, against the oracle recall@FPR the Task 5 report had to use.
  * **cost_table** — S10 cost: latency, enrolment time, peak memory, reference size.
  * **ceiling_table** — each few-shot number divided by the same category's k = full
    ceiling, so "PaDiM at k = 4 reaches 80 % of what PaDiM can reach" is a number.
  * **stress_table** — S11a/b: false alarms on a withheld mode of normality, and
    recall on a defect type that was enrolled as normal.

Every mean below is over categories; categories where a quantity is undefined (NaN,
e.g. the threshold at k = full) are left out of that mean and the count is shown.
"""

from __future__ import annotations

import numpy as np

from decision import TARGET_RECALL
from metrics import DEFAULT_FPR_BUDGET

ORACLE = f"recall@FPR<={DEFAULT_FPR_BUDGET}"


def _k_values(summary: dict, method: str) -> list[str]:
    return sorted({k for category in summary[method].values() for k in category}, key=int)


def _values(summary: dict, method: str, k: str, pick) -> list[float]:
    """pick(entry) for every category that ran this (method, k), NaNs dropped."""
    found = []
    for category in summary[method].values():
        if k in category:
            value = pick(category[k])
            if value is not None and np.isfinite(value):
                found.append(float(value))
    return found


def _mean(values: list[float]) -> str:
    return f"{np.mean(values):.3f}" if values else "-"


def _metric(name: str):
    return lambda entry: entry[name][0] if name in entry else None


def _nested(group: str, name: str):
    return lambda entry: entry.get(group, {}).get(name)


def decision_table(summary: dict) -> str:
    """Per (method, k): oracle recall, S7 recall and FPR, categories meeting the target."""
    lines = ["| Method | k | recall @ FPR<=0.1 (oracle tau) | recall @ S7 tau | "
             "FPR @ S7 tau | categories meeting both |",
             "|---|---|---|---|---|---|"]
    for method in sorted(summary):
        for k in _k_values(summary, method):
            recalls = _values(summary, method, k, _metric("recall@tau"))
            fprs = _values(summary, method, k, _metric("FPR@tau"))
            meeting = sum(
                1 for category in summary[method].values() if k in category
                and "recall@tau" in category[k]
                and category[k]["recall@tau"][0] >= TARGET_RECALL
                and category[k]["FPR@tau"][0] <= DEFAULT_FPR_BUDGET)
            lines.append(f"| {method} | {k} | {_mean(_values(summary, method, k, _metric(ORACLE)))}"
                         f" | {_mean(recalls)} | {_mean(fprs)} | {meeting} of {len(recalls)} |")
    return "\n".join(lines)


def cost_table(summary: dict) -> str:
    """Per (method, k): ms/image, enrolment ms, peak enrol/score memory, size of R."""
    lines = ["| Method | k | ms/image | enrol ms | enrol peak MiB | score peak MiB | R MiB |",
             "|---|---|---|---|---|---|---|"]
    for method in sorted(summary):
        for k in _k_values(summary, method):
            cells = [_mean(_values(summary, method, k, lambda e: e.get("latency_ms"))),
                     _mean(_values(summary, method, k, lambda e: e.get("enrol_ms")))]
            cells += [_mean(_values(summary, method, k, _nested("memory", name)))
                      for name in ("enrol_peak_mb", "score_peak_mb", "reference_mb")]
            lines.append(f"| {method} | {k} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def ceiling_ratios(summary: dict, ceiling: dict, method: str, metric: str
                   ) -> dict[str, list[float]]:
    """k -> per-category few-shot/ceiling ratios, for categories that have both."""
    reference = ceiling.get(f"{method}+full", ceiling.get(method, {}))
    ratios: dict[str, list[float]] = {}
    for category, entries in summary[method].items():
        top = next(iter(reference.get(category, {}).values()), None)
        if top is None or top[metric][0] <= 0:
            continue
        for k, entry in entries.items():
            ratios.setdefault(k, []).append(entry[metric][0] / top[metric][0])
    return ratios


def ceiling_table(summary: dict, ceiling: dict, metric: str = "I-AUROC") -> str:
    """Mean over categories of (few-shot / own ceiling), per (method, k)."""
    lines = [f"| Method | k | {metric} as a fraction of the k = full ceiling | categories |",
             "|---|---|---|---|"]
    for method in sorted(summary):
        ratios = ceiling_ratios(summary, ceiling, method, metric)
        for k in sorted(ratios, key=int):
            lines.append(f"| {method} | {k} | {_mean(ratios[k])} | {len(ratios[k])} |")
    return "\n".join(lines)


def ceiling_values(ceiling: dict, metrics: tuple[str, ...] = ("I-AUROC", "P-AUROC", "PRO")
                   ) -> str:
    """The ceiling itself: mean over categories at k = full."""
    lines = ["| Method | " + " | ".join(metrics) + " | ms/image | enrol ms |",
             "|---|" + "---|" * (len(metrics) + 2)]
    for method in sorted(ceiling):
        entries = [next(iter(c.values())) for c in ceiling[method].values()]
        cells = [f"{np.mean([e[m][0] for e in entries]):.3f}" for m in metrics]
        cells += [f"{np.mean([e['latency_ms'] for e in entries]):.1f}",
                  f"{np.mean([e['enrol_ms'] for e in entries]):.0f}"]
        lines.append(f"| {method} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def stress_table(summary: dict) -> str:
    """Per (method, k): the S11 measurement next to the matching uniform-run quantity."""
    lines = ["| Run | k | I-AUROC | FPR @ tau, test normals | FPR @ tau, withheld group | "
             "recall @ tau, poisoned type | recall @ tau, other types |",
             "|---|---|---|---|---|---|---|"]
    for method in sorted(summary):
        for k in _k_values(summary, method):
            cells = [_mean(_values(summary, method, k, _metric("I-AUROC"))),
                     _mean(_values(summary, method, k, _metric("FPR@tau")))]
            cells += [_mean(_values(summary, method, k, _nested("stress", name)))
                      for name in ("held_out_fpr", "same_type_recall", "other_type_recall")]
            lines.append(f"| {method} | {k} | " + " | ".join(cells) + " |")
    return "\n".join(lines)

"""S8 — turning a result file into the figures and tables the report quotes.

Reads nothing but `experiment.py`'s output, so every number in the write-up comes
from the run that produced the file and cannot be edited into it by hand. Three
outputs:

  * **k-vs-accuracy curves** — the project's central intended result (SPEC §1).
  * **Per-category curves** — the average hides that the methods fail on different
    categories, which is the failure catalogue Gap G8 asks for.
  * **Markdown tables** — mean +/- std over the draws, pasted into the report.

The qualitative grid (input / ground truth / anomaly map / overlay) needs the
backbone and the images, so it lives in its own entry point below.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # non-GUI backend: render straight to file
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import tables as tables_module  # noqa: E402

HEADLINE_METRICS = ("I-AUROC", "P-AUROC", "PRO")
FIG_DPI = 150
METHOD_STYLE = {"patchcore": ("tab:blue", "o"), "padim": ("tab:orange", "s"),
                "winclip": ("tab:green", "d")}
TAG_LINESTYLE = {"": "-", "clip": "-.", "aug": ":", "visual": ":", "zeroshot": "--"}
FLOOR_STYLE = ("tab:grey", "^")


def style(method: str) -> tuple[str, str, str]:
    """(colour, marker, linestyle): colour by base method, line style by variant tag.

    Run variants are named "<method>+<tag>" by the runner (e.g. "patchcore+clip"),
    so one method keeps one colour across backbones and ablations.
    """
    base, _, tag = method.partition("+")
    colour, marker = METHOD_STYLE.get(base, ("tab:purple", "v"))
    return colour, marker, TAG_LINESTYLE.get(tag, "--")


def load_results(path: str | Path) -> dict:
    """Read a result file written by `experiment.write_results`."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def merge_results(payloads: list[dict]) -> dict:
    """Combine result files covering **disjoint categories** into one payload.

    The sweep is split across processes by category to use more than one core, and
    the split must not be able to change a number: merging is therefore refused if
    two files claim the same (method, category), rather than letting one silently
    overwrite the other.
    """
    merged: dict = {"config": payloads[0]["config"], "summary": {}, "runs": []}
    for payload in payloads:
        for method, categories in payload["summary"].items():
            target = merged["summary"].setdefault(method, {})
            clashes = set(target) & set(categories)
            if clashes:
                raise ValueError(f"{method}: {sorted(clashes)} appears in more than one file")
            target.update(categories)
        merged["runs"].extend(payload["runs"])
    return merged


def k_axis(summary: dict, method: str) -> list[int]:
    """Sorted k values present for a method, across every category."""
    values = {int(k) for category in summary[method].values() for k in category}
    return sorted(values)


def mean_over_categories(summary: dict, method: str, metric: str) -> tuple[list[int], np.ndarray, np.ndarray]:
    """(k, mean over categories, mean within-category spread) for one metric.

    Two different spreads exist here and conflating them would be dishonest: the
    error bars plotted are the *within-category spread over draws* (Gap G2, the
    reproducibility question), not the spread between categories, which is much
    larger and is shown by the per-category figure instead.
    """
    ks = k_axis(summary, method)
    means, spreads = [], []
    for k in ks:
        per_category = [
            categories[str(k)][metric]
            for categories in summary[method].values()
            if str(k) in categories
        ]
        means.append(float(np.mean([mean for mean, _ in per_category])))
        spreads.append(float(np.mean([std for _, std in per_category])))
    return ks, np.array(means), np.array(spreads)


def plot_k_curves(
    summary: dict,
    path: Path,
    metrics: tuple[str, ...] = HEADLINE_METRICS,
    floor: dict | None = None,
) -> Path:
    """The headline figure: accuracy against support-set size, one panel per metric."""
    figure, axes = plt.subplots(1, len(metrics), figsize=(4.4 * len(metrics), 3.8))
    axes = np.atleast_1d(axes)
    swept = [method for method in sorted(summary) if len(k_axis(summary, method)) > 1]
    ticks = sorted({k for method in swept for k in k_axis(summary, method)})
    for axis, metric in zip(axes, metrics):
        for method in sorted(summary):
            colour, marker, line = style(method)
            ks, means, spreads = mean_over_categories(summary, method, metric)
            if method not in swept:
                # One k only (zero-shot, or a ceiling): a reference line, not a curve.
                axis.axhline(means[0], color=colour, linestyle=line, linewidth=1.2,
                             label=f"{method} (k-independent)")
                continue
            axis.errorbar(ks, means, yerr=spreads, color=colour, marker=marker,
                          linestyle=line, capsize=3, label=method)
        if floor is not None:
            ks, means, _ = mean_over_categories(floor, "floor", metric)
            axis.plot(ks, means, color=FLOOR_STYLE[0], marker=FLOOR_STYLE[1],
                      linestyle="--", label="floor (week04)")
        axis.set_xscale("log", base=2)
        axis.set_xticks(ticks or k_axis(summary, sorted(summary)[0]))
        axis.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        axis.set_xlabel("support set size k")
        axis.set_ylabel(metric)
        axis.set_title(f"{metric} vs k")
        axis.grid(alpha=0.3)
    # One legend for the figure, below the panels, so no curve is hidden behind it.
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=min(len(labels), 4), fontsize=8,
                  bbox_to_anchor=(0.5, -0.02))
    figure.tight_layout(rect=(0, 0.06 + 0.035 * ((len(labels) - 1) // 4), 1, 1))
    figure.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(figure)
    return path


def plot_category_curves(summary: dict, path: Path, metric: str = "I-AUROC") -> Path:
    """Small multiples: one panel per category, so the failures stay visible."""
    categories = sorted({category for method in summary.values() for category in method})
    columns = 5
    rows = (len(categories) + columns - 1) // columns
    figure, axes = plt.subplots(rows, columns, figsize=(3.0 * columns, 2.6 * rows),
                                sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()
    ticks = sorted({int(k) for method in summary.values() for entries in method.values()
                    if len(entries) > 1 for k in entries})
    for axis, category in zip(axes, categories):
        for method in sorted(summary):
            if category not in summary[method]:
                continue
            colour, marker, line = style(method)
            entries = summary[method][category]
            ks = sorted(int(k) for k in entries)
            means = [entries[str(k)][metric][0] for k in ks]
            spreads = [entries[str(k)][metric][1] for k in ks]
            if len(ks) == 1:
                axis.axhline(means[0], color=colour, linestyle=line, linewidth=1.0,
                             label=method)
                continue
            axis.errorbar(ks, means, yerr=spreads, color=colour, marker=marker,
                          linestyle=line, markersize=4, capsize=2, label=method)
        axis.set_xscale("log", base=2)
        axis.set_xticks(ticks or ks)
        axis.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        axis.set_title(category, fontsize=9)
        axis.grid(alpha=0.3)
    for axis in axes[len(categories):]:
        axis.set_visible(False)
    axes[0].legend(fontsize=6)
    figure.supxlabel("support set size k")
    figure.supylabel(metric)
    figure.tight_layout()
    figure.savefig(path, dpi=FIG_DPI)
    plt.close(figure)
    return path


def markdown_table(summary: dict, metrics: tuple[str, ...] = HEADLINE_METRICS) -> str:
    """Mean +/- std over categories and draws, one row per (method, k)."""
    lines = ["| Method | k | " + " | ".join(metrics) + " | ms/image |",
             "|---|---|" + "---|" * (len(metrics) + 1)]
    for method in sorted(summary):
        for k in k_axis(summary, method):
            cells = []
            for metric in metrics:
                _, means, spreads = mean_over_categories(summary, method, metric)
                index = k_axis(summary, method).index(k)
                cells.append(f"{means[index]:.3f} +/- {spreads[index]:.3f}")
            latency = float(np.mean([
                categories[str(k)]["latency_ms"]
                for categories in summary[method].values() if str(k) in categories
            ]))
            lines.append(f"| {method} | {k} | " + " | ".join(cells) + f" | {latency:.1f} |")
    return "\n".join(lines)


def per_category_table(summary: dict, metric: str = "I-AUROC") -> str:
    """One row per category, one column per (method, k) — the failure catalogue."""
    methods = sorted(summary)
    categories = sorted({category for method in summary.values() for category in method})
    ks = k_axis(summary, methods[0])
    header = "| Category | " + " | ".join(f"{m} k={k}" for m in methods for k in ks) + " |"
    lines = [header, "|---|" + "---|" * (len(methods) * len(ks))]
    for category in categories:
        cells = []
        for method in methods:
            for k in ks:
                entry = summary[method].get(category, {}).get(str(k))
                cells.append("-" if entry is None else f"{entry[metric][0]:.3f}")
        lines.append(f"| {category} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def floor_summary(path: str | Path) -> dict | None:
    """The week04 baseline file, reshaped into the summary layout for plotting."""
    path = Path(path)
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {"floor": payload["results"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", nargs="+", default=["output/experiment_fewshot.json"])
    parser.add_argument("--floor", default="output/baseline_results.json")
    parser.add_argument("--output", default="output")
    parser.add_argument("--merged", default=None, help="where to write the merged payload")
    parser.add_argument("--ceiling", nargs="*", default=None, help="k = full result files")
    parser.add_argument("--stress", nargs="*", default=None, help="S11 result files")
    arguments = parser.parse_args()

    payload = merge_results([load_results(path) for path in arguments.results])
    summary = payload["summary"]
    output_dir = Path(arguments.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    curves = plot_k_curves(summary, output_dir / "k_vs_accuracy.png",
                           floor=floor_summary(arguments.floor))
    per_category = plot_category_curves(summary, output_dir / "k_vs_accuracy_per_category.png")
    sections = [
        ("Mean over categories", markdown_table(summary)),
        ("Image AUROC per category", per_category_table(summary)),
        ("S7 decision stage: threshold fitted on held-out normals",
         tables_module.decision_table(summary)),
        ("Cost: latency, enrolment, peak memory, reference size",
         tables_module.cost_table(summary)),
    ]
    if arguments.ceiling:
        ceiling = merge_results([load_results(path) for path in arguments.ceiling])["summary"]
        sections += [("Ceiling (k = full)", tables_module.ceiling_values(ceiling)),
                     ("Few-shot as a fraction of the ceiling",
                      tables_module.ceiling_table(summary, ceiling))]
    if arguments.stress:
        stress = merge_results([load_results(path) for path in arguments.stress])["summary"]
        sections.append(("S11 stress tests", tables_module.stress_table(stress)))
    tables = output_dir / "results_tables.md"
    tables.write_text("\n\n".join(f"## {title}\n\n{body}" for title, body in sections)
                      + "\n", encoding="utf-8")
    if arguments.merged:
        Path(arguments.merged).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {curves}\n      {per_category}\n      {tables}")


if __name__ == "__main__":
    main()

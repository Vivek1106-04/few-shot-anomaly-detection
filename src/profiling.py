"""S10 cost metrics beyond latency: peak memory and the size of the reference model.

Week03 §S10 lists peak memory next to enrol time and ms/image, because accuracy per
cost is the comparison a line engineer actually makes (Gap G5), and a method that
is 2 points better but holds a gigabyte per product is not the same proposition.

Two numbers, measured differently on purpose:

  * **Peak working memory** of a call, via `tracemalloc`. NumPy registers its
    array allocations with tracemalloc, so this sees the method's matrices — and
    only Python/NumPy memory, never the backbone's torch buffers, which is the
    right scope since the backbone is shared by every method.
  * **Reference size** — the bytes of every array the fitted model R holds, i.e.
    what must be kept per product on the line.

tracemalloc slows allocation-heavy Python loops, so the runner measures memory on a
separate call and never inside a timed one.
"""

from __future__ import annotations

import tracemalloc
from collections.abc import Callable
from dataclasses import fields, is_dataclass
from typing import TypeVar

import numpy as np

T = TypeVar("T")
MEBIBYTE = float(2**20)


def peak_memory_mb(function: Callable[[], T]) -> tuple[T, float]:
    """Run `function` and return its result with the peak traced allocation, in MiB."""
    already_tracing = tracemalloc.is_tracing()
    if not already_tracing:
        tracemalloc.start()
    tracemalloc.reset_peak()
    baseline, _ = tracemalloc.get_traced_memory()
    try:
        result = function()
        _, peak = tracemalloc.get_traced_memory()
    finally:
        if not already_tracing:
            tracemalloc.stop()
    return result, max(peak - baseline, 0) / MEBIBYTE


def reference_size_mb(reference: object) -> float:
    """Total bytes of the arrays a fitted reference model holds, in MiB."""
    if isinstance(reference, np.ndarray):
        return reference.nbytes / MEBIBYTE
    if is_dataclass(reference):
        return sum(reference_size_mb(getattr(reference, f.name)) for f in fields(reference))
    return 0.0

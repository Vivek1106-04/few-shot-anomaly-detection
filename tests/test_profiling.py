"""Cost-metric tests: peak memory sees NumPy allocations; reference size sums arrays."""

from __future__ import annotations

import tracemalloc
from dataclasses import dataclass

import numpy as np
import pytest

from profiling import peak_memory_mb, reference_size_mb


def test_peak_memory_sees_a_numpy_allocation():
    # Act: allocate and drop an 8 MiB array
    result, peak = peak_memory_mb(lambda: float(np.ones(2**20, dtype=np.float64).sum()))

    # Assert
    assert result == 2**20
    assert peak == pytest.approx(8.0, rel=0.1)


def test_peak_memory_leaves_tracing_as_it_found_it():
    # Act
    peak_memory_mb(lambda: None)

    # Assert
    assert not tracemalloc.is_tracing()


def test_peak_memory_nests_inside_an_existing_trace():
    # Arrange
    tracemalloc.start()
    try:
        # Act
        _, peak = peak_memory_mb(lambda: np.zeros(2**18).sum())

        # Assert
        assert tracemalloc.is_tracing()
        assert peak > 1.0
    finally:
        tracemalloc.stop()


@dataclass(frozen=True)
class FakeReference:
    vectors: np.ndarray
    extra: np.ndarray
    k: int


def test_reference_size_sums_every_array_field():
    # Arrange
    reference = FakeReference(np.zeros(2**17), np.zeros(2**17, dtype=np.float32), k=3)

    # Act / Assert
    assert reference_size_mb(reference) == pytest.approx(1.5)


def test_a_bare_array_is_its_own_size_and_other_objects_weigh_nothing():
    assert reference_size_mb(np.zeros(2**17)) == pytest.approx(1.0)
    assert reference_size_mb("not a model") == 0.0

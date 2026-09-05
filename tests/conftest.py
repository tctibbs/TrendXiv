"""Shared pytest fixtures."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def periods() -> list[str]:
    """A 120-month axis starting 2016-01."""
    out = []
    year, month = 2016, 1
    for _ in range(120):
        out.append(f"{year:04d}-{month:02d}")
        month += 1
        if month == 13:
            year, month = year + 1, 1
    return out


@pytest.fixture
def flat_exposure() -> np.ndarray:
    """A constant monthly denominator, for tests that isolate the numerator."""
    return np.full(120, 10_000, dtype=float)


@pytest.fixture
def seasonal_series(periods: list[str]) -> np.ndarray:
    """A trending series with a clean 12-month multiplicative cycle."""
    t = np.arange(len(periods))
    trend = 100 * np.exp(0.004 * t)
    season = 1 + 0.30 * np.sin(2 * np.pi * (t % 12) / 12)
    return trend * season

"""Tests for the analysis orchestration layer."""

from __future__ import annotations

import numpy as np
import pytest

from src.pipeline.analyze import (
    burst_artifact,
    group_month_matrix,
    lifecycle_artifact,
    rising_artifact,
    seasonal_artifact,
    specific_terms,
)


@pytest.fixture
def axis() -> list[str]:
    out = []
    year, month = 2010, 1
    for _ in range(180):
        out.append(f"{year:04d}-{month:02d}")
        month += 1
        if month == 13:
            year, month = year + 1, 1
    return out


@pytest.fixture
def totals(axis) -> list[int]:
    return [10_000] * len(axis)


@pytest.mark.unit
def test_seasonal_artifact_reports_the_peak_month(axis):
    """A category with a deliberate May spike must be fingerprinted at month 5."""
    t = np.arange(len(axis))
    values = 1000 + 300 * (np.array([int(p[5:]) for p in axis]) == 5) + 0.5 * t
    cube = {"cs.CL": {"any": values.tolist(), "primary": values.tolist(),
                      "frac": values.tolist()}}

    out = seasonal_artifact(cube, axis, len(axis))

    assert out["cs.CL"]["peak_month"] == 5
    assert out["cs.CL"]["amplitude"] > 1.0


@pytest.mark.unit
def test_seasonal_artifact_skips_short_series(axis):
    cube = {"new.CAT": {"any": [0] * (len(axis) - 10) + [5] * 10,
                        "primary": [0] * len(axis), "frac": [0.0] * len(axis)}}

    assert seasonal_artifact(cube, axis, len(axis)) == {}


@pytest.mark.unit
def test_burst_artifact_finds_a_planted_burst(axis, totals):
    counts = np.full(len(axis), 100.0)
    counts[100:130] = 600.0
    vectors = {"planted": counts}

    out = burst_artifact(vectors, totals, axis, len(axis))

    assert out["rows"], "a 6x step over 30 months must register as a burst"
    row = out["rows"][0]
    assert row["term"] == "planted"
    assert axis[95] <= row["start"] <= axis[105]


@pytest.mark.unit
def test_burst_artifact_attaches_the_dominant_field(axis, totals):
    counts = np.full(len(axis), 100.0)
    counts[100:130] = 600.0

    out = burst_artifact(
        {"planted": counts}, totals, axis, len(axis),
        profiles={"planted": {"cs": 900.0, "math": 100.0}},
    )

    assert out["rows"][0]["field"] == "cs"
    assert out["rows"][0]["field_share"] == pytest.approx(0.9)


@pytest.mark.unit
def test_burst_weight_is_normalised_by_exposure(axis, totals):
    """Raw Kleinberg weight scales with volume, which would rank by popularity.

    Two terms with an identical relative burst but 10x different volume must
    receive comparable weights.
    """
    small = np.full(len(axis), 50.0)
    small[100:130] = 300.0
    big = small * 10

    out = burst_artifact({"small": small, "big": big}, totals, axis, len(axis))
    weights = {row["term"]: row["weight"] for row in out["rows"]}

    assert set(weights) == {"small", "big"}
    assert weights["big"] == pytest.approx(weights["small"], rel=0.35)


@pytest.mark.unit
def test_rising_artifact_reports_the_screening_count(axis, totals):
    flat = np.full(len(axis), 50.0)
    grew = np.full(len(axis), 10.0)
    grew[-12:] = 400.0

    out = rising_artifact({"flat": flat, "grew": grew}, totals, len(axis))

    assert out["n_screened"] == 2
    assert [row["term"] for row in out["rows"]] == ["grew"]


@pytest.mark.unit
def test_lifecycle_artifact_classifies_terms(axis, totals):
    ramp = np.linspace(1, 400, len(axis))

    out = lifecycle_artifact({"ramp": ramp}, totals, len(axis))

    assert out["rows"][0]["term"] == "ramp"
    assert out["rows"][0]["trend"] == "increasing"


@pytest.mark.unit
def test_group_month_matrix_sums_categories_into_groups(axis):
    cube = {
        "cs.LG": {"primary": [1] * len(axis), "any": [], "frac": []},
        "cs.CV": {"primary": [2] * len(axis), "any": [], "frac": []},
        "math.AG": {"primary": [4] * len(axis), "any": [], "frac": []},
    }
    groups = {"cs.LG": "cs", "cs.CV": "cs", "math.AG": "math"}

    out = group_month_matrix(cube, groups, axis)

    assert out["cs"][0] == 3
    assert out["math"][0] == 4


@pytest.mark.unit
def test_specific_terms_drops_prose_and_keeps_topics(axis):
    """Prose tracks the corpus's own field mix; a topic does not."""
    group_month = {
        "cs": np.full(len(axis), 700.0),
        "math": np.full(len(axis), 200.0),
        "physics": np.full(len(axis), 100.0),
    }
    usage = np.full(len(axis), 100.0)
    profiles = {
        "prose": {"cs": 700.0, "math": 200.0, "physics": 100.0},
        "topic": {"physics": 1000.0},
    }

    kept = specific_terms({"prose": usage, "topic": usage}, profiles, group_month)

    assert set(kept) == {"topic"}

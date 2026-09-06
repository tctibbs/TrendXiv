"""Tests for the build-time integrity gates."""

import pytest

from src.pipeline.validate import MAX_GROWTH, validate

PERIODS = ["2020-01", "2020-02", "2020-03"]
TOTALS = {"papers": [100, 110, 120], "listings": [180, 190, 200]}
CUBE = {"cs.LG": {"any": [10, 11, 12], "primary": [8, 9, 10], "frac": [9.0, 10.0, 11.0]}}
OFFICIAL = {"2020-01": 100, "2020-02": 110, "2020-03": 120}


def _run(corpus_rows, expected_rows=330):
    return validate(PERIODS, TOTALS, CUBE, corpus_rows, expected_rows, official=OFFICIAL)


def _check(report, name):
    return next(ok for check, ok, _ in report.checks if check == name)


@pytest.mark.unit
def test_a_corpus_that_has_grown_still_passes():
    """The scheduled rebuild exists to pick up new papers.

    Demanding an exact row count would abort the build the first week the
    snapshot grows, freezing the site on its last good deploy. That is a
    self-disabling pipeline, not a safety gate.
    """
    report = _run(340)

    assert _check(report, "row_count")
    assert report.passed


@pytest.mark.unit
def test_an_unchanged_corpus_passes():
    assert _check(_run(330), "row_count")


@pytest.mark.unit
def test_a_shrinking_corpus_fails():
    """Papers are never withdrawn in bulk, so fewer rows means rows went missing."""
    report = _run(300)

    assert not _check(report, "row_count")
    assert not report.passed


@pytest.mark.unit
def test_implausible_growth_fails():
    """A corpus that suddenly jumps has changed shape, not merely grown."""
    report = _run(int(330 * MAX_GROWTH) + 10)

    assert not _check(report, "row_count")


@pytest.mark.unit
def test_official_totals_are_only_a_check_never_a_denominator():
    """Shares divide by this snapshot; the published series only certifies it."""
    report = _run(330)

    assert _check(report, "official_aggregate")
    assert any(check.startswith("official_decade") for check, _, _ in report.checks)


@pytest.mark.unit
def test_a_mismatched_axis_fails():
    broken = {"cs.LG": {"any": [1, 2], "primary": [1, 2], "frac": [1.0, 2.0]}}
    report = validate(PERIODS, TOTALS, broken, 330, 330, official=OFFICIAL)

    assert not _check(report, "axis_alignment")


@pytest.mark.unit
def test_report_renders_every_check():
    rendered = _run(330).render()

    assert "row_count" in rendered
    assert rendered.startswith(("PASS", "FAIL"))

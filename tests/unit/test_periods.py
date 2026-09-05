"""Tests for the shared monthly period axis."""

from datetime import date

import pytest

from src.pipeline.periods import completeness, month_range, previous_month


@pytest.mark.unit
def test_month_range_is_inclusive_at_both_ends():
    assert month_range("2020-11", "2021-02") == ["2020-11", "2020-12", "2021-01", "2021-02"]


@pytest.mark.unit
def test_month_range_single_month():
    assert month_range("1991-07", "1991-07") == ["1991-07"]


@pytest.mark.unit
def test_month_range_spans_decades_without_drift():
    axis = month_range("1991-07", "2026-08")
    assert axis[0] == "1991-07"
    assert axis[-1] == "2026-08"
    # 35 years and 2 months inclusive.
    assert len(axis) == (2026 - 1991) * 12 + (8 - 7) + 1


@pytest.mark.unit
def test_previous_month_wraps_the_year():
    assert previous_month("2021-01") == "2020-12"
    assert previous_month("2021-07") == "2021-06"


@pytest.mark.unit
def test_completeness_marks_the_snapshot_month_provisional():
    """A snapshot taken mid-August makes August partial and July the last full month."""
    complete_through, provisional_from = completeness(date(2026, 8, 27))
    assert provisional_from == "2026-08"
    assert complete_through == "2026-07"


@pytest.mark.unit
def test_completeness_across_a_year_boundary():
    assert completeness(date(2021, 1, 3)) == ("2020-12", "2021-01")

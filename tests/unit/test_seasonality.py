"""Tests for seasonal decomposition of monthly arXiv series."""

import calendar

import numpy as np
import pytest

from src.analysis.seasonality import (
    month_of_year_index,
    stl_decompose,
    trading_day_adjust,
)

SEASONAL_AMPLITUDE = 0.30
MONTHLY_LOG_GROWTH = 0.01


def _month_labels(start_year: int, n_months: int, start_month: int = 1) -> list[str]:
    """Build consecutive YYYY-MM labels starting at start_month of start_year."""
    first = start_year * 12 + start_month - 1
    return [f"{(first + index) // 12}-{(first + index) % 12 + 1:02d}" for index in range(n_months)]


def _log_linear_with_sinusoid(n_months: int = 120) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a series whose log is exactly a line plus a 12-month sinusoid.

    Returns:
        Tuple of the observed series, its trend on the original scale, and the
        log-scale seasonal component that a decomposition should recover.
    """
    time = np.arange(n_months, dtype=float)
    log_trend = np.log(1000.0) + MONTHLY_LOG_GROWTH * time
    seasonal = SEASONAL_AMPLITUDE * np.sin(2.0 * np.pi * time / 12.0)
    return np.exp(log_trend + seasonal), np.exp(log_trend), seasonal


@pytest.mark.unit
class TestStlDecompose:
    """Tests for stl_decompose."""

    def test_recovers_known_sinusoid_and_flattens_series(self):
        values, trend, seasonal = _log_linear_with_sinusoid()

        result = stl_decompose(values)

        assert result.used_log is True
        assert np.allclose(result.seasonal, seasonal, atol=0.03)
        assert np.allclose(result.adjusted, trend, rtol=0.03)

    def test_adjusted_series_is_monotonically_increasing(self):
        values, _, _ = _log_linear_with_sinusoid()

        adjusted = stl_decompose(values).adjusted

        # The input rises and falls with the sinusoid; adjusting must leave only
        # the underlying growth.
        assert np.any(np.diff(values) < 0)
        assert np.all(np.diff(adjusted) > 0)

    def test_strength_is_high_for_strongly_seasonal_series(self):
        values, _, _ = _log_linear_with_sinusoid()

        assert stl_decompose(values).strength > 0.95

    def test_strength_is_low_for_nonseasonal_series(self):
        rng = np.random.default_rng(20240501)
        values = 1000.0 + rng.normal(0.0, 25.0, size=120)

        assert stl_decompose(values).strength < 0.4

    def test_components_reconstruct_the_log_series(self):
        values, _, _ = _log_linear_with_sinusoid()

        result = stl_decompose(values)

        reconstructed = result.trend + result.seasonal + result.resid
        assert np.allclose(reconstructed, np.log(values), atol=1e-6)

    def test_falls_back_to_additive_when_series_contains_zero(self):
        values, _, _ = _log_linear_with_sinusoid(n_months=48)
        values[10] = 0.0

        result = stl_decompose(values)

        assert result.used_log is False
        # Additive adjustment stays on the original scale, so the level is kept.
        assert float(np.mean(result.adjusted)) == pytest.approx(float(np.mean(values)), rel=0.01)

    def test_log_can_be_disabled(self):
        values, _, _ = _log_linear_with_sinusoid(n_months=48)

        assert stl_decompose(values, log=False).used_log is False

    def test_rejects_series_shorter_than_two_periods(self):
        values = np.linspace(100.0, 200.0, 18)

        with pytest.raises(ValueError, match="two full periods"):
            stl_decompose(values, period=12)

    def test_accepts_series_of_exactly_two_periods(self):
        values, _, _ = _log_linear_with_sinusoid(n_months=24)

        assert stl_decompose(values).period == 12

    def test_rejects_degenerate_period(self):
        values = np.linspace(100.0, 200.0, 48)

        with pytest.raises(ValueError, match="period must be at least 2"):
            stl_decompose(values, period=1)

    @pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
    def test_rejects_non_finite_values(self, bad):
        # STL yields all-NaN components here, whose strength computes to 0.0 --
        # a "not seasonal" verdict that would look like a real answer.
        values = np.full(48, 1000.0)
        values[7] = bad

        with pytest.raises(ValueError, match="finite"):
            stl_decompose(values)

    def test_strength_is_zero_for_constant_series(self):
        # STL returns components of ~1e-16 here; those must not read as seasonal.
        values = np.full(48, 500.0)

        assert stl_decompose(values).strength == 0.0


@pytest.mark.unit
class TestMonthOfYearIndex:
    """Tests for month_of_year_index."""

    @staticmethod
    def _may_spike_series(
        n_years: int = 8,
        spike: float = 1.5,
        start_month: int = 1,
    ) -> tuple[np.ndarray, list[str]]:
        """Build a flat series with a known multiplicative spike every May."""
        periods = _month_labels(2016, n_years * 12, start_month)
        values = np.full(len(periods), 1000.0)
        for index, period in enumerate(periods):
            if period.endswith("-05"):
                values[index] *= spike
        return values, periods

    def test_peaks_at_may_for_a_may_spike(self):
        values, periods = self._may_spike_series()

        index = month_of_year_index(values, periods)

        assert max(index, key=index.get) == 5
        # One month in twelve carries the whole 1.5x, and the index is
        # normalised to geometric mean 1, so May must land on 1.5**(11/12).
        assert index[5] == pytest.approx(1.5 ** (11 / 12), rel=0.02)
        assert index[1] == pytest.approx(1.5 ** (-1 / 12), rel=0.02)

    def test_maps_months_by_label_not_by_position(self):
        # The same May spike on a series that starts in July. Code that assumed
        # index 0 is January would report the peak in November instead.
        values, periods = self._may_spike_series(start_month=7)

        assert periods[0] == "2016-07"

        index = month_of_year_index(values, periods)

        assert max(index, key=index.get) == 5
        assert index[5] == pytest.approx(1.5 ** (11 / 12), rel=0.02)

    def test_geometric_mean_is_one(self):
        values, periods = self._may_spike_series()

        index = month_of_year_index(values, periods)

        geometric_mean = float(np.exp(np.mean(np.log(list(index.values())))))
        assert round(geometric_mean, 6) == 1.0

    def test_covers_every_calendar_month(self):
        values, periods = self._may_spike_series()

        assert sorted(month_of_year_index(values, periods)) == list(range(1, 13))

    def test_growth_does_not_leak_into_the_fingerprint(self):
        # A purely exponential series has no seasonality; a group-by-month mean
        # would report December above January simply because it comes later.
        periods = _month_labels(2016, 96)
        values = 1000.0 * np.exp(MONTHLY_LOG_GROWTH * np.arange(len(periods)))

        index = month_of_year_index(values, periods)

        assert max(index.values()) - min(index.values()) < 0.05

    def test_works_on_additive_series_containing_zero(self):
        values, periods = self._may_spike_series()
        values[3] = 0.0

        index = month_of_year_index(values, periods)

        assert max(index, key=index.get) == 5
        assert round(float(np.exp(np.mean(np.log(list(index.values()))))), 6) == 1.0

    def test_rejects_misaligned_periods(self):
        values, periods = self._may_spike_series()

        with pytest.raises(ValueError, match="must align"):
            month_of_year_index(values, periods[:-1])

    def test_rejects_a_gap_in_the_month_axis(self):
        # STL reads position as time: dropping 2018-06 shifts every later month
        # by one, which smears the May spike into April instead of failing.
        values, periods = self._may_spike_series()
        gap = periods.index("2018-06")
        periods = periods[:gap] + periods[gap + 1 :]
        values = np.concatenate([values[:gap], values[gap + 1 :]])

        with pytest.raises(ValueError, match="consecutive months"):
            month_of_year_index(values, periods)

    def test_rejects_duplicated_periods(self):
        values, periods = self._may_spike_series()
        periods[10] = periods[9]

        with pytest.raises(ValueError, match="consecutive months"):
            month_of_year_index(values, periods)

    def test_rejects_series_missing_a_calendar_month(self):
        periods = [f"2016-{month:02d}" for month in (1, 2, 3, 4)] * 8
        values = np.full(len(periods), 1000.0)

        with pytest.raises(ValueError, match="consecutive months"):
            month_of_year_index(values, periods)

    def test_rejects_malformed_period(self):
        values, periods = self._may_spike_series()
        periods[0] = "2016/01"

        with pytest.raises(ValueError, match="YYYY-MM"):
            month_of_year_index(values, periods)

    def test_rejects_nonnumeric_period(self):
        values, periods = self._may_spike_series()
        periods[0] = "20xx-01"

        with pytest.raises(ValueError, match="YYYY-MM"):
            month_of_year_index(values, periods)

    def test_rejects_all_zero_series(self):
        periods = _month_labels(2016, 48)
        values = np.zeros(len(periods))

        with pytest.raises(ValueError, match="non-positive series"):
            month_of_year_index(values, periods)

    def test_rejects_out_of_range_month(self):
        values, periods = self._may_spike_series()
        periods[0] = "2016-13"

        with pytest.raises(ValueError, match="month must be in 1..12"):
            month_of_year_index(values, periods)


@pytest.mark.unit
class TestTradingDayAdjust:
    """Tests for trading_day_adjust."""

    def test_leaves_constant_series_constant(self):
        # Non-leap Februaries are exactly four weeks, so every month here holds
        # the same 20 weekdays and no rescaling should occur.
        periods = [f"{year}-02" for year in (2018, 2019, 2021, 2022, 2023, 2025)]
        values = np.full(len(periods), 750.0)

        assert np.allclose(trading_day_adjust(values, periods), values, atol=1e-12)

    def test_flattens_a_series_proportional_to_weekday_count(self):
        periods = _month_labels(2021, 24)
        weekdays = np.array(
            [_weekday_count(int(p[:4]), int(p[5:])) for p in periods], dtype=float
        )
        values = 50.0 * weekdays

        adjusted = trading_day_adjust(values, periods)

        assert np.allclose(adjusted, adjusted[0], atol=1e-9)

    def test_preserves_the_overall_level(self):
        periods = _month_labels(2021, 24)
        values = np.full(len(periods), 1000.0)

        adjusted = trading_day_adjust(values, periods)

        # Rescaling is relative to the mean weekday count, so months move but
        # the series level barely does.
        assert abs(float(np.mean(adjusted)) - 1000.0) < 5.0

    def test_shortens_months_with_extra_weekdays(self):
        # March 2021 has 23 weekdays, February 2021 only 20.
        periods = ["2021-02", "2021-03"]
        values = np.array([1000.0, 1000.0])

        adjusted = trading_day_adjust(values, periods)

        assert adjusted[0] > adjusted[1]

    def test_handles_empty_input(self):
        assert trading_day_adjust(np.array([]), []).size == 0

    def test_rejects_misaligned_periods(self):
        with pytest.raises(ValueError, match="must align"):
            trading_day_adjust(np.array([1.0, 2.0]), ["2021-01"])


def _weekday_count(year: int, month: int) -> int:
    """Count weekdays independently of the module under test."""
    return sum(
        1
        for day in range(1, calendar.monthrange(year, month)[1] + 1)
        if calendar.weekday(year, month, day) < 5
    )

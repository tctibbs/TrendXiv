"""Unit tests for TimeSeriesAggregator."""

from datetime import date

import pandas as pd

from src.data.aggregator import TimeSeriesAggregator


class TestTimeSeriesAggregator:
    """Tests for TimeSeriesAggregator class."""

    def test_generate_period_index_monthly(self):
        """Test monthly period index generation."""
        aggregator = TimeSeriesAggregator()
        index = aggregator._generate_period_index(
            start_date=date(2024, 1, 1),
            end_date=date(2024, 6, 30),
            period="month",
        )

        assert len(index) == 6
        assert index[0] == pd.Timestamp("2024-01-01")
        assert index[-1] == pd.Timestamp("2024-06-01")

    def test_generate_period_index_weekly(self):
        """Test weekly period index generation."""
        aggregator = TimeSeriesAggregator()
        index = aggregator._generate_period_index(
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 31),
            period="week",
        )

        assert len(index) >= 4

    def test_generate_period_index_daily(self):
        """Test daily period index generation."""
        aggregator = TimeSeriesAggregator()
        index = aggregator._generate_period_index(
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 10),
            period="day",
        )

        assert len(index) == 10

    def test_empty_time_series(self):
        """Test empty time series creation."""
        aggregator = TimeSeriesAggregator()
        df = aggregator._empty_time_series(
            categories=["cs.LG", "cs.AI"],
            start_date=date(2024, 1, 1),
            end_date=date(2024, 3, 31),
            period="month",
        )

        assert "date" in df.columns
        assert "cs.LG" in df.columns
        assert "cs.AI" in df.columns
        assert df["cs.LG"].sum() == 0
        assert df["cs.AI"].sum() == 0

    def test_empty_keyword_series(self):
        """Test empty keyword series creation."""
        aggregator = TimeSeriesAggregator()
        df = aggregator._empty_keyword_series(
            column_name="transformer",
            start_date=date(2024, 1, 1),
            end_date=date(2024, 3, 31),
            period="month",
        )

        assert "date" in df.columns
        assert "transformer" in df.columns
        assert df["transformer"].sum() == 0

    def test_get_period_monthly(self):
        """Test monthly period extraction."""
        aggregator = TimeSeriesAggregator()
        dates = pd.Series(pd.to_datetime(["2024-01-15", "2024-01-20", "2024-02-10"]))

        periods = aggregator._get_period(dates, "month")

        assert periods.iloc[0] == pd.Timestamp("2024-01-01")
        assert periods.iloc[1] == pd.Timestamp("2024-01-01")
        assert periods.iloc[2] == pd.Timestamp("2024-02-01")

    def test_get_period_daily(self):
        """Test daily period extraction."""
        aggregator = TimeSeriesAggregator()
        dates = pd.Series(pd.to_datetime(["2024-01-15 10:30:00", "2024-01-15 14:45:00"]))

        periods = aggregator._get_period(dates, "day")

        assert periods.iloc[0] == pd.Timestamp("2024-01-15")
        assert periods.iloc[1] == pd.Timestamp("2024-01-15")

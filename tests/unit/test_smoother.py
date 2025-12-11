"""Unit tests for MovingAverageCalculator."""

import pandas as pd
import pytest

from src.data.smoother import MovingAverageCalculator


class TestMovingAverageCalculator:
    """Tests for MovingAverageCalculator class."""

    @pytest.fixture
    def smoother(self):
        """Create smoother instance."""
        return MovingAverageCalculator()

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        return pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=6, freq="MS"),
                "cs.LG": [10, 20, 30, 40, 50, 60],
            }
        )

    def test_apply_with_zero_window(self, smoother, sample_data):
        """Test that zero window returns unchanged data."""
        result = smoother.apply(sample_data, window_months=0)

        pd.testing.assert_frame_equal(result, sample_data)

    def test_apply_3_month_window(self, smoother, sample_data):
        """Test 3-month moving average."""
        result = smoother.apply(sample_data, window_months=3)

        assert pd.isna(result["cs.LG"].iloc[0])
        assert pd.isna(result["cs.LG"].iloc[1])
        assert result["cs.LG"].iloc[2] == pytest.approx(20.0)
        assert result["cs.LG"].iloc[3] == pytest.approx(30.0)
        assert result["cs.LG"].iloc[4] == pytest.approx(40.0)
        assert result["cs.LG"].iloc[5] == pytest.approx(50.0)

    def test_apply_preserves_date_column(self, smoother, sample_data):
        """Test that date column is preserved."""
        result = smoother.apply(sample_data, window_months=3)

        pd.testing.assert_series_equal(result["date"], sample_data["date"])

    def test_apply_with_min_periods(self, smoother, sample_data):
        """Test moving average with custom min_periods."""
        result = smoother.apply(sample_data, window_months=3, min_periods=1)

        assert not pd.isna(result["cs.LG"].iloc[0])
        assert result["cs.LG"].iloc[0] == 10.0
        assert result["cs.LG"].iloc[1] == 15.0

    def test_apply_centered(self, smoother, sample_data):
        """Test centered moving average."""
        result = smoother.apply_centered(sample_data, window_months=3)

        assert not pd.isna(result["cs.LG"].iloc[0])
        assert result["cs.LG"].iloc[1] == pytest.approx(20.0)

    def test_apply_centered_zero_window(self, smoother, sample_data):
        """Test centered moving average with zero window."""
        result = smoother.apply_centered(sample_data, window_months=0)

        pd.testing.assert_frame_equal(result, sample_data)

    def test_apply_exponential(self, smoother, sample_data):
        """Test exponential moving average."""
        result = smoother.apply_exponential(sample_data, span_months=3)

        assert not pd.isna(result["cs.LG"].iloc[0])
        assert result["cs.LG"].iloc[0] == 10.0

        for i in range(1, len(result)):
            assert result["cs.LG"].iloc[i] != sample_data["cs.LG"].iloc[i]

    def test_apply_exponential_zero_span(self, smoother, sample_data):
        """Test exponential moving average with zero span."""
        result = smoother.apply_exponential(sample_data, span_months=0)

        pd.testing.assert_frame_equal(result, sample_data)

    def test_remove_seasonality(self, smoother):
        """Test seasonal differencing."""
        data = pd.DataFrame(
            {
                "date": pd.date_range("2023-01-01", periods=24, freq="MS"),
                "value": list(range(1, 13)) * 2,
            }
        )

        result = smoother.remove_seasonality(data, period=12)

        assert pd.isna(result["value"].iloc[:12]).all()
        assert (result["value"].iloc[12:] == 0).all()

    def test_multiple_columns(self, smoother):
        """Test smoothing with multiple value columns."""
        data = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=6, freq="MS"),
                "cs.LG": [10, 20, 30, 40, 50, 60],
                "cs.AI": [5, 10, 15, 20, 25, 30],
            }
        )

        result = smoother.apply(data, window_months=3)

        assert result["cs.LG"].iloc[3] == pytest.approx(30.0)
        assert result["cs.AI"].iloc[3] == pytest.approx(15.0)

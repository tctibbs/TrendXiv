"""Unit tests for DataNormalizer."""

import pandas as pd
import pytest

from src.data.normalizer import DataNormalizer


class TestDataNormalizer:
    """Tests for DataNormalizer class."""

    @pytest.fixture
    def normalizer(self):
        """Create normalizer instance."""
        return DataNormalizer()

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        return pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=3, freq="MS"),
                "cs.LG": [100, 150, 200],
                "cs.AI": [50, 75, 100],
            }
        )

    def test_to_absolute_returns_copy(self, normalizer, sample_data):
        """Test that to_absolute returns a copy without modification."""
        result = normalizer.to_absolute(sample_data)

        assert result is not sample_data
        pd.testing.assert_frame_equal(result, sample_data)

    def test_to_relative_row_total(self, normalizer, sample_data):
        """Test relative normalization using row totals."""
        result = normalizer.to_relative(sample_data)

        assert result["cs.LG"].iloc[0] == pytest.approx(66.67, rel=0.01)
        assert result["cs.AI"].iloc[0] == pytest.approx(33.33, rel=0.01)

        assert (result["cs.LG"] + result["cs.AI"]).iloc[0] == pytest.approx(100, rel=0.01)

    def test_to_relative_with_baseline(self, normalizer):
        """Test relative normalization with baseline column."""
        data = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=2, freq="MS"),
                "cs.LG": [100, 150],
                "total": [1000, 1200],
            }
        )

        result = normalizer.to_relative(data, baseline_column="total")

        assert result["cs.LG"].iloc[0] == pytest.approx(10.0)
        assert result["cs.LG"].iloc[1] == pytest.approx(12.5)
        assert "total" not in result.columns

    def test_to_relative_handles_zero_totals(self, normalizer):
        """Test that zero totals don't cause division errors."""
        data = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=2, freq="MS"),
                "cs.LG": [0, 100],
                "cs.AI": [0, 50],
            }
        )

        result = normalizer.to_relative(data)

        assert not result["cs.LG"].isna().any()
        assert not result["cs.AI"].isna().any()

    def test_normalize_absolute_mode(self, normalizer, sample_data):
        """Test normalize function in absolute mode."""
        result = normalizer.normalize(sample_data, mode="absolute")

        pd.testing.assert_frame_equal(result, sample_data)

    def test_normalize_relative_mode(self, normalizer, sample_data):
        """Test normalize function in relative mode."""
        result = normalizer.normalize(sample_data, mode="relative")

        total_per_row = result["cs.LG"] + result["cs.AI"]
        for val in total_per_row:
            assert val == pytest.approx(100, rel=0.01)

    def test_calculate_growth_rate(self, normalizer, sample_data):
        """Test growth rate calculation."""
        result = normalizer.calculate_growth_rate(sample_data)

        assert pd.isna(result["cs.LG"].iloc[0])
        assert result["cs.LG"].iloc[1] == pytest.approx(50.0)
        assert result["cs.LG"].iloc[2] == pytest.approx(33.33, rel=0.01)

    def test_calculate_cumulative(self, normalizer, sample_data):
        """Test cumulative sum calculation."""
        result = normalizer.calculate_cumulative(sample_data)

        assert result["cs.LG"].iloc[0] == 100
        assert result["cs.LG"].iloc[1] == 250
        assert result["cs.LG"].iloc[2] == 450

    def test_index_to_base_period(self, normalizer, sample_data):
        """Test indexing to base period."""
        result = normalizer.index_to_base_period(sample_data)

        assert result["cs.LG"].iloc[0] == 100.0
        assert result["cs.LG"].iloc[1] == 150.0
        assert result["cs.LG"].iloc[2] == 200.0

    def test_index_to_base_period_handles_zero(self, normalizer):
        """Test that zero base value is handled."""
        data = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=2, freq="MS"),
                "cs.LG": [0, 100],
            }
        )

        result = normalizer.index_to_base_period(data)

        assert result["cs.LG"].iloc[0] == 0
        assert result["cs.LG"].iloc[1] == 0

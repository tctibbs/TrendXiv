"""Data normalization for absolute/relative calculations."""

from typing import Literal

import pandas as pd


class DataNormalizer:
    """Normalizes time series data for comparison."""

    def normalize(
        self,
        data: pd.DataFrame,
        mode: Literal["absolute", "relative"],
        date_column: str = "date",
        baseline_column: str | None = None,
    ) -> pd.DataFrame:
        """Normalize data based on the specified mode.

        Args:
            data: DataFrame with time series data
            mode: 'absolute' for raw counts, 'relative' for percentages
            date_column: Name of the date column
            baseline_column: Column to use as baseline for relative mode

        Returns:
            Normalized DataFrame
        """
        if mode == "absolute":
            return self.to_absolute(data)
        else:
            return self.to_relative(data, date_column, baseline_column)

    def to_absolute(self, data: pd.DataFrame) -> pd.DataFrame:
        """Return data with absolute counts (no transformation).

        Args:
            data: DataFrame with time series data

        Returns:
            Original DataFrame unchanged
        """
        return data.copy()

    def to_relative(
        self,
        data: pd.DataFrame,
        date_column: str = "date",
        baseline_column: str | None = None,
    ) -> pd.DataFrame:
        """Convert counts to relative percentages.

        If baseline_column is provided, calculate percentage relative to that column.
        Otherwise, calculate percentage of row total.

        Args:
            data: DataFrame with time series data
            date_column: Name of the date column to exclude from calculation
            baseline_column: Optional column to use as denominator

        Returns:
            DataFrame with percentage values
        """
        result = data.copy()
        value_columns = [col for col in result.columns if col != date_column]

        if baseline_column and baseline_column in result.columns:
            baseline = result[baseline_column].replace(0, 1)
            for col in value_columns:
                if col != baseline_column:
                    result[col] = (result[col] / baseline) * 100
            result = result.drop(columns=[baseline_column])
        else:
            row_totals = result[value_columns].sum(axis=1).replace(0, 1)
            for col in value_columns:
                result[col] = (result[col] / row_totals) * 100

        return result

    def calculate_growth_rate(
        self,
        data: pd.DataFrame,
        date_column: str = "date",
        periods: int = 1,
    ) -> pd.DataFrame:
        """Calculate period-over-period growth rate.

        Args:
            data: DataFrame with time series data
            date_column: Name of the date column
            periods: Number of periods for growth calculation

        Returns:
            DataFrame with growth rates (percentage change)
        """
        result = data.copy()
        value_columns = [col for col in result.columns if col != date_column]

        for col in value_columns:
            result[col] = result[col].pct_change(periods=periods) * 100

        return result

    def calculate_cumulative(
        self,
        data: pd.DataFrame,
        date_column: str = "date",
    ) -> pd.DataFrame:
        """Calculate cumulative sums over time.

        Args:
            data: DataFrame with time series data
            date_column: Name of the date column

        Returns:
            DataFrame with cumulative values
        """
        result = data.copy()
        value_columns = [col for col in result.columns if col != date_column]

        for col in value_columns:
            result[col] = result[col].cumsum()

        return result

    def index_to_base_period(
        self,
        data: pd.DataFrame,
        date_column: str = "date",
        base_index: int = 0,
    ) -> pd.DataFrame:
        """Index values relative to a base period (base = 100).

        Args:
            data: DataFrame with time series data
            date_column: Name of the date column
            base_index: Row index to use as base period

        Returns:
            DataFrame with indexed values
        """
        result = data.copy()
        value_columns = [col for col in result.columns if col != date_column]

        for col in value_columns:
            base_value = result[col].iloc[base_index]
            if base_value != 0:
                result[col] = (result[col] / base_value) * 100
            else:
                result[col] = 0

        return result

"""Moving average calculations for time series smoothing."""

import pandas as pd


class MovingAverageCalculator:
    """Applies moving averages to time series data."""

    def apply(
        self,
        data: pd.DataFrame,
        window_months: int,
        date_column: str = "date",
        min_periods: int | None = None,
    ) -> pd.DataFrame:
        """Apply a moving average to all numeric columns.

        Args:
            data: DataFrame with time series data
            window_months: Size of the moving average window in months
            date_column: Name of the date column to exclude
            min_periods: Minimum observations required. Defaults to window_months.

        Returns:
            DataFrame with smoothed values
        """
        if window_months <= 0:
            return data.copy()

        result = data.copy()
        value_columns = [col for col in result.columns if col != date_column]

        min_obs = min_periods if min_periods is not None else window_months

        for col in value_columns:
            result[col] = (
                result[col]
                .rolling(window=window_months, min_periods=min_obs)
                .mean()
            )

        return result

    def apply_centered(
        self,
        data: pd.DataFrame,
        window_months: int,
        date_column: str = "date",
    ) -> pd.DataFrame:
        """Apply a centered moving average.

        A centered moving average looks at values before AND after each point,
        providing a smoother trend line.

        Args:
            data: DataFrame with time series data
            window_months: Size of the moving average window in months
            date_column: Name of the date column to exclude

        Returns:
            DataFrame with smoothed values
        """
        if window_months <= 0:
            return data.copy()

        result = data.copy()
        value_columns = [col for col in result.columns if col != date_column]

        for col in value_columns:
            result[col] = (
                result[col]
                .rolling(window=window_months, center=True, min_periods=1)
                .mean()
            )

        return result

    def apply_exponential(
        self,
        data: pd.DataFrame,
        span_months: int,
        date_column: str = "date",
    ) -> pd.DataFrame:
        """Apply exponential moving average (EMA).

        EMA gives more weight to recent observations, making it more
        responsive to recent changes while still smoothing noise.

        Args:
            data: DataFrame with time series data
            span_months: Span for EMA calculation
            date_column: Name of the date column to exclude

        Returns:
            DataFrame with smoothed values
        """
        if span_months <= 0:
            return data.copy()

        result = data.copy()
        value_columns = [col for col in result.columns if col != date_column]

        for col in value_columns:
            result[col] = result[col].ewm(span=span_months, adjust=False).mean()

        return result

    def remove_seasonality(
        self,
        data: pd.DataFrame,
        period: int = 12,
        date_column: str = "date",
    ) -> pd.DataFrame:
        """Remove seasonal component using differencing.

        Subtracts the value from the same period in the previous cycle
        to remove seasonal patterns.

        Args:
            data: DataFrame with time series data
            period: Seasonal period (12 for monthly data with yearly seasonality)
            date_column: Name of the date column to exclude

        Returns:
            DataFrame with seasonally adjusted values
        """
        result = data.copy()
        value_columns = [col for col in result.columns if col != date_column]

        for col in value_columns:
            result[col] = result[col].diff(periods=period)

        return result

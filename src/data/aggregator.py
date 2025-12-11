"""Time-series aggregation for paper counts."""

from datetime import date
from typing import Literal

import pandas as pd

from src.repository.papers import PaperRepository


class TimeSeriesAggregator:
    """Aggregates paper data into time series for visualization."""

    def __init__(self, repository: PaperRepository | None = None) -> None:
        """Initialize the aggregator.

        Args:
            repository: Paper repository for data access
        """
        self._repository = repository or PaperRepository()

    def aggregate_by_category(
        self,
        categories: list[str],
        start_date: date,
        end_date: date,
        period: Literal["day", "week", "month"] = "month",
    ) -> pd.DataFrame:
        """Aggregate paper counts by category over time.

        Args:
            categories: List of arXiv category codes
            start_date: Start of date range
            end_date: End of date range
            period: Aggregation period

        Returns:
            DataFrame with columns: date, and one column per category
        """
        df = self._repository.get_by_date_range(start_date, end_date, categories)

        if df.empty:
            return self._empty_time_series(categories, start_date, end_date, period)

        df["period"] = self._get_period(df["submitted_date"], period)

        pivot = df.pivot_table(
            index="period",
            columns="primary_category",
            values="arxiv_id",
            aggfunc="count",
            fill_value=0,
        )

        pivot = pivot.reindex(columns=categories, fill_value=0)

        full_index = self._generate_period_index(start_date, end_date, period)
        pivot = pivot.reindex(full_index, fill_value=0)

        pivot = pivot.reset_index()
        pivot = pivot.rename(columns={"index": "date"})

        return pivot

    def aggregate_by_keyword(
        self,
        keyword: str,
        start_date: date,
        end_date: date,
        period: Literal["day", "week", "month"] = "month",
    ) -> pd.DataFrame:
        """Aggregate paper counts containing a keyword over time.

        Args:
            keyword: Search keyword
            start_date: Start of date range
            end_date: End of date range
            period: Aggregation period

        Returns:
            DataFrame with columns: date, count
        """
        df = self._repository.search_by_keyword(keyword, start_date, end_date)

        if df.empty:
            return self._empty_keyword_series(keyword, start_date, end_date, period)

        df["period"] = self._get_period(df["submitted_date"], period)

        counts = df.groupby("period").size().reset_index(name=keyword)

        full_index = self._generate_period_index(start_date, end_date, period)
        counts = counts.set_index("period").reindex(full_index, fill_value=0)
        counts = counts.reset_index().rename(columns={"index": "date"})

        return counts

    def aggregate_total(
        self,
        start_date: date,
        end_date: date,
        period: Literal["day", "week", "month"] = "month",
    ) -> pd.DataFrame:
        """Aggregate total paper counts over time.

        Args:
            start_date: Start of date range
            end_date: End of date range
            period: Aggregation period

        Returns:
            DataFrame with columns: date, total
        """
        df = self._repository.get_by_date_range(start_date, end_date)

        if df.empty:
            return self._empty_keyword_series("total", start_date, end_date, period)

        df["period"] = self._get_period(df["submitted_date"], period)

        counts = df.groupby("period").size().reset_index(name="total")

        full_index = self._generate_period_index(start_date, end_date, period)
        counts = counts.set_index("period").reindex(full_index, fill_value=0).reset_index()
        counts = counts.rename(columns={"index": "date"})

        return counts

    def _get_period(
        self,
        dates: pd.Series,
        period: Literal["day", "week", "month"],
    ) -> pd.Series:
        """Convert dates to period start dates.

        Args:
            dates: Series of dates
            period: Aggregation period

        Returns:
            Series of period start dates
        """
        if period == "day":
            return dates.dt.normalize()
        elif period == "week":
            return dates.dt.to_period("W").dt.start_time
        else:
            return dates.dt.to_period("M").dt.start_time

    def _generate_period_index(
        self,
        start_date: date,
        end_date: date,
        period: Literal["day", "week", "month"],
    ) -> pd.DatetimeIndex:
        """Generate a complete period index for the date range.

        Args:
            start_date: Start date
            end_date: End date
            period: Aggregation period

        Returns:
            DatetimeIndex covering all periods
        """
        freq_map = {"day": "D", "week": "W-MON", "month": "MS"}
        return pd.date_range(start=start_date, end=end_date, freq=freq_map[period])

    def _empty_time_series(
        self,
        categories: list[str],
        start_date: date,
        end_date: date,
        period: Literal["day", "week", "month"],
    ) -> pd.DataFrame:
        """Create an empty time series DataFrame.

        Args:
            categories: Category columns
            start_date: Start date
            end_date: End date
            period: Aggregation period

        Returns:
            DataFrame with zero values
        """
        index = self._generate_period_index(start_date, end_date, period)
        df = pd.DataFrame(index=index, columns=categories)
        df = df.fillna(0).reset_index()
        df = df.rename(columns={"index": "date"})
        return df

    def _empty_keyword_series(
        self,
        column_name: str,
        start_date: date,
        end_date: date,
        period: Literal["day", "week", "month"],
    ) -> pd.DataFrame:
        """Create an empty keyword series DataFrame.

        Args:
            column_name: Name of the count column
            start_date: Start date
            end_date: End date
            period: Aggregation period

        Returns:
            DataFrame with zero values
        """
        index = self._generate_period_index(start_date, end_date, period)
        df = pd.DataFrame({"date": index, column_name: 0})
        return df

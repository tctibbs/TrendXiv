"""Plotly chart builders for trend visualization."""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.common.constants import ARXIV_CATEGORIES

DARK_COLORS = [
    "#6C63FF",  # Purple (primary)
    "#FF6B6B",  # Coral
    "#4ECDC4",  # Teal
    "#FFE66D",  # Yellow
    "#95E1D3",  # Mint
    "#F38181",  # Pink
    "#AA96DA",  # Lavender
    "#FCBAD3",  # Light Pink
]

DARK_THEME = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#FAFAFA"),
    title=dict(font=dict(color="#FAFAFA", size=18)),
    xaxis=dict(
        gridcolor="rgba(255,255,255,0.1)",
        linecolor="rgba(255,255,255,0.2)",
        tickcolor="rgba(255,255,255,0.2)",
        title_font=dict(color="#888"),
        tickfont=dict(color="#888"),
    ),
    yaxis=dict(
        gridcolor="rgba(255,255,255,0.1)",
        linecolor="rgba(255,255,255,0.2)",
        tickcolor="rgba(255,255,255,0.2)",
        title_font=dict(color="#888"),
        tickfont=dict(color="#888"),
    ),
    legend=dict(
        font=dict(color="#FAFAFA"),
        bgcolor="rgba(0,0,0,0)",
    ),
)


class ChartBuilder:
    """Builds Plotly charts for arXiv trend visualization."""

    def __init__(self, colors: list[str] | None = None, dark_mode: bool = True) -> None:
        """Initialize the chart builder.

        Args:
            colors: Optional custom color palette
            dark_mode: Whether to use dark theme
        """
        self._colors = colors or DARK_COLORS
        self._dark_mode = dark_mode

    def _apply_theme(self, fig: go.Figure) -> go.Figure:
        """Apply dark theme to figure."""
        if self._dark_mode:
            fig.update_layout(**DARK_THEME)
        return fig

    def line_chart(
        self,
        data: pd.DataFrame,
        title: str = "",
        y_axis_title: str = "Papers",
        date_column: str = "date",
        show_legend: bool = True,
    ) -> go.Figure:
        """Create a multi-series line chart.

        Args:
            data: DataFrame with date column and value columns
            title: Chart title
            y_axis_title: Y-axis label
            date_column: Name of the date column
            show_legend: Whether to show the legend

        Returns:
            Plotly Figure object
        """
        fig = go.Figure()

        value_columns = [col for col in data.columns if col != date_column]

        for i, col in enumerate(value_columns):
            display_name = self._get_display_name(col)
            color = self._colors[i % len(self._colors)]

            fig.add_trace(
                go.Scatter(
                    x=data[date_column],
                    y=data[col],
                    mode="lines",
                    name=display_name,
                    line=dict(color=color, width=2.5),
                    hovertemplate=(
                        f"<b>{display_name}</b><br>"
                        "Date: %{x|%B %Y}<br>"
                        f"{y_axis_title}: %{{y:,.0f}}<extra></extra>"
                    ),
                )
            )

        fig.update_layout(
            title=dict(text=title) if title else None,
            xaxis=dict(title=""),
            yaxis=dict(title=y_axis_title),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="left",
                x=0,
            ) if show_legend else dict(visible=False),
            hovermode="x unified",
            margin=dict(l=60, r=30, t=60 if title else 30, b=50),
            height=450,
        )

        return self._apply_theme(fig)

    def area_chart(
        self,
        data: pd.DataFrame,
        title: str = "",
        y_axis_title: str = "Papers",
        date_column: str = "date",
        stacked: bool = True,
    ) -> go.Figure:
        """Create a stacked area chart.

        Args:
            data: DataFrame with date column and value columns
            title: Chart title
            y_axis_title: Y-axis label
            date_column: Name of the date column
            stacked: Whether to stack areas

        Returns:
            Plotly Figure object
        """
        fig = go.Figure()

        value_columns = [col for col in data.columns if col != date_column]

        for i, col in enumerate(value_columns):
            display_name = self._get_display_name(col)
            color = self._colors[i % len(self._colors)]

            fig.add_trace(
                go.Scatter(
                    x=data[date_column],
                    y=data[col],
                    mode="lines",
                    name=display_name,
                    fill="tonexty" if stacked and i > 0 else "tozeroy",
                    line=dict(color=color, width=1),
                    stackgroup="one" if stacked else None,
                    hovertemplate=(
                        f"<b>{display_name}</b><br>"
                        "Date: %{x|%B %Y}<br>"
                        f"{y_axis_title}: %{{y:,.0f}}<extra></extra>"
                    ),
                )
            )

        fig.update_layout(
            title=dict(text=title) if title else None,
            xaxis=dict(title=""),
            yaxis=dict(title=y_axis_title),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="left",
                x=0,
            ),
            hovermode="x unified",
            margin=dict(l=60, r=30, t=60 if title else 30, b=50),
            height=450,
        )

        return self._apply_theme(fig)

    def bar_chart(
        self,
        data: pd.DataFrame,
        title: str = "",
        y_axis_title: str = "Papers",
        date_column: str = "date",
    ) -> go.Figure:
        """Create a grouped bar chart.

        Args:
            data: DataFrame with date column and value columns
            title: Chart title
            y_axis_title: Y-axis label
            date_column: Name of the date column

        Returns:
            Plotly Figure object
        """
        value_columns = [col for col in data.columns if col != date_column]

        melted = data.melt(
            id_vars=[date_column],
            value_vars=value_columns,
            var_name="category",
            value_name="count",
        )

        melted["display_name"] = melted["category"].apply(self._get_display_name)

        fig = px.bar(
            melted,
            x=date_column,
            y="count",
            color="display_name",
            barmode="group",
            title=title if title else None,
            labels={"count": y_axis_title, date_column: ""},
            color_discrete_sequence=self._colors,
        )

        fig.update_layout(
            legend_title_text="",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="left",
                x=0,
            ),
            margin=dict(l=60, r=30, t=60 if title else 30, b=50),
            height=450,
        )

        return self._apply_theme(fig)

    def comparison_chart(
        self,
        data: pd.DataFrame,
        title: str = "Category Comparison",
        date_column: str = "date",
        normalize: bool = False,
    ) -> go.Figure:
        """Create a comparison chart with optional normalization.

        Args:
            data: DataFrame with date column and value columns
            title: Chart title
            date_column: Name of the date column
            normalize: Whether to normalize to index 100 at start

        Returns:
            Plotly Figure object
        """
        plot_data = data.copy()
        value_columns = [col for col in plot_data.columns if col != date_column]

        if normalize:
            for col in value_columns:
                col_data = plot_data[col]
                has_positive = (col_data > 0).any()
                first_nonzero = col_data[col_data > 0].iloc[0] if has_positive else 1
                plot_data[col] = (col_data / first_nonzero) * 100

        y_title = "Index (Base = 100)" if normalize else "Papers"

        return self.line_chart(
            plot_data,
            title=title,
            y_axis_title=y_title,
            date_column=date_column,
        )

    def _get_display_name(self, category: str) -> str:
        """Get the display name for a category.

        Args:
            category: Category code or keyword

        Returns:
            Human-readable display name
        """
        if category in ARXIV_CATEGORIES:
            return f"{category}: {ARXIV_CATEGORIES[category].name}"
        return category

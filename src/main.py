"""TrendXiv - Google Trends for arXiv.

Main Streamlit application for visualizing arXiv publication trends.
"""

import logging
from datetime import date, timedelta

import streamlit as st

from src.api.client import ArxivClient
from src.common.config import get_settings
from src.common.constants import (
    ARXIV_CATEGORIES,
    DEFAULT_CATEGORIES,
    SMOOTHING_OPTIONS,
)
from src.data.aggregator import TimeSeriesAggregator
from src.data.normalizer import DataNormalizer
from src.data.smoother import MovingAverageCalculator
from src.repository.database import DatabaseManager
from src.repository.papers import PaperRepository
from src.visualization.charts import ChartBuilder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def init_database() -> tuple[DatabaseManager, PaperRepository]:
    """Initialize database connection."""
    db = DatabaseManager()
    if not db.database_exists:
        db.initialize_schema()
    return db, PaperRepository(db)


@st.cache_resource
def get_db_manager() -> DatabaseManager:
    """Get cached database manager."""
    db = DatabaseManager()
    if not db.database_exists:
        db.initialize_schema()
    return db


@st.cache_resource
def get_repository() -> PaperRepository:
    """Get cached paper repository."""
    return PaperRepository(get_db_manager())


@st.cache_data(ttl=3600)
def fetch_papers_for_category(
    category: str,
    start_date: date,
    end_date: date,
    max_results: int = 5000,
) -> int:
    """Fetch papers from arXiv API and store in database.

    Args:
        category: arXiv category code
        start_date: Start date
        end_date: End date
        max_results: Maximum papers to fetch

    Returns:
        Number of papers fetched
    """
    client = ArxivClient()
    repo = get_repository()

    papers = list(
        client.search_by_category(
            category=category,
            start_date=start_date,
            end_date=end_date,
            max_results=max_results,
        )
    )

    if papers:
        inserted = repo.batch_insert(papers)
        repo.log_harvest(category, "incremental", start_date, end_date, inserted)
        return inserted

    return 0


def render_sidebar() -> dict:
    """Render the sidebar controls and return selected options."""
    st.sidebar.title("TrendXiv")
    st.sidebar.markdown("*Google Trends for arXiv*")
    st.sidebar.divider()

    category_options = {
        f"{code}: {cat.name}": code for code, cat in ARXIV_CATEGORIES.items()
    }

    selected_display = st.sidebar.multiselect(
        "Categories",
        options=list(category_options.keys()),
        default=[f"{c}: {ARXIV_CATEGORIES[c].name}" for c in DEFAULT_CATEGORIES[:3]],
        help="Select arXiv categories to compare",
    )
    selected_categories = [category_options[d] for d in selected_display]

    st.sidebar.divider()

    keyword = st.sidebar.text_input(
        "Keyword Search",
        placeholder="e.g., transformer, diffusion",
        help="Search for papers containing this keyword in title/abstract",
    )

    st.sidebar.divider()

    settings = get_settings()
    default_start = date.today() - timedelta(days=365 * settings.default_lookback_years)

    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_date = st.date_input(
            "Start Date",
            value=default_start,
            min_value=date(2010, 1, 1),
            max_value=date.today(),
        )
    with col2:
        end_date = st.date_input(
            "End Date",
            value=date.today(),
            min_value=date(2010, 1, 1),
            max_value=date.today(),
        )

    st.sidebar.divider()

    normalization = st.sidebar.radio(
        "Normalization",
        options=["Absolute", "Relative"],
        help="Absolute: raw paper counts. Relative: percentage of total.",
    )

    smoothing = st.sidebar.selectbox(
        "Smoothing",
        options=list(SMOOTHING_OPTIONS.keys()),
        help="Apply moving average to smooth the data",
    )

    st.sidebar.divider()

    chart_type = st.sidebar.radio(
        "Chart Type",
        options=["Line", "Area", "Bar"],
    )

    return {
        "categories": selected_categories,
        "keyword": keyword.strip() if keyword else None,
        "start_date": start_date,
        "end_date": end_date,
        "normalization": normalization.lower(),
        "smoothing_months": SMOOTHING_OPTIONS[smoothing],
        "chart_type": chart_type.lower(),
    }


def render_data_status(repo: PaperRepository) -> None:
    """Render data status information."""
    total = repo.get_total_count()
    date_range = repo.get_date_range()

    if total > 0 and date_range[0]:
        st.sidebar.divider()
        st.sidebar.caption(
            f"Database: {total:,} papers\n\n"
            f"Range: {date_range[0]} to {date_range[1]}"
        )


def fetch_data_with_progress(
    categories: list[str],
    start_date: date,
    end_date: date,
) -> None:
    """Fetch data for categories with a progress bar."""
    if not categories:
        return

    progress_bar = st.progress(0, text="Fetching data from arXiv...")

    for i, category in enumerate(categories):
        progress_bar.progress(
            (i + 1) / len(categories),
            text=f"Fetching {category}...",
        )
        fetch_papers_for_category(category, start_date, end_date)

    progress_bar.empty()


def main() -> None:
    """Main application entry point."""
    st.set_page_config(
        page_title="TrendXiv",
        page_icon="📈",
        layout="wide",
    )

    repo = get_repository()
    options = render_sidebar()
    render_data_status(repo)

    st.title("📈 TrendXiv")
    st.markdown(
        "Visualize publication trends across arXiv categories and keywords. "
        "Compare the growth of Machine Learning, NLP, Computer Vision, and more."
    )

    if not options["categories"] and not options["keyword"]:
        st.info("Select categories or enter a keyword to get started.")
        return

    if st.sidebar.button("🔄 Fetch Latest Data", type="primary"):
        with st.spinner("Fetching data from arXiv API..."):
            fetch_data_with_progress(
                options["categories"],
                options["start_date"],
                options["end_date"],
            )
            st.cache_data.clear()
        st.success("Data updated!")
        st.rerun()

    aggregator = TimeSeriesAggregator(repo)
    normalizer = DataNormalizer()
    smoother = MovingAverageCalculator()
    chart_builder = ChartBuilder()

    if options["categories"]:
        data = aggregator.aggregate_by_category(
            categories=options["categories"],
            start_date=options["start_date"],
            end_date=options["end_date"],
        )

        if data.empty or data.drop(columns=["date"]).sum().sum() == 0:
            st.warning(
                "No data found for the selected categories and date range. "
                "Click 'Fetch Latest Data' to download papers from arXiv."
            )
        else:
            if options["normalization"] == "relative":
                data = normalizer.to_relative(data)

            if options["smoothing_months"] > 0:
                data = smoother.apply(data, options["smoothing_months"])

            y_title = "Papers" if options["normalization"] == "absolute" else "Share (%)"

            if options["chart_type"] == "line":
                fig = chart_builder.line_chart(
                    data,
                    title="arXiv Publication Trends by Category",
                    y_axis_title=y_title,
                )
            elif options["chart_type"] == "area":
                fig = chart_builder.area_chart(
                    data,
                    title="arXiv Publication Trends by Category",
                    y_axis_title=y_title,
                )
            else:
                fig = chart_builder.bar_chart(
                    data,
                    title="arXiv Publication Trends by Category",
                    y_axis_title=y_title,
                )

            st.plotly_chart(fig, use_container_width=True)

            with st.expander("View Data Table"):
                st.dataframe(data, use_container_width=True)

    if options["keyword"]:
        st.divider()
        st.subheader(f"Keyword: '{options['keyword']}'")

        keyword_data = aggregator.aggregate_by_keyword(
            keyword=options["keyword"],
            start_date=options["start_date"],
            end_date=options["end_date"],
        )

        if keyword_data.empty or keyword_data[options["keyword"]].sum() == 0:
            st.info(
                f"No papers found containing '{options['keyword']}'. "
                "Make sure to fetch data first, then the keyword search will work."
            )
        else:
            if options["smoothing_months"] > 0:
                keyword_data = smoother.apply(
                    keyword_data, options["smoothing_months"]
                )

            fig = chart_builder.line_chart(
                keyword_data,
                title=f"Papers Mentioning '{options['keyword']}'",
                y_axis_title="Papers",
            )
            st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()

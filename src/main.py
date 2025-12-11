"""TrendXiv - Google Trends for arXiv.

Main Streamlit application for visualizing arXiv publication trends.
"""

import logging
import time
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

CUSTOM_CSS = """
<style>
/* Clean up sidebar */
section[data-testid="stSidebar"] {
    background-color: #1E1E2E;
}

section[data-testid="stSidebar"] .stMarkdown h3 {
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #888;
    margin-top: 1.5rem;
    margin-bottom: 0.5rem;
}

/* Muted multiselect tags */
div[data-testid="stMultiSelect"] span[data-baseweb="tag"] {
    background-color: #3D3D5C !important;
    border-radius: 4px;
}

/* Remove default Streamlit branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

/* Better metrics styling */
div[data-testid="stMetric"] {
    background-color: #1E1E2E;
    padding: 1rem;
    border-radius: 8px;
}

div[data-testid="stMetric"] label {
    color: #888 !important;
}

/* Status container styling */
div[data-testid="stStatusWidget"] {
    background-color: #1E1E2E;
    border-radius: 8px;
}

/* Expander styling */
div[data-testid="stExpander"] {
    background-color: #1E1E2E;
    border-radius: 8px;
    border: none;
}

/* Button styling */
button[kind="primary"] {
    background-color: #6C63FF !important;
    border: none !important;
}

button[kind="primary"]:hover {
    background-color: #5A52E0 !important;
}
</style>
"""


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


def fetch_with_streaming_progress(
    categories: list[str],
    start_date: date,
    end_date: date,
) -> dict[str, int]:
    """Fetch papers with detailed streaming progress, chunked by year.

    Args:
        categories: List of arXiv category codes
        start_date: Start date for fetching
        end_date: End date for fetching

    Returns:
        Dict mapping category to papers fetched count
    """
    client = ArxivClient()
    repo = get_repository()
    results = {}
    total_start = time.time()

    years = list(range(start_date.year, end_date.year + 1))

    with st.status("Fetching papers from arXiv...", expanded=True) as status:
        for category in categories:
            cat_start = time.time()
            cat_total = 0

            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"**{category}**")
            count_placeholder = col2.empty()
            time_placeholder = col3.empty()

            count_placeholder.markdown("`0` papers")
            time_placeholder.markdown("`0s` elapsed")

            progress_bar = st.progress(0)

            for year_idx, year in enumerate(years):
                year_start = date(year, 1, 1) if year > start_date.year else start_date
                year_end = date(year, 12, 31) if year < end_date.year else end_date

                papers_batch = []
                for paper in client.search_by_category(
                    category=category,
                    start_date=year_start,
                    end_date=year_end,
                    max_results=30000,
                ):
                    papers_batch.append(paper)

                    if len(papers_batch) % 100 == 0:
                        elapsed = time.time() - cat_start
                        count_placeholder.markdown(f"`{cat_total + len(papers_batch):,}` papers")
                        time_placeholder.markdown(f"`{elapsed:.0f}s` ({year})")

                if papers_batch:
                    repo.batch_insert(papers_batch)
                    cat_total += len(papers_batch)

                progress = (year_idx + 1) / len(years)
                progress_bar.progress(progress)

                elapsed = time.time() - cat_start
                count_placeholder.markdown(f"`{cat_total:,}` papers")
                time_placeholder.markdown(f"`{elapsed:.0f}s` elapsed")

            progress_bar.progress(1.0)
            results[category] = cat_total

            repo.log_harvest(category, "incremental", start_date, end_date, cat_total)

        total_elapsed = time.time() - total_start
        total_papers = sum(results.values())
        status.update(
            label=f"Fetched {total_papers:,} papers in {total_elapsed:.0f}s",
            state="complete",
            expanded=False,
        )

    return results


def render_sidebar() -> dict:
    """Render the sidebar controls and return selected options."""
    with st.sidebar:
        st.markdown("## TrendXiv")
        st.caption("Google Trends for arXiv")

        st.markdown("### Categories")
        selected_categories = st.multiselect(
            "Select categories",
            options=list(ARXIV_CATEGORIES.keys()),
            default=DEFAULT_CATEGORIES[:3],
            format_func=lambda x: x,
            label_visibility="collapsed",
        )

        if selected_categories:
            names = [ARXIV_CATEGORIES[c].name for c in selected_categories]
            st.caption(", ".join(names))

        st.markdown("### Keyword")
        keyword = st.text_input(
            "Search abstracts",
            placeholder="transformer, diffusion...",
            label_visibility="collapsed",
        )

        st.markdown("### Date Range")
        settings = get_settings()
        default_start = date.today() - timedelta(days=365 * settings.default_lookback_years)

        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "From",
                value=default_start,
                min_value=date(2010, 1, 1),
                max_value=date.today(),
            )
        with col2:
            end_date = st.date_input(
                "To",
                value=date.today(),
                min_value=date(2010, 1, 1),
                max_value=date.today(),
            )

        st.markdown("### Display")
        col1, col2 = st.columns(2)
        with col1:
            normalization = st.radio(
                "Scale",
                options=["Absolute", "Relative"],
                label_visibility="collapsed",
                horizontal=True,
            )
        with col2:
            smoothing = st.selectbox(
                "Smooth",
                options=list(SMOOTHING_OPTIONS.keys()),
                label_visibility="collapsed",
            )

        chart_type = st.radio(
            "Chart",
            options=["Line", "Area", "Bar"],
            horizontal=True,
            label_visibility="collapsed",
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


def render_metrics(repo: PaperRepository) -> None:
    """Render metrics row."""
    total = repo.get_total_count()
    date_range = repo.get_date_range()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Papers", f"{total:,}" if total > 0 else "No data")

    with col2:
        if date_range[0]:
            st.metric("Date Range", f"{date_range[0]} to {date_range[1]}")
        else:
            st.metric("Date Range", "No data")

    with col3:
        st.metric("Categories", len(ARXIV_CATEGORIES))


def main() -> None:
    """Main application entry point."""
    st.set_page_config(
        page_title="TrendXiv",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    repo = get_repository()
    options = render_sidebar()

    st.markdown("# 📈 TrendXiv")
    st.markdown(
        "Visualize publication trends across arXiv categories. "
        "Compare Machine Learning, NLP, Computer Vision, and more."
    )

    render_metrics(repo)

    st.markdown("---")

    if not options["categories"] and not options["keyword"]:
        st.info("👈 Select categories from the sidebar to get started.")
        return

    total_papers = repo.get_total_count()

    col1, col2 = st.columns([3, 1])
    with col2:
        fetch_clicked = st.button(
            "🔄 Fetch Data",
            type="primary",
            use_container_width=True,
        )

    if fetch_clicked:
        fetch_with_streaming_progress(
            options["categories"],
            options["start_date"],
            options["end_date"],
        )
        st.cache_data.clear()
        st.rerun()

    if total_papers == 0:
        st.warning(
            "No papers in database yet. Click **Fetch Data** to download papers from arXiv."
        )
        return

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
                "No data for selected categories in this date range. "
                "Try fetching more data or adjusting the date range."
            )
        else:
            if options["normalization"] == "relative":
                data = normalizer.to_relative(data)

            if options["smoothing_months"] > 0:
                data = smoother.apply(data, options["smoothing_months"])

            y_title = "Papers" if options["normalization"] == "absolute" else "Share (%)"

            if options["chart_type"] == "line":
                fig = chart_builder.line_chart(data, y_axis_title=y_title)
            elif options["chart_type"] == "area":
                fig = chart_builder.area_chart(data, y_axis_title=y_title)
            else:
                fig = chart_builder.bar_chart(data, y_axis_title=y_title)

            st.plotly_chart(fig, use_container_width=True)

            with st.expander("📊 View Data Table"):
                st.dataframe(
                    data.style.format({col: "{:,.0f}" for col in data.columns if col != "date"}),
                    use_container_width=True,
                )

    if options["keyword"]:
        st.markdown("---")
        st.markdown(f"### Keyword: `{options['keyword']}`")

        keyword_data = aggregator.aggregate_by_keyword(
            keyword=options["keyword"],
            start_date=options["start_date"],
            end_date=options["end_date"],
        )

        if keyword_data.empty or keyword_data[options["keyword"]].sum() == 0:
            st.info(
                f"No papers found containing '{options['keyword']}'. "
                "Fetch data first to enable keyword search."
            )
        else:
            if options["smoothing_months"] > 0:
                keyword_data = smoother.apply(keyword_data, options["smoothing_months"])

            fig = chart_builder.line_chart(
                keyword_data,
                title=f"Papers Mentioning '{options['keyword']}'",
                y_axis_title="Papers",
            )
            st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()

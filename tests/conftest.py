"""Shared pytest fixtures for TrendXiv tests."""

from datetime import date, timedelta

import pandas as pd
import pytest

from src.api.models import ArxivPaper
from src.common.config import Settings
from src.repository.database import DatabaseManager
from src.repository.papers import PaperRepository


@pytest.fixture
def sample_papers() -> list[ArxivPaper]:
    """Generate sample papers for testing."""
    papers = []
    base_date = date(2024, 1, 1)

    categories = ["cs.LG", "cs.AI", "cs.CL", "cs.CV", "stat.ML"]

    for i in range(100):
        category = categories[i % len(categories)]
        papers.append(
            ArxivPaper(
                arxiv_id=f"2401.{i:05d}",
                title=f"Test Paper {i} on {category}",
                abstract=f"Abstract for paper {i}. Contains keyword transformer for testing.",
                primary_category=category,
                categories=[category],
                submitted_date=base_date + timedelta(days=i % 30),
                authors=[f"Author {i}", f"CoAuthor {i}"],
            )
        )

    return papers


@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    """Generate sample time series DataFrame."""
    dates = pd.date_range(start="2024-01-01", periods=12, freq="MS")
    return pd.DataFrame(
        {
            "date": dates,
            "cs.LG": [100, 120, 150, 180, 200, 220, 250, 280, 300, 320, 350, 400],
            "cs.AI": [50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160],
            "cs.CL": [30, 35, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130],
        }
    )


@pytest.fixture
def test_settings(tmp_path) -> Settings:
    """Create test settings with temporary database."""
    return Settings(
        database_path=str(tmp_path / "test.db"),
        arxiv_delay_seconds=0.1,
        arxiv_page_size=10,
        arxiv_max_results=100,
    )


@pytest.fixture
def test_db(test_settings) -> DatabaseManager:
    """Create test database with initialized schema."""
    db = DatabaseManager(test_settings)
    db.initialize_schema()
    return db


@pytest.fixture
def test_repository(test_db, test_settings) -> PaperRepository:
    """Create test paper repository."""
    return PaperRepository(test_db, test_settings)


@pytest.fixture
def populated_repository(test_repository, sample_papers) -> PaperRepository:
    """Create repository populated with sample papers."""
    test_repository.batch_insert(sample_papers)
    return test_repository

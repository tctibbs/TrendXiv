"""Data models for arXiv API responses."""

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import arxiv


@dataclass
class ArxivPaper:
    """Represents an arXiv paper with metadata."""

    arxiv_id: str
    title: str
    abstract: str
    primary_category: str
    categories: list[str]
    submitted_date: date
    authors: list[str] = field(default_factory=list)
    updated_date: date | None = None

    @classmethod
    def from_arxiv_result(cls, result: "arxiv.Result") -> "ArxivPaper":
        """Create ArxivPaper from arxiv library Result object.

        Args:
            result: Result object from arxiv library search

        Returns:
            ArxivPaper instance with extracted metadata
        """
        return cls(
            arxiv_id=result.entry_id.split("/")[-1],
            title=result.title.replace("\n", " ").strip(),
            abstract=result.summary.replace("\n", " ").strip(),
            primary_category=result.primary_category,
            categories=[cat for cat in result.categories],
            submitted_date=result.published.date(),
            authors=[author.name for author in result.authors],
            updated_date=result.updated.date() if result.updated else None,
        )

    def __str__(self) -> str:
        return f"ArxivPaper({self.arxiv_id}: {self.title[:50]}...)"


@dataclass
class SearchQuery:
    """Represents a search query for arXiv papers."""

    categories: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    start_date: date | None = None
    end_date: date | None = None

    def to_arxiv_query(self) -> str:
        """Convert to arXiv API query string.

        Returns:
            Query string compatible with arXiv API
        """
        parts = []

        if self.categories:
            cat_query = " OR ".join(f"cat:{cat}" for cat in self.categories)
            if len(self.categories) > 1:
                cat_query = f"({cat_query})"
            parts.append(cat_query)

        if self.keywords:
            kw_query = " OR ".join(f'abs:"{kw}"' for kw in self.keywords)
            if len(self.keywords) > 1:
                kw_query = f"({kw_query})"
            parts.append(kw_query)

        if self.start_date and self.end_date:
            start_str = self.start_date.strftime("%Y%m%d") + "0000"
            end_str = self.end_date.strftime("%Y%m%d") + "2359"
            parts.append(f"submittedDate:[{start_str} TO {end_str}]")

        return " AND ".join(parts) if parts else "*"


@dataclass
class FetchProgress:
    """Tracks progress of data fetching operations."""

    total_expected: int
    fetched: int = 0
    category: str = ""
    started_at: datetime = field(default_factory=datetime.now)

    @property
    def progress_pct(self) -> float:
        """Return progress as percentage."""
        if self.total_expected == 0:
            return 0.0
        return (self.fetched / self.total_expected) * 100

    @property
    def is_complete(self) -> bool:
        """Check if fetching is complete."""
        return self.fetched >= self.total_expected

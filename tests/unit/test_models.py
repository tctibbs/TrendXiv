"""Unit tests for API models."""

from datetime import date

from src.api.models import ArxivPaper, FetchProgress, SearchQuery


class TestArxivPaper:
    """Tests for ArxivPaper dataclass."""

    def test_creation(self):
        """Test basic paper creation."""
        paper = ArxivPaper(
            arxiv_id="2401.00001",
            title="Test Paper",
            abstract="Test abstract",
            primary_category="cs.LG",
            categories=["cs.LG", "cs.AI"],
            submitted_date=date(2024, 1, 1),
        )

        assert paper.arxiv_id == "2401.00001"
        assert paper.title == "Test Paper"
        assert paper.primary_category == "cs.LG"
        assert len(paper.categories) == 2

    def test_str_representation(self):
        """Test string representation."""
        paper = ArxivPaper(
            arxiv_id="2401.00001",
            title="A Very Long Title That Should Be Truncated For Display",
            abstract="Abstract",
            primary_category="cs.LG",
            categories=["cs.LG"],
            submitted_date=date(2024, 1, 1),
        )

        result = str(paper)

        assert "2401.00001" in result
        assert "..." in result

    def test_default_authors(self):
        """Test default empty authors list."""
        paper = ArxivPaper(
            arxiv_id="2401.00001",
            title="Test",
            abstract="Abstract",
            primary_category="cs.LG",
            categories=["cs.LG"],
            submitted_date=date(2024, 1, 1),
        )

        assert paper.authors == []


class TestSearchQuery:
    """Tests for SearchQuery dataclass."""

    def test_category_query(self):
        """Test query with single category."""
        query = SearchQuery(categories=["cs.LG"])

        result = query.to_arxiv_query()

        assert result == "cat:cs.LG"

    def test_multiple_categories(self):
        """Test query with multiple categories."""
        query = SearchQuery(categories=["cs.LG", "cs.AI"])

        result = query.to_arxiv_query()

        assert "cat:cs.LG" in result
        assert "cat:cs.AI" in result
        assert "OR" in result

    def test_keyword_query(self):
        """Test query with keyword."""
        query = SearchQuery(keywords=["transformer"])

        result = query.to_arxiv_query()

        assert 'abs:"transformer"' in result

    def test_multiple_keywords(self):
        """Test query with multiple keywords."""
        query = SearchQuery(keywords=["transformer", "attention"])

        result = query.to_arxiv_query()

        assert "transformer" in result
        assert "attention" in result
        assert "OR" in result

    def test_combined_query(self):
        """Test query with categories and keywords."""
        query = SearchQuery(
            categories=["cs.LG"],
            keywords=["transformer"],
        )

        result = query.to_arxiv_query()

        assert "cat:cs.LG" in result
        assert "transformer" in result
        assert "AND" in result

    def test_empty_query(self):
        """Test empty query returns wildcard."""
        query = SearchQuery()

        result = query.to_arxiv_query()

        assert result == "*"


class TestFetchProgress:
    """Tests for FetchProgress dataclass."""

    def test_progress_pct(self):
        """Test progress percentage calculation."""
        progress = FetchProgress(total_expected=100, fetched=50)

        assert progress.progress_pct == 50.0

    def test_progress_pct_zero_total(self):
        """Test progress percentage with zero total."""
        progress = FetchProgress(total_expected=0, fetched=0)

        assert progress.progress_pct == 0.0

    def test_is_complete_false(self):
        """Test incomplete status."""
        progress = FetchProgress(total_expected=100, fetched=50)

        assert not progress.is_complete

    def test_is_complete_true(self):
        """Test complete status."""
        progress = FetchProgress(total_expected=100, fetched=100)

        assert progress.is_complete

    def test_is_complete_over(self):
        """Test complete status when fetched exceeds expected."""
        progress = FetchProgress(total_expected=100, fetched=150)

        assert progress.is_complete

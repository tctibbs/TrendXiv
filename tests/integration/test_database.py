"""Integration tests for database operations."""

from datetime import date

from src.api.models import ArxivPaper


class TestDatabaseManager:
    """Integration tests for DatabaseManager."""

    def test_initialize_schema(self, test_db):
        """Test schema initialization."""
        counts = test_db.get_table_counts()

        assert "papers" in counts
        assert "paper_categories" in counts
        assert "aggregation_cache" in counts

    def test_database_exists(self, test_db):
        """Test database exists after initialization."""
        assert test_db.database_exists

    def test_get_table_counts_empty(self, test_db):
        """Test table counts on empty database."""
        counts = test_db.get_table_counts()

        assert counts["papers"] == 0
        assert counts["paper_categories"] == 0


class TestPaperRepository:
    """Integration tests for PaperRepository."""

    def test_insert_paper(self, test_repository):
        """Test inserting a single paper."""
        paper = ArxivPaper(
            arxiv_id="2401.00001",
            title="Test Paper",
            abstract="Test abstract",
            primary_category="cs.LG",
            categories=["cs.LG", "cs.AI"],
            submitted_date=date(2024, 1, 15),
            authors=["Author One", "Author Two"],
        )

        result = test_repository.insert(paper)

        assert result is True
        assert test_repository.get_total_count() == 1

    def test_insert_duplicate_fails(self, test_repository):
        """Test that inserting duplicate returns False."""
        paper = ArxivPaper(
            arxiv_id="2401.00001",
            title="Test Paper",
            abstract="Test abstract",
            primary_category="cs.LG",
            categories=["cs.LG"],
            submitted_date=date(2024, 1, 15),
        )

        test_repository.insert(paper)
        result = test_repository.insert(paper)

        assert result is False
        assert test_repository.get_total_count() == 1

    def test_get_by_id(self, test_repository):
        """Test retrieving paper by ID."""
        paper = ArxivPaper(
            arxiv_id="2401.00001",
            title="Test Paper",
            abstract="Test abstract",
            primary_category="cs.LG",
            categories=["cs.LG", "cs.AI"],
            submitted_date=date(2024, 1, 15),
            authors=["Author One"],
        )
        test_repository.insert(paper)

        retrieved = test_repository.get_by_id("2401.00001")

        assert retrieved is not None
        assert retrieved.arxiv_id == "2401.00001"
        assert retrieved.title == "Test Paper"
        assert "cs.LG" in retrieved.categories
        assert "cs.AI" in retrieved.categories
        assert "Author One" in retrieved.authors

    def test_get_by_id_not_found(self, test_repository):
        """Test retrieving non-existent paper."""
        result = test_repository.get_by_id("nonexistent")

        assert result is None

    def test_batch_insert(self, test_repository, sample_papers):
        """Test batch inserting papers."""
        inserted = test_repository.batch_insert(sample_papers)

        assert inserted == len(sample_papers)
        assert test_repository.get_total_count() == len(sample_papers)

    def test_get_by_date_range(self, populated_repository):
        """Test retrieving papers by date range."""
        df = populated_repository.get_by_date_range(
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 31),
        )

        assert not df.empty
        assert "arxiv_id" in df.columns
        assert "primary_category" in df.columns

    def test_get_by_date_range_with_categories(self, populated_repository):
        """Test retrieving papers by date range with category filter."""
        df = populated_repository.get_by_date_range(
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 31),
            categories=["cs.LG"],
        )

        assert not df.empty
        assert all(df["primary_category"] == "cs.LG")

    def test_get_date_range(self, populated_repository):
        """Test getting date range of stored papers."""
        min_date, max_date = populated_repository.get_date_range()

        assert min_date is not None
        assert max_date is not None
        assert min_date <= max_date

    def test_get_date_range_empty(self, test_repository):
        """Test getting date range from empty database."""
        min_date, max_date = test_repository.get_date_range()

        assert min_date is None
        assert max_date is None

    def test_get_category_counts_by_month(self, populated_repository):
        """Test monthly category counts."""
        df = populated_repository.get_category_counts_by_month(
            categories=["cs.LG", "cs.AI"],
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 31),
        )

        assert not df.empty
        assert "month" in df.columns
        assert "category" in df.columns
        assert "count" in df.columns

    def test_search_by_keyword(self, populated_repository):
        """Test keyword search using FTS."""
        df = populated_repository.search_by_keyword("transformer")

        assert not df.empty

    def test_aggregation_cache(self, test_repository, sample_dataframe):
        """Test aggregation cache save and retrieve."""
        cache_key = "test_cache_key"
        test_repository.save_aggregation_cache(cache_key, sample_dataframe)

        retrieved = test_repository.get_aggregation_cache(cache_key)

        assert retrieved is not None
        assert len(retrieved) == len(sample_dataframe)

    def test_aggregation_cache_not_found(self, test_repository):
        """Test retrieving non-existent cache."""
        result = test_repository.get_aggregation_cache("nonexistent")

        assert result is None

    def test_harvest_logging(self, test_repository):
        """Test harvest log operations."""
        test_repository.log_harvest(
            category="cs.LG",
            harvest_type="full",
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 31),
            records_fetched=100,
        )

        last_date = test_repository.get_last_harvest_date("cs.LG")

        assert last_date == date(2024, 1, 31)

    def test_get_last_harvest_date_none(self, test_repository):
        """Test getting harvest date for unharvested category."""
        result = test_repository.get_last_harvest_date("stat.ML")

        assert result is None

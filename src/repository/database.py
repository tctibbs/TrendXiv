"""SQLite database connection and schema management."""

import logging
import sqlite3
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

from src.common.config import Settings, get_settings

logger = logging.getLogger(__name__)

SCHEMA_SQL = """
-- Papers table (core metadata)
CREATE TABLE IF NOT EXISTS papers (
    arxiv_id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    abstract TEXT,
    primary_category TEXT NOT NULL,
    submitted_date DATE NOT NULL,
    updated_date DATE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Paper categories (many-to-many for cross-listed papers)
CREATE TABLE IF NOT EXISTS paper_categories (
    arxiv_id TEXT REFERENCES papers(arxiv_id) ON DELETE CASCADE,
    category TEXT NOT NULL,
    PRIMARY KEY (arxiv_id, category)
);

-- Paper authors
CREATE TABLE IF NOT EXISTS paper_authors (
    arxiv_id TEXT REFERENCES papers(arxiv_id) ON DELETE CASCADE,
    author_name TEXT NOT NULL,
    author_order INTEGER,
    PRIMARY KEY (arxiv_id, author_order)
);

-- Aggregation cache for pre-computed time series
CREATE TABLE IF NOT EXISTS aggregation_cache (
    cache_key TEXT PRIMARY KEY,
    data_json TEXT NOT NULL,
    computed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP
);

-- Harvest tracking for incremental updates
CREATE TABLE IF NOT EXISTS harvest_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    category TEXT,
    harvest_type TEXT,
    start_date DATE,
    end_date DATE,
    records_fetched INTEGER,
    completed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_papers_primary_category
ON papers(primary_category);

CREATE INDEX IF NOT EXISTS idx_papers_submitted_date
ON papers(submitted_date);

CREATE INDEX IF NOT EXISTS idx_papers_category_date
ON papers(primary_category, submitted_date);

CREATE INDEX IF NOT EXISTS idx_paper_categories_category
ON paper_categories(category);

CREATE INDEX IF NOT EXISTS idx_aggregation_cache_expires
ON aggregation_cache(expires_at);
"""

FTS_SCHEMA_SQL = """
-- Full-text search for keywords in titles and abstracts
CREATE VIRTUAL TABLE IF NOT EXISTS papers_fts USING fts5(
    arxiv_id,
    title,
    abstract,
    content='papers',
    content_rowid='rowid'
);

-- Triggers to keep FTS index in sync
CREATE TRIGGER IF NOT EXISTS papers_ai AFTER INSERT ON papers BEGIN
    INSERT INTO papers_fts(rowid, arxiv_id, title, abstract)
    VALUES (new.rowid, new.arxiv_id, new.title, new.abstract);
END;

CREATE TRIGGER IF NOT EXISTS papers_ad AFTER DELETE ON papers BEGIN
    INSERT INTO papers_fts(papers_fts, rowid, arxiv_id, title, abstract)
    VALUES ('delete', old.rowid, old.arxiv_id, old.title, old.abstract);
END;

CREATE TRIGGER IF NOT EXISTS papers_au AFTER UPDATE ON papers BEGIN
    INSERT INTO papers_fts(papers_fts, rowid, arxiv_id, title, abstract)
    VALUES ('delete', old.rowid, old.arxiv_id, old.title, old.abstract);
    INSERT INTO papers_fts(rowid, arxiv_id, title, abstract)
    VALUES (new.rowid, new.arxiv_id, new.title, new.abstract);
END;
"""


class DatabaseManager:
    """Manages SQLite database connections and schema initialization."""

    def __init__(self, settings: Settings | None = None) -> None:
        """Initialize the database manager.

        Args:
            settings: Application settings. Uses defaults if not provided.
        """
        self._settings = settings or get_settings()
        self._db_path = Path(self._settings.database_path)
        self._ensure_directory()

    def _ensure_directory(self) -> None:
        """Ensure the database directory exists."""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Get a database connection with proper cleanup.

        Yields:
            SQLite connection object
        """
        conn = sqlite3.connect(
            self._db_path,
            detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES,
        )
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")

        try:
            yield conn
        finally:
            conn.close()

    def initialize_schema(self) -> None:
        """Create database tables and indexes if they don't exist."""
        logger.info(f"Initializing database schema at {self._db_path}")

        with self.get_connection() as conn:
            conn.executescript(SCHEMA_SQL)
            conn.executescript(FTS_SCHEMA_SQL)
            conn.commit()

        logger.info("Database schema initialized successfully")

    def reset_database(self) -> None:
        """Drop all tables and recreate the schema.

        Warning: This will delete all data!
        """
        logger.warning("Resetting database - all data will be lost!")

        if self._db_path.exists():
            self._db_path.unlink()

        self.initialize_schema()

    def get_table_counts(self) -> dict[str, int]:
        """Get row counts for all tables.

        Returns:
            Dictionary mapping table names to row counts
        """
        tables = ["papers", "paper_categories", "paper_authors", "aggregation_cache"]
        counts = {}

        with self.get_connection() as conn:
            for table in tables:
                result = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
                counts[table] = result[0] if result else 0

        return counts

    @property
    def database_exists(self) -> bool:
        """Check if the database file exists."""
        return self._db_path.exists()

    @property
    def database_path(self) -> Path:
        """Get the database file path."""
        return self._db_path

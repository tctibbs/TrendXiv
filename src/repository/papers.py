"""Paper repository for database operations."""

import logging
from datetime import date, datetime, timedelta

import pandas as pd

from src.api.models import ArxivPaper
from src.common.config import Settings, get_settings
from src.repository.database import DatabaseManager

logger = logging.getLogger(__name__)


class PaperRepository:
    """Repository for paper storage and retrieval operations."""

    def __init__(
        self,
        db_manager: DatabaseManager | None = None,
        settings: Settings | None = None,
    ) -> None:
        """Initialize the paper repository.

        Args:
            db_manager: Database manager instance
            settings: Application settings
        """
        self._settings = settings or get_settings()
        self._db = db_manager or DatabaseManager(self._settings)

    def insert(self, paper: ArxivPaper) -> bool:
        """Insert a single paper into the database.

        Args:
            paper: ArxivPaper to insert

        Returns:
            True if inserted, False if already exists
        """
        with self._db.get_connection() as conn:
            try:
                conn.execute(
                    """
                    INSERT INTO papers (
                        arxiv_id, title, abstract, primary_category,
                        submitted_date, updated_date
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        paper.arxiv_id,
                        paper.title,
                        paper.abstract,
                        paper.primary_category,
                        paper.submitted_date.isoformat(),
                        paper.updated_date.isoformat() if paper.updated_date else None,
                    ),
                )

                for category in paper.categories:
                    conn.execute(
                        "INSERT OR IGNORE INTO paper_categories (arxiv_id, category) VALUES (?, ?)",
                        (paper.arxiv_id, category),
                    )

                for i, author in enumerate(paper.authors):
                    sql = """
                        INSERT OR IGNORE INTO paper_authors
                        (arxiv_id, author_name, author_order) VALUES (?, ?, ?)
                    """
                    conn.execute(sql, (paper.arxiv_id, author, i))

                conn.commit()
                return True

            except Exception as e:
                if "UNIQUE constraint failed" in str(e):
                    return False
                raise

    def batch_insert(self, papers: list[ArxivPaper]) -> int:
        """Insert multiple papers efficiently.

        Args:
            papers: List of papers to insert

        Returns:
            Number of papers successfully inserted
        """
        inserted = 0

        with self._db.get_connection() as conn:
            for paper in papers:
                try:
                    conn.execute(
                        """
                        INSERT OR IGNORE INTO papers (
                            arxiv_id, title, abstract, primary_category,
                            submitted_date, updated_date
                        ) VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (
                            paper.arxiv_id,
                            paper.title,
                            paper.abstract,
                            paper.primary_category,
                            paper.submitted_date.isoformat(),
                            paper.updated_date.isoformat() if paper.updated_date else None,
                        ),
                    )

                    if conn.total_changes > 0:
                        inserted += 1

                        for category in paper.categories:
                            cat_sql = """
                                INSERT OR IGNORE INTO paper_categories
                                (arxiv_id, category) VALUES (?, ?)
                            """
                            conn.execute(cat_sql, (paper.arxiv_id, category))

                        for i, author in enumerate(paper.authors[:10]):
                            author_sql = """
                                INSERT OR IGNORE INTO paper_authors
                                (arxiv_id, author_name, author_order) VALUES (?, ?, ?)
                            """
                            conn.execute(author_sql, (paper.arxiv_id, author, i))

                except Exception as e:
                    logger.warning(f"Failed to insert paper {paper.arxiv_id}: {e}")
                    continue

            conn.commit()

        logger.info(f"Batch inserted {inserted}/{len(papers)} papers")
        return inserted

    def get_by_id(self, arxiv_id: str) -> ArxivPaper | None:
        """Get a paper by its arXiv ID.

        Args:
            arxiv_id: arXiv identifier

        Returns:
            ArxivPaper if found, None otherwise
        """
        with self._db.get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM papers WHERE arxiv_id = ?",
                (arxiv_id,),
            ).fetchone()

            if not row:
                return None

            categories = [
                r[0]
                for r in conn.execute(
                    "SELECT category FROM paper_categories WHERE arxiv_id = ?",
                    (arxiv_id,),
                ).fetchall()
            ]

            author_sql = """
                SELECT author_name FROM paper_authors
                WHERE arxiv_id = ? ORDER BY author_order
            """
            authors = [r[0] for r in conn.execute(author_sql, (arxiv_id,)).fetchall()]

            submitted = row["submitted_date"]
            if isinstance(submitted, str):
                submitted = date.fromisoformat(submitted)

            updated = row["updated_date"]
            if updated and isinstance(updated, str):
                updated = date.fromisoformat(updated)

            return ArxivPaper(
                arxiv_id=row["arxiv_id"],
                title=row["title"],
                abstract=row["abstract"] or "",
                primary_category=row["primary_category"],
                categories=categories,
                submitted_date=submitted,
                authors=authors,
                updated_date=updated,
            )

    def get_by_date_range(
        self,
        start_date: date,
        end_date: date,
        categories: list[str] | None = None,
    ) -> pd.DataFrame:
        """Get papers within a date range.

        Args:
            start_date: Start of date range
            end_date: End of date range
            categories: Optional list of categories to filter

        Returns:
            DataFrame with paper data
        """
        query = """
            SELECT p.arxiv_id, p.title, p.primary_category, p.submitted_date
            FROM papers p
            WHERE p.submitted_date BETWEEN ? AND ?
        """
        params: list = [start_date.isoformat(), end_date.isoformat()]

        if categories:
            placeholders = ",".join("?" * len(categories))
            query += f" AND p.primary_category IN ({placeholders})"
            params.extend(categories)

        query += " ORDER BY p.submitted_date"

        with self._db.get_connection() as conn:
            df = pd.read_sql_query(query, conn, params=params)

        if not df.empty:
            df["submitted_date"] = pd.to_datetime(df["submitted_date"])

        return df

    def search_by_keyword(
        self,
        keyword: str,
        start_date: date | None = None,
        end_date: date | None = None,
        limit: int = 10000,
    ) -> pd.DataFrame:
        """Search papers by keyword in title/abstract using FTS.

        Args:
            keyword: Search keyword
            start_date: Optional start date filter
            end_date: Optional end date filter
            limit: Maximum results

        Returns:
            DataFrame with matching papers
        """
        query = """
            SELECT p.arxiv_id, p.title, p.primary_category, p.submitted_date
            FROM papers p
            JOIN papers_fts fts ON p.arxiv_id = fts.arxiv_id
            WHERE papers_fts MATCH ?
        """
        params: list = [keyword]

        if start_date:
            query += " AND p.submitted_date >= ?"
            params.append(start_date.isoformat())

        if end_date:
            query += " AND p.submitted_date <= ?"
            params.append(end_date.isoformat())

        query += f" ORDER BY p.submitted_date LIMIT {limit}"

        with self._db.get_connection() as conn:
            try:
                df = pd.read_sql_query(query, conn, params=params)
            except Exception as e:
                logger.warning(f"FTS search failed: {e}")
                return pd.DataFrame()

        if not df.empty:
            df["submitted_date"] = pd.to_datetime(df["submitted_date"])

        return df

    def get_category_counts_by_month(
        self,
        categories: list[str],
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame:
        """Get monthly paper counts per category.

        Args:
            categories: List of categories to include
            start_date: Start of date range
            end_date: End of date range

        Returns:
            DataFrame with columns: date, category, count
        """
        placeholders = ",".join("?" * len(categories))
        query = f"""
            SELECT
                strftime('%Y-%m', submitted_date) as month,
                primary_category as category,
                COUNT(*) as count
            FROM papers
            WHERE primary_category IN ({placeholders})
              AND submitted_date BETWEEN ? AND ?
            GROUP BY month, category
            ORDER BY month, category
        """
        params = [*categories, start_date.isoformat(), end_date.isoformat()]

        with self._db.get_connection() as conn:
            df = pd.read_sql_query(query, conn, params=params)

        if not df.empty:
            df["month"] = pd.to_datetime(df["month"] + "-01")

        return df

    def get_total_count(self) -> int:
        """Get total number of papers in the database.

        Returns:
            Total paper count
        """
        with self._db.get_connection() as conn:
            result = conn.execute("SELECT COUNT(*) FROM papers").fetchone()
            return result[0] if result else 0

    def get_date_range(self) -> tuple[date | None, date | None]:
        """Get the date range of papers in the database.

        Returns:
            Tuple of (earliest_date, latest_date) or (None, None) if empty
        """
        with self._db.get_connection() as conn:
            result = conn.execute(
                "SELECT MIN(submitted_date), MAX(submitted_date) FROM papers"
            ).fetchone()

            if result and result[0] and result[1]:
                return (
                    date.fromisoformat(result[0]),
                    date.fromisoformat(result[1]),
                )

            return None, None

    def save_aggregation_cache(
        self,
        cache_key: str,
        data: pd.DataFrame,
        ttl_hours: int | None = None,
    ) -> None:
        """Save aggregation results to cache.

        Args:
            cache_key: Unique cache key
            data: DataFrame to cache
            ttl_hours: Cache TTL in hours
        """
        ttl = ttl_hours or self._settings.cache_ttl_hours
        expires_at = datetime.now() + timedelta(hours=ttl)

        with self._db.get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO aggregation_cache
                (cache_key, data_json, expires_at)
                VALUES (?, ?, ?)
                """,
                (cache_key, data.to_json(), expires_at.isoformat()),
            )
            conn.commit()

    def get_aggregation_cache(self, cache_key: str) -> pd.DataFrame | None:
        """Get cached aggregation results.

        Args:
            cache_key: Cache key to lookup

        Returns:
            Cached DataFrame or None if not found/expired
        """
        with self._db.get_connection() as conn:
            result = conn.execute(
                """
                SELECT data_json FROM aggregation_cache
                WHERE cache_key = ? AND expires_at > datetime('now')
                """,
                (cache_key,),
            ).fetchone()

            if result:
                return pd.read_json(result[0])

            return None

    def log_harvest(
        self,
        category: str,
        harvest_type: str,
        start_date: date,
        end_date: date,
        records_fetched: int,
    ) -> None:
        """Log a data harvest operation.

        Args:
            category: Category harvested
            harvest_type: Type of harvest ('full' or 'incremental')
            start_date: Start of harvest range
            end_date: End of harvest range
            records_fetched: Number of records fetched
        """
        with self._db.get_connection() as conn:
            conn.execute(
                """
                INSERT INTO harvest_log
                (category, harvest_type, start_date, end_date, records_fetched)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    category,
                    harvest_type,
                    start_date.isoformat(),
                    end_date.isoformat(),
                    records_fetched,
                ),
            )
            conn.commit()

    def get_last_harvest_date(self, category: str) -> date | None:
        """Get the end date of the last harvest for a category.

        Args:
            category: Category to check

        Returns:
            Last harvest end date or None
        """
        with self._db.get_connection() as conn:
            result = conn.execute(
                """
                SELECT MAX(end_date) FROM harvest_log
                WHERE category = ?
                """,
                (category,),
            ).fetchone()

            if result and result[0]:
                return date.fromisoformat(result[0])

            return None

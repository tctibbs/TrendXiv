"""arXiv API client with rate limiting."""

import logging
import time
from collections.abc import Generator
from datetime import date

import arxiv

from src.api.models import ArxivPaper, SearchQuery
from src.common.config import Settings, get_settings

logger = logging.getLogger(__name__)


class ArxivClient:
    """Client for interacting with the arXiv API with built-in rate limiting."""

    def __init__(self, settings: Settings | None = None) -> None:
        """Initialize the arXiv client.

        Args:
            settings: Application settings. Uses defaults if not provided.
        """
        self._settings = settings or get_settings()
        self._client = arxiv.Client(
            page_size=self._settings.arxiv_page_size,
            delay_seconds=self._settings.arxiv_delay_seconds,
            num_retries=3,
        )
        self._last_request_time = 0.0

    def _enforce_rate_limit(self) -> None:
        """Ensure minimum delay between requests per arXiv ToU."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self._settings.arxiv_delay_seconds:
            sleep_time = self._settings.arxiv_delay_seconds - elapsed
            logger.debug(f"Rate limiting: sleeping {sleep_time:.2f}s")
            time.sleep(sleep_time)
        self._last_request_time = time.time()

    def search_by_category(
        self,
        category: str,
        start_date: date | None = None,
        end_date: date | None = None,
        max_results: int | None = None,
    ) -> Generator[ArxivPaper, None, None]:
        """Search for papers by category.

        Args:
            category: arXiv category code (e.g., 'cs.LG')
            start_date: Optional start date filter
            end_date: Optional end date filter
            max_results: Maximum number of results to return

        Yields:
            ArxivPaper instances matching the search criteria
        """
        query = SearchQuery(
            categories=[category],
            start_date=start_date,
            end_date=end_date,
        )
        yield from self._execute_search(query, max_results)

    def search_by_categories(
        self,
        categories: list[str],
        start_date: date | None = None,
        end_date: date | None = None,
        max_results: int | None = None,
    ) -> Generator[ArxivPaper, None, None]:
        """Search for papers across multiple categories.

        Args:
            categories: List of arXiv category codes
            start_date: Optional start date filter
            end_date: Optional end date filter
            max_results: Maximum number of results to return

        Yields:
            ArxivPaper instances matching the search criteria
        """
        query = SearchQuery(
            categories=categories,
            start_date=start_date,
            end_date=end_date,
        )
        yield from self._execute_search(query, max_results)

    def search_by_keyword(
        self,
        keyword: str,
        categories: list[str] | None = None,
        start_date: date | None = None,
        end_date: date | None = None,
        max_results: int | None = None,
    ) -> Generator[ArxivPaper, None, None]:
        """Search for papers containing a keyword in the abstract.

        Args:
            keyword: Keyword to search for in abstracts
            categories: Optional list of categories to filter
            start_date: Optional start date filter
            end_date: Optional end date filter
            max_results: Maximum number of results to return

        Yields:
            ArxivPaper instances matching the search criteria
        """
        query = SearchQuery(
            categories=categories or [],
            keywords=[keyword],
            start_date=start_date,
            end_date=end_date,
        )
        yield from self._execute_search(query, max_results)

    def _execute_search(
        self,
        query: SearchQuery,
        max_results: int | None = None,
    ) -> Generator[ArxivPaper, None, None]:
        """Execute a search query against the arXiv API.

        Args:
            query: SearchQuery object defining the search parameters
            max_results: Maximum number of results to return

        Yields:
            ArxivPaper instances from the search results
        """
        self._enforce_rate_limit()

        query_str = query.to_arxiv_query()
        limit = max_results or self._settings.arxiv_max_results

        logger.info(f"Executing arXiv search: {query_str} (max: {limit})")

        search = arxiv.Search(
            query=query_str,
            max_results=limit,
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending,
        )

        count = 0
        for result in self._client.results(search):
            paper = ArxivPaper.from_arxiv_result(result)

            if query.start_date and paper.submitted_date < query.start_date:
                continue
            if query.end_date and paper.submitted_date > query.end_date:
                continue

            yield paper
            count += 1

            if count >= limit:
                break

        logger.info(f"Search returned {count} papers")

    def get_paper_count_estimate(self, category: str) -> int:
        """Get an estimate of total papers in a category.

        Note: This is an approximation based on a quick search.

        Args:
            category: arXiv category code

        Returns:
            Estimated number of papers in the category
        """
        self._enforce_rate_limit()

        search = arxiv.Search(
            query=f"cat:{category}",
            max_results=1,
        )

        try:
            results = list(self._client.results(search))
            return len(results) * 1000
        except Exception as e:
            logger.warning(f"Could not estimate count for {category}: {e}")
            return 0

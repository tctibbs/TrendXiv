"""Build the category x month cube from the ingested working set.

Three attribution modes ship side by side because each answers a different
question and no single one is defensible for every view:

``any``      the paper lists this category anywhere. Matches arXiv's own
             ``cat:`` query semantics, so a curious reader can verify a number
             against arxiv.org. This is the default.
``primary``  the paper's first-listed category only. Undercounts
             interdisciplinary work.
``frac``     1/k of a paper to each of its k categories. The only mode whose
             shares sum to exactly 1.0, so it backs the stacked landscape.

The size difference between shipping one mode and all three is a few tens of
kilobytes, so the choice is exposed rather than hidden.
"""

from __future__ import annotations

import logging

import duckdb

from src.common.taxonomy import canonical
from src.pipeline.periods import AXIS_START, month_range

logger = logging.getLogger(__name__)

#: Categories below this lifetime paper count are dropped from the cube. They
#: are almost all typos and defunct codes that would otherwise clutter the
#: picker with permanently flat lines.
MIN_CATEGORY_PAPERS = 200


def build_cube(con: duckdb.DuckDBPyConnection, periods: list[str]) -> dict:
    """Aggregate ``cat_month`` into dense per-category arrays.

    Alias resolution happens here rather than at ingest so that a change to the
    alias table does not require a re-ingest.

    Args:
        con: Connection to the ingested working set.
        periods: The shared period axis.

    Returns:
        Mapping of category code to ``{"any": [...], "primary": [...], "frac": [...]}``.
    """
    rows = con.execute("""
        SELECT period, category, sum(n_any), sum(n_primary), sum(n_frac)
        FROM cat_month GROUP BY 1, 2
    """).fetchall()

    index = {period: i for i, period in enumerate(periods)}
    size = len(periods)
    series: dict[str, dict[str, list]] = {}
    totals: dict[str, float] = {}

    for period, category, n_any, n_primary, n_frac in rows:
        slot = index.get(period)
        if slot is None:
            continue  # Backdated pre-1991 submissions: real metadata, not arXiv activity.
        code = canonical(category)
        entry = series.get(code)
        if entry is None:
            entry = {"any": [0] * size, "primary": [0] * size, "frac": [0.0] * size}
            series[code] = entry
        entry["any"][slot] += int(n_any)
        entry["primary"][slot] += int(n_primary)
        entry["frac"][slot] += float(n_frac)
        totals[code] = totals.get(code, 0) + int(n_any)

    dropped = [c for c, total in totals.items() if total < MIN_CATEGORY_PAPERS]
    for code in dropped:
        del series[code]
    logger.info("cube: %d categories kept, %d dropped below %d papers",
                len(series), len(dropped), MIN_CATEGORY_PAPERS)

    for entry in series.values():
        entry["frac"] = [round(value, 2) for value in entry["frac"]]
    return series


def build_totals(con: duckdb.DuckDBPyConnection, periods: list[str]) -> dict:
    """Build the denominator series.

    The denominator comes from this same snapshot, never from arXiv's published
    monthly CSV. Mixing sources puts a numerator and a denominator that disagree
    by up to 2.6% in a given month into the same ratio, and that disagreement is
    non-stationary, so it reads as signal. The published CSV is used only as a
    build-time integrity check.

    Args:
        con: Connection to the ingested working set.
        periods: The shared period axis.

    Returns:
        Mapping with ``papers`` (distinct papers per month) and ``listings``
        (sum of category listings, the denominator for ``any``-mode shares).
    """
    index = {period: i for i, period in enumerate(periods)}
    papers = [0] * len(periods)
    for period, n in con.execute("SELECT period, sum(n) FROM corpus_month GROUP BY 1").fetchall():
        if (slot := index.get(period)) is not None:
            papers[slot] += int(n)

    listings = [0] * len(periods)
    for period, n in con.execute("SELECT period, sum(n_any) FROM cat_month GROUP BY 1").fetchall():
        if (slot := index.get(period)) is not None:
            listings[slot] += int(n)

    return {"papers": papers, "listings": listings}


def axis_for(con: duckdb.DuckDBPyConnection) -> tuple[list[str], str]:
    """Return the shared period axis and the snapshot's latest submission month."""
    max_period = con.execute(
        "SELECT max(period) FROM corpus_month WHERE period <= strftime(current_date, '%Y-%m')"
    ).fetchone()[0]
    return month_range(AXIS_START, max_period), max_period

"""Build the term x month index that powers arbitrary keyword search.

Counts are DOCUMENT frequency -- papers mentioning a term at least once --
never term occurrences. Two reasons: Google-Trends semantics are "share of
papers about X", and abstracts have grown steadily longer over 35 years, so an
occurrence count bakes that drift into every series as a spurious upward trend.

The index ships as plain JSON sharded by first character. A shard is fetched
with one ordinary ``fetch()`` and parsed by the browser's own JSON parser: no
WASM engine, no CDN dependency, no mobile failure mode. Loading a 7 MB query
engine to read a 12 MB index would cost more than downloading the index whole.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import duckdb

logger = logging.getLogger(__name__)

#: Terms appearing in fewer papers than this are not indexed. They are still
#: recorded in the vocabulary so the UI can distinguish "below the indexing
#: threshold" from "genuinely zero papers" -- silently returning zeros for an
#: unindexed term is actively misleading.
MIN_DOCUMENT_FREQUENCY = 10

#: Function words, paper boilerplate, and URL fragments. Excluded from
#: *discovery* only -- a reader may still search any of these explicitly.
#:
#: The corpus-share ceiling and the field-specificity filter catch most
#: boilerplate automatically, but neither catches everything: "github" and
#: "outperforms" are genuinely computer-science-specific, so they survive a
#: specificity test while saying nothing about research topics. A curated list
#: is the honest complement to the measured filters, not a substitute for them.
STOPWORDS = frozenset("""
the and for that with this are from which these our can was were has have had
not but all any its their they them then than there here when where while who
whom whose how why what into onto upon over under about above below between
been being both each either neither more most much many some such only other
another same different new novel using used use uses very well also however
thus hence therefore moreover furthermore additionally respectively via per
based given shown show shows showed present presents presented propose proposed
proposes paper study studies studied result results method methods approach
approaches model models data set sets case cases time times number numbers
first second third one two three four five six seven eight nine ten
we it he she his her you your they i a an of in on to as at by is be do does
did done will would should could may might must shall can cannot if or nor so
because although though since until unless whether within without across
during before after again further once here there all any both few more most
other some such no nor not only own same than too very just now
github https http www com org arxiv doi url href
outperforms outperform outperforming baselines baseline sota
diverse capabilities scalable success comprehensive enabling enhance enhances
enhancing improving improves introduces achieving leveraging leverages
integrates offering rely relies traditional crucial particularly extensive
significantly demonstrate demonstrates validate validates effectiveness
efficient efficiency robust robustness adaptive automated challenging
""".split())

_TOKEN = re.compile(r"^[a-z][a-z0-9+#_-]{2,29}$")


def shard_key(term: str) -> str:
    """Return the shard a term belongs to.

    Args:
        term: Lowercased term.

    Returns:
        Single-character shard key: ``a``-``z``, ``0`` for a digit, ``_`` otherwise.
    """
    head = term[0]
    if "a" <= head <= "z":
        return head
    return "0" if head.isdigit() else "_"


def _trim(vector: list[int]) -> tuple[int, list[int]]:
    """Drop leading and trailing zeros, returning the offset and remainder."""
    start = 0
    end = len(vector)
    while start < end and vector[start] == 0:
        start += 1
    while end > start and vector[end - 1] == 0:
        end -= 1
    return start, vector[start:end]


def build_term_index(
    con: duckdb.DuckDBPyConnection,
    periods: list[str],
    out_dir: Path,
    min_df: int = MIN_DOCUMENT_FREQUENCY,
) -> dict:
    """Write sharded term files and return the vocabulary artifact.

    Args:
        con: Connection to the ingested working set.
        periods: The shared period axis.
        out_dir: Directory to write ``terms/t-<shard>.json`` into.
        min_df: Lifetime document-frequency floor for indexing.

    Returns:
        Vocabulary mapping suitable for serialisation as ``vocab.json``.
    """
    index = {period: i for i, period in enumerate(periods)}
    size = len(periods)

    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE kept AS
        SELECT term, sum(df) AS total
        FROM term_month GROUP BY 1 HAVING sum(df) >= {min_df};
    """)
    kept = con.execute("SELECT count(*) FROM kept").fetchone()[0]
    below = con.execute(
        f"SELECT count(*) FROM (SELECT term FROM term_month GROUP BY 1 HAVING sum(df) < {min_df})"
    ).fetchone()[0]
    logger.info("terms: %d indexed at df>=%d, %d below threshold", kept, min_df, below)

    shards: dict[str, dict] = {}
    vocab: dict[str, dict] = {}
    vectors: dict[str, list[int]] = {}
    current: str | None = None

    rows = con.execute("""
        SELECT tm.term, tm.period, tm.df
        FROM term_month tm JOIN kept k USING (term)
        ORDER BY tm.term
    """).fetchall()

    def flush(term: str) -> None:
        offset, values = _trim(vectors.pop(term))
        if not values:
            return
        key = shard_key(term)
        shards.setdefault(key, {})[term] = {"o": offset, "v": values}
        # Only the document frequency is stored. The shard key is a pure function
        # of the first character, so shipping it would be ~190k redundant strings
        # in the one artifact a searching user waits on.
        vocab[term] = sum(values)

    for term, period, df in rows:
        if term != current:
            if current is not None:
                flush(current)
            current = term
            vectors[term] = [0] * size
        if (slot := index.get(period)) is not None:
            vectors[term][slot] += int(df)
    if current is not None:
        flush(current)

    terms_dir = out_dir / "terms"
    terms_dir.mkdir(parents=True, exist_ok=True)
    for key, payload in shards.items():
        (terms_dir / f"t-{key}.json").write_text(
            json.dumps(payload, separators=(",", ":"), sort_keys=True), encoding="utf-8"
        )
    logger.info("terms: wrote %d shards", len(shards))

    return {"min_df": min_df, "below_threshold_count": below, "terms": vocab}


#: A term used in more than this fraction of every paper arXiv has ever
#: published, across every field and 35 years, is doing rhetorical work rather
#: than naming a topic. The corpus itself draws the line cleanly: "transformer"
#: sits at 0.9%, "diffusion" at 2.2% and "attention" at 2.8%, while "extensive"
#: is 3.8%, "discuss" 5.6% and "performance" 12.0%. A hand-maintained stoplist
#: cannot keep up with academic boilerplate; this is measured, not guessed.
MAX_DOCUMENT_FRACTION = 0.03


def discovery_candidates(
    vocab: dict,
    min_df: int = 200,
    corpus_size: int | None = None,
    max_fraction: float = MAX_DOCUMENT_FRACTION,
) -> list[str]:
    """Select terms eligible for automated discovery, bursts and lifecycle fits.

    This filters only *discovery*. A reader may still search any indexed term
    directly, including the ones excluded here.

    Args:
        vocab: The vocabulary artifact from :func:`build_term_index`.
        min_df: Lifetime document-frequency floor for a candidate.
        corpus_size: Total papers, used to apply ``max_fraction``. Omitting it
            disables the upper bound.
        max_fraction: Upper bound on a candidate's share of the whole corpus.

    Returns:
        Terms that are neither boilerplate nor too rare to score.
    """
    ceiling = corpus_size * max_fraction if corpus_size else None
    return [
        term
        for term, df in vocab["terms"].items()
        if df >= min_df
        and (ceiling is None or df <= ceiling)
        and term not in STOPWORDS
        and _TOKEN.match(term)
    ]

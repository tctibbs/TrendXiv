"""Stream the arXiv metadata snapshot into a local DuckDB working set.

The snapshot is ~3 GB across ten Parquet shards. We never hold more than one
shard on disk: each is downloaded, aggregated into the two facts we actually
need, and deleted. Everything downstream reads the local database, so the
network is touched exactly once per build.

Two aggregates are produced:

``cat_month``   (period, category, kind)      -> paper counts per category
``term_month``  (period, term)                -> *document* frequency, i.e. the
                number of papers mentioning a term at least once. Never term
                occurrences: abstracts have grown longer over 35 years, so
                occurrence counts bake that drift into every series.
"""

from __future__ import annotations

import logging
import subprocess
import time
from pathlib import Path

import duckdb

from src.common.taxonomy import CATEGORY_ALIASES

logger = logging.getLogger(__name__)

REPO = "librarian-bots/arxiv-metadata-snapshot"
REVISION = "47141d6fd17f52b65424d246665334914cac3011"
N_SHARDS = 10
SHARD_URL = (
    "https://huggingface.co/api/datasets/{repo}/parquet/default/train/{i}.parquet"
)

#: v1 timestamp format used throughout the snapshot, e.g.
#: "Wed, 31 Oct 2012 09:13:21 GMT". Verified to parse 3,148,882 / 3,148,882 rows.
V1_FORMAT = "%a, %-d %b %Y %H:%M:%S GMT"

#: Tokens are lowercased words of 3-30 chars. Leading char must be alphabetic so
#: that stripped LaTeX residue and bare numbers do not enter the vocabulary.
TOKEN_RE = r"[a-z][a-z0-9+#_-]{2,29}"

#: Applied before tokenising. Inline math and control sequences otherwise inject
#: \alpha, mathbb, frac and friends into the term index.
LATEX_STRIP = [
    (r"\$[^$]*\$", " "),
    (r"\\[a-zA-Z]+\s*", " "),
    (r"[{}\\^_~]", " "),
]


def _sql_strip_latex(col: str) -> str:
    expr = f"lower({col})"
    for pattern, repl in LATEX_STRIP:
        expr = f"regexp_replace({expr}, '{pattern}', '{repl}', 'g')"
    return expr


def _alias_map_sql() -> str:
    """Render the alias table as a DuckDB MAP literal.

    arXiv lists BOTH members of an alias pair on the same paper: a paper in
    systems and control carries ``cs.SY eess.SY``, which is why the two codes
    return near-identical lifetime counts. Canonicalising without deduplicating
    therefore counts every such paper twice. Both steps happen here, per paper,
    so no downstream aggregate can reintroduce the error.
    """
    entries = ", ".join(
        f"'{alias}': '{target}'" for alias, target in CATEGORY_ALIASES.items()
    )
    return "MAP {" + entries + "}"


def _is_valid_parquet(path: Path) -> bool:
    """Check that a downloaded file is a complete Parquet file.

    A size threshold is not enough: an interrupted transfer leaves a large,
    plausible-looking, unreadable file that would then be silently ingested as
    a partial shard. Parquet ends with a 4-byte "PAR1" magic, so a truncated
    download fails this check.
    """
    if not path.exists() or path.stat().st_size < 1_000_000:
        return False
    with path.open("rb") as handle:
        handle.seek(-4, 2)
        if handle.read(4) != b"PAR1":
            return False
    try:
        duckdb.connect().execute(
            f"SELECT count(*) FROM read_parquet('{path}') LIMIT 1"
        ).fetchone()
    except duckdb.Error:
        return False
    return True


def download_shard(index: int, dest: Path, retries: int = 6) -> Path:
    """Download one Parquet shard, backing off on HTTP 429.

    Hugging Face rate-limits anonymous bulk reads aggressively; a plain
    ``read_parquet`` across all shards fails partway through a build.
    """
    url = SHARD_URL.format(repo=REPO, i=index)
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(retries):
        result = subprocess.run(
            ["curl", "-sSL", "--fail", "--max-time", "1800", "-o", str(dest), url],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0 and _is_valid_parquet(dest):
            logger.info("shard %d downloaded (%.0f MB)", index, dest.stat().st_size / 1e6)
            return dest
        dest.unlink(missing_ok=True)
        wait = min(120, 5 * 2**attempt)
        logger.warning(
            "shard %d attempt %d failed (%s); retrying in %ds",
            index, attempt + 1, result.stderr.strip()[:120], wait,
        )
        time.sleep(wait)
    raise RuntimeError(f"could not download shard {index} after {retries} attempts")


def init_db(con: duckdb.DuckDBPyConnection) -> None:
    """Create the accumulator tables."""
    con.execute("""
        CREATE TABLE IF NOT EXISTS cat_month (
            period   VARCHAR NOT NULL,
            category VARCHAR NOT NULL,
            n_any    BIGINT  NOT NULL,
            n_primary BIGINT NOT NULL,
            n_frac   DOUBLE  NOT NULL
        );
        CREATE TABLE IF NOT EXISTS term_month (
            period VARCHAR NOT NULL,
            term   VARCHAR NOT NULL,
            df     BIGINT  NOT NULL
        );
        CREATE TABLE IF NOT EXISTS term_group (
            term  VARCHAR NOT NULL,
            grp   VARCHAR NOT NULL,
            df    BIGINT  NOT NULL
        );
        CREATE TABLE IF NOT EXISTS corpus_month (
            period VARCHAR NOT NULL,
            n      BIGINT  NOT NULL
        );
        CREATE TABLE IF NOT EXISTS shard_log (
            shard INTEGER PRIMARY KEY,
            rows  BIGINT
        );
    """)


def ingest_shard(con: duckdb.DuckDBPyConnection, path: Path, shard: int) -> int:
    """Aggregate one shard into the accumulator tables and return its row count."""
    alias = _alias_map_sql()
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE slim AS
        SELECT
            id,
            strftime(strptime(versions[1].created, '{V1_FORMAT}'), '%Y-%m') AS period,
            list_distinct(list_transform(
                str_split(trim(categories), ' '),
                c -> coalesce({alias}[c], c)
            )) AS cats,
            coalesce({alias}[str_split(trim(categories), ' ')[1]],
                     str_split(trim(categories), ' ')[1]) AS primary_cat,
            {_sql_strip_latex("title || ' ' || abstract")} AS text
        FROM read_parquet('{path}')
        WHERE versions[1].created IS NOT NULL;
    """)
    rows = con.execute("SELECT count(*) FROM slim").fetchone()[0]

    con.execute("""
        INSERT INTO corpus_month
        SELECT period, count(*) FROM slim GROUP BY 1;
    """)

    # `any` counts a paper in every category it lists (matching arXiv `cat:`
    # semantics, and therefore externally checkable). `primary` counts only the
    # first-listed category. `frac` splits one paper's mass across its k
    # categories so shares sum to exactly 1.0 for the stacked landscape.
    con.execute("""
        INSERT INTO cat_month
        SELECT
            period,
            cat,
            count(*)                                    AS n_any,
            count(*) FILTER (WHERE cat = primary_cat)   AS n_primary,
            sum(1.0 / k)                                AS n_frac
        FROM (
            SELECT period, primary_cat, unnest(cats) AS cat, len(cats) AS k
            FROM slim
        )
        WHERE cat <> ''
        GROUP BY 1, 2;
    """)

    # list_distinct BEFORE unnest: the naive global DISTINCT over (term, doc)
    # pairs materialises hundreds of millions of rows and exhausts disk.
    con.execute(f"""
        INSERT INTO term_month
        SELECT period, term, count(*) AS df
        FROM (
            SELECT period,
                   unnest(list_distinct(regexp_extract_all(text, '{TOKEN_RE}'))) AS term
            FROM slim
        )
        GROUP BY 1, 2;
    """)

    # Term usage by top-level archive group. This is what separates a topic from
    # academic prose: "graphene" lives almost entirely in cond-mat, while
    # "enabling" is spread evenly across every field. Frequency alone cannot tell
    # them apart -- both surged after 2023 -- but concentration can. Grouped at
    # the archive level rather than the category level to keep this aggregate two
    # orders of magnitude smaller than term x category would be.
    con.execute(f"""
        INSERT INTO term_group
        SELECT term, grp, count(*) AS df
        FROM (
            SELECT
                CASE
                    WHEN primary_cat LIKE 'hep-%'  THEN 'hep'
                    WHEN primary_cat LIKE 'nucl-%' THEN 'nucl'
                    WHEN primary_cat LIKE 'astro-ph%' THEN 'astro-ph'
                    WHEN contains(primary_cat, '.') THEN split_part(primary_cat, '.', 1)
                    ELSE primary_cat
                END AS grp,
                unnest(list_distinct(regexp_extract_all(text, '{TOKEN_RE}'))) AS term
            FROM slim
        )
        GROUP BY 1, 2;
    """)

    con.execute("INSERT OR REPLACE INTO shard_log VALUES (?, ?)", [shard, rows])
    con.execute("DROP TABLE slim")
    return rows


def ingest_all(db_path: Path, cache_dir: Path, keep_shards: bool = False) -> int:
    """Run the full ingest, resuming from whatever shards are already logged."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path))
    con.execute("SET enable_progress_bar=false; SET preserve_insertion_order=false;")
    init_db(con)
    done = {r[0] for r in con.execute("SELECT shard FROM shard_log").fetchall()}

    total = 0
    for i in range(N_SHARDS):
        if i in done:
            logger.info("shard %d already ingested, skipping", i)
            continue
        shard_path = cache_dir / f"{i}.parquet"
        # Validate on resume, not only after downloading: a run interrupted
        # mid-transfer leaves a large, plausible-looking, truncated file, and
        # trusting it here would ingest a partial shard as if it were complete.
        if not _is_valid_parquet(shard_path):
            if shard_path.exists():
                logger.warning("shard %d on disk is incomplete; re-downloading", i)
                shard_path.unlink()
            download_shard(i, shard_path)
        started = time.time()
        rows = ingest_shard(con, shard_path, i)
        total += rows
        logger.info("shard %d: %d rows in %.0fs", i, rows, time.time() - started)
        if not keep_shards:
            shard_path.unlink(missing_ok=True)

    con.close()
    return total

"""Tests for the ingest's per-paper category canonicalisation.

The bug these guard against is silent and severe: arXiv lists both members of
an alias pair on the same paper (a systems-and-control paper carries
``cs.SY eess.SY``), so canonicalising without deduplicating counts that paper
twice in the merged category and inflates the denominator it feeds.
"""

import duckdb
import pytest

from src.common.taxonomy import CATEGORY_ALIASES
from src.pipeline.ingest import _alias_map_sql


def _canonicalise(categories: str) -> tuple[list[str], str]:
    """Run one category string through the ingest's canonicalisation SQL."""
    alias = _alias_map_sql()
    query = f"""
        SELECT
            list_distinct(list_transform(
                str_split(trim(?), ' '), c -> coalesce({alias}[c], c)
            )) AS cats,
            coalesce({alias}[str_split(trim(?), ' ')[1]],
                     str_split(trim(?), ' ')[1]) AS primary_cat
    """
    row = duckdb.connect().execute(query, [categories, categories, categories]).fetchone()
    return list(row[0]), row[1]


@pytest.mark.unit
def test_alias_pair_on_one_paper_collapses_to_a_single_category():
    """cs.SY and eess.SY on the same paper must count once, not twice."""
    cats, _ = _canonicalise("cs.SY eess.SY math.OC")
    assert sorted(cats) == ["eess.SY", "math.OC"]
    assert len(cats) == 2


@pytest.mark.unit
@pytest.mark.parametrize("pair", [
    ("cs.SY", "eess.SY"), ("stat.TH", "math.ST"), ("q-fin.EC", "econ.GN"),
    ("cs.NA", "math.NA"), ("cs.IT", "math.IT"), ("math.MP", "math-ph"),
])
def test_every_alias_pair_collapses(pair):
    cats, _ = _canonicalise(f"{pair[0]} {pair[1]}")
    assert cats == [pair[1]]


@pytest.mark.unit
def test_primary_category_is_canonicalised():
    """The primary is read positionally BEFORE dedup, so it must be mapped too."""
    _, primary = _canonicalise("cs.SY math.OC")
    assert primary == "eess.SY"


@pytest.mark.unit
def test_primary_survives_reordering_by_list_distinct():
    """list_distinct does not preserve order, which is why primary is its own column."""
    cats, primary = _canonicalise("cs.SY eess.SY math.OC")
    assert primary == "eess.SY"
    assert primary in cats


@pytest.mark.unit
def test_legacy_rename_is_remapped():
    cats, primary = _canonicalise("cmp-lg")
    assert cats == ["cs.CL"]
    assert primary == "cs.CL"


@pytest.mark.unit
def test_non_alias_categories_are_untouched():
    cats, primary = _canonicalise("cs.LG stat.ML cs.AI")
    assert sorted(cats) == ["cs.AI", "cs.LG", "stat.ML"]
    assert primary == "cs.LG"


@pytest.mark.unit
def test_alias_map_covers_every_declared_alias():
    """A code missing from the SQL map would silently keep double-counting."""
    alias_sql = _alias_map_sql()
    for code in CATEGORY_ALIASES:
        assert f"'{code}'" in alias_sql


@pytest.mark.unit
def test_truncated_parquet_is_rejected(tmp_path):
    """A run killed mid-download leaves a big, plausible, unreadable file.

    Size alone accepts it; the trailing PAR1 magic does not. Without this the
    resume path ingests a partial shard as though it were complete.
    """
    from src.pipeline.ingest import _is_valid_parquet

    good = tmp_path / "good.parquet"
    duckdb.connect().execute(
        f"COPY (SELECT range AS i FROM range(300000)) TO '{good}' (FORMAT PARQUET)"
    )
    assert _is_valid_parquet(good)

    truncated = tmp_path / "truncated.parquet"
    truncated.write_bytes(good.read_bytes()[:-64])
    assert not _is_valid_parquet(truncated)

    assert not _is_valid_parquet(tmp_path / "missing.parquet")


@pytest.mark.unit
def test_revision_change_resets_the_working_set(tmp_path, monkeypatch):
    """A scheduled rebuild must pick up new papers, not republish the old cube.

    Shards are skipped once logged, so a restored working set combined with an
    unconditional skip would make every weekly build a no-op for ever. The
    revision comparison is what prevents that.
    """
    from src.pipeline import ingest as ingest_module

    db = tmp_path / "w.duckdb"
    con = duckdb.connect(str(db))
    ingest_module.init_db(con)
    con.execute("INSERT INTO shard_log VALUES (0, 100)")
    con.execute("INSERT INTO corpus_month VALUES ('2020-01', 100)")
    con.execute("INSERT INTO build_meta VALUES ('source_revision', 'old-sha')")
    con.close()

    downloaded: list[int] = []
    monkeypatch.setattr(ingest_module, "N_SHARDS", 1)
    monkeypatch.setattr(ingest_module, "upstream_revision", lambda: "new-sha")
    monkeypatch.setattr(
        ingest_module, "download_shard",
        lambda i, dest, revision, **kw: downloaded.append(i) or dest,
    )
    # No shard is cached in tmp_path, so the download path must be exercised.
    monkeypatch.setattr(ingest_module, "_is_valid_parquet", lambda path: False)
    monkeypatch.setattr(ingest_module, "ingest_shard", lambda con, path, shard: 7)

    ingest_module.ingest_all(db, tmp_path)

    con = duckdb.connect(str(db), read_only=True)
    assert downloaded == [0], "a moved upstream revision must trigger re-ingest"
    assert con.execute("SELECT count(*) FROM corpus_month").fetchone()[0] == 0, (
        "stale rows must be cleared, not appended to"
    )
    assert con.execute(
        "SELECT value FROM build_meta WHERE key = 'source_revision'"
    ).fetchone()[0] == "new-sha"


@pytest.mark.unit
def test_unchanged_revision_skips_reingest(tmp_path, monkeypatch):
    """When upstream has not moved, the build is a cheap, correct no-op."""
    from src.pipeline import ingest as ingest_module

    db = tmp_path / "w.duckdb"
    con = duckdb.connect(str(db))
    ingest_module.init_db(con)
    con.execute("INSERT INTO shard_log VALUES (0, 100)")
    con.execute("INSERT INTO corpus_month VALUES ('2020-01', 100)")
    con.execute("INSERT INTO build_meta VALUES ('source_revision', 'same-sha')")
    con.close()

    downloaded: list[int] = []
    monkeypatch.setattr(ingest_module, "N_SHARDS", 1)
    monkeypatch.setattr(ingest_module, "upstream_revision", lambda: "same-sha")
    monkeypatch.setattr(
        ingest_module, "download_shard",
        lambda i, dest, revision, **kw: downloaded.append(i) or dest,
    )

    ingest_module.ingest_all(db, tmp_path)

    con = duckdb.connect(str(db), read_only=True)
    assert downloaded == []
    assert con.execute("SELECT count(*) FROM corpus_month").fetchone()[0] == 1


@pytest.mark.unit
def test_shard_url_pins_every_shard_to_one_revision():
    """A ten-shard ingest spans minutes; an upstream refresh mid-run would
    otherwise splice two snapshots into a single cube."""
    from src.pipeline.ingest import REPO, SHARD_URL

    url = SHARD_URL.format(repo=REPO, revision="abc123", i=7)

    assert "/resolve/abc123/" in url
    assert url.endswith("/0007.parquet"), "shard files are zero-padded to four digits"

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

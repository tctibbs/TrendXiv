"""End-to-end pipeline test over a synthetic working set.

Golden unit tests cover each analysis function in isolation. This covers the
part they cannot: that a cube built from realistic input has correct totals,
correct attribution, and artifacts the frontend can actually read.
"""

from __future__ import annotations

import json

import duckdb
import pytest

from src.pipeline.build import build
from src.pipeline.cube import axis_for, build_cube, build_totals
from src.pipeline.ingest import _alias_map_sql, init_db
from src.pipeline.periods import month_range


@pytest.fixture(autouse=True)
def _keep_small_categories(monkeypatch):
    """The cube drops categories under 200 lifetime papers; fixtures are smaller."""
    monkeypatch.setattr("src.pipeline.cube.MIN_CATEGORY_PAPERS", 1)


@pytest.fixture
def working_set(tmp_path):
    """A three-month working set with a known, hand-checkable composition."""
    db = tmp_path / "test.duckdb"
    con = duckdb.connect(str(db))
    init_db(con)
    alias = _alias_map_sql()

    # Twelve papers across three months. One carries an alias PAIR, which must
    # collapse to a single category rather than being counted twice.
    papers = []
    for month in ("2020-01", "2020-02", "2020-03"):
        papers += [
            (f"{month}-a", month, "cs.LG stat.ML"),
            (f"{month}-b", month, "cs.LG"),
            (f"{month}-c", month, "cs.SY eess.SY"),
            (f"{month}-d", month, "math.AG"),
        ]
    con.execute("CREATE TEMP TABLE raw(id VARCHAR, period VARCHAR, categories VARCHAR)")
    con.executemany("INSERT INTO raw VALUES (?, ?, ?)", papers)
    con.execute(f"""
        CREATE TEMP TABLE slim AS
        SELECT id, period,
          list_distinct(list_transform(str_split(trim(categories), ' '),
                                       c -> coalesce({alias}[c], c))) AS cats,
          coalesce({alias}[str_split(trim(categories), ' ')[1]],
                   str_split(trim(categories), ' ')[1]) AS primary_cat
        FROM raw
    """)
    con.execute("INSERT INTO corpus_month SELECT period, count(*) FROM slim GROUP BY 1")
    con.execute("""
        INSERT INTO cat_month
        SELECT period, cat, count(*), count(*) FILTER (WHERE cat = primary_cat), sum(1.0 / k)
        FROM (SELECT period, primary_cat, unnest(cats) AS cat, len(cats) AS k FROM slim)
        WHERE cat <> '' GROUP BY 1, 2
    """)
    con.execute("INSERT INTO shard_log VALUES (0, ?)", [len(papers)])
    con.close()
    return db


@pytest.mark.integration
def test_alias_pair_is_not_double_counted(working_set):
    con = duckdb.connect(str(working_set), read_only=True)
    periods = month_range("2020-01", "2020-03")
    cube = build_cube(con, periods)
    # cs.SY must not exist at all; eess.SY must hold exactly one paper per month.
    assert "cs.SY" not in cube
    assert cube["eess.SY"]["any"] == [1, 1, 1]
    # And that paper contributes its whole mass to one category, not half to two.
    assert cube["eess.SY"]["frac"] == [1.0, 1.0, 1.0]


@pytest.mark.integration
def test_attribution_modes_are_consistent(working_set):
    con = duckdb.connect(str(working_set), read_only=True)
    periods = month_range("2020-01", "2020-03")
    cube = build_cube(con, periods)
    # cs.LG is listed on two papers a month and is primary on both.
    assert cube["cs.LG"]["any"] == [2, 2, 2]
    assert cube["cs.LG"]["primary"] == [2, 2, 2]
    # One of those papers also lists stat.ML, so cs.LG's fractional mass is 1.5.
    assert cube["cs.LG"]["frac"] == [1.5, 1.5, 1.5]
    assert cube["stat.ML"]["primary"] == [0, 0, 0]


@pytest.mark.integration
def test_fractional_shares_sum_to_the_paper_count(working_set):
    """The defining property of fractional attribution, and the reason it exists."""
    con = duckdb.connect(str(working_set), read_only=True)
    periods = month_range("2020-01", "2020-03")
    cube = build_cube(con, periods)
    totals = build_totals(con, periods)
    for i in range(len(periods)):
        assert sum(entry["frac"][i] for entry in cube.values()) == pytest.approx(
            totals["papers"][i]
        )


@pytest.mark.integration
def test_axis_starts_at_arxiv_launch(working_set):
    con = duckdb.connect(str(working_set), read_only=True)
    periods, _ = axis_for(con)
    assert periods[0] == "1991-07"
    assert periods[-1] == "2020-03"


@pytest.mark.integration
def test_build_emits_a_readable_manifest(working_set, tmp_path, monkeypatch):
    """The artifacts must be self-describing enough for the frontend to boot."""
    out = tmp_path / "build"
    monkeypatch.setattr("src.pipeline.build.EXPECTED_ROWS", 12)
    manifest = build(working_set, out, skip_terms=True, strict=False)

    assert (out / "manifest.json").exists()
    assert manifest["periods"][-1] == "2020-03"
    assert manifest["provisional_from"] == "2020-03"
    assert manifest["data_complete_through"] == "2020-02"

    for name in ("cube", "totals", "taxonomy", "seasonal"):
        assert name in manifest["files"]
        payload = json.loads((out / manifest["files"][name]).read_text())
        assert payload is not None

    cube = json.loads((out / manifest["files"]["cube"]).read_text())
    for entry in cube["series"].values():
        assert len(entry["any"]) == len(manifest["periods"])


@pytest.mark.integration
def test_content_hash_changes_with_content(working_set, tmp_path, monkeypatch):
    """Stable URLs plus browser caching would otherwise mix old and new bytes."""
    monkeypatch.setattr("src.pipeline.build.EXPECTED_ROWS", 12)
    first = build(working_set, tmp_path / "a", skip_terms=True, strict=False)
    second = build(working_set, tmp_path / "b", skip_terms=True, strict=False)
    # Identical input must produce an identical hash...
    assert first["files"]["cube"] == second["files"]["cube"]

    con = duckdb.connect(str(working_set))
    con.execute("INSERT INTO cat_month VALUES ('2020-02', 'cs.AI', 5, 5, 5.0)")
    con.close()
    third = build(working_set, tmp_path / "c", skip_terms=True, strict=False)
    # ...and changed input must not.
    assert third["files"]["cube"] != first["files"]["cube"]

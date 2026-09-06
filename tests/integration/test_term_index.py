"""End-to-end test of the term index build and its on-disk encoding."""

from __future__ import annotations

import json

import duckdb
import pytest

from src.pipeline.ingest import init_db
from src.pipeline.terms import build_term_index


@pytest.fixture
def con(tmp_path):
    connection = duckdb.connect(str(tmp_path / "terms.duckdb"))
    init_db(connection)
    rows = [
        ("2020-01", "transformer", 5), ("2020-03", "transformer", 9),
        ("2020-02", "rare", 2),
        ("2020-01", "graphene", 40), ("2020-02", "graphene", 30),
    ]
    connection.executemany("INSERT INTO term_month VALUES (?, ?, ?)", rows)
    return connection


@pytest.mark.integration
def test_index_applies_the_frequency_floor(con, tmp_path):
    periods = ["2020-01", "2020-02", "2020-03"]

    vocab = build_term_index(con, periods, tmp_path, min_df=10)

    assert set(vocab["terms"]) == {"transformer", "graphene"}
    # The rare term is counted, not silently forgotten: the UI needs to say
    # "below the indexing threshold" rather than render a confident zero.
    assert vocab["below_threshold_count"] == 1
    assert vocab["min_df"] == 10


@pytest.mark.integration
def test_shards_are_written_by_first_character(con, tmp_path):
    vocab = build_term_index(con, ["2020-01", "2020-02", "2020-03"], tmp_path, min_df=10)

    # Filenames are content-hashed, so the manifest's shard map is the only way
    # to find them. That indirection is the point: a rebuilt shard gets a new
    # name and cannot be served from cache alongside a fresh manifest.
    assert set(vocab["shards"]) == {"t", "g"}
    for key, filename in vocab["shards"].items():
        assert filename.startswith(f"t-{key}-") and filename.endswith(".json")
        assert (tmp_path / "terms" / filename).exists()

    payload = json.loads((tmp_path / "terms" / vocab["shards"]["t"]).read_text())
    assert "transformer" in payload


@pytest.mark.integration
def test_sparse_encoding_round_trips(con, tmp_path):
    """The {offset, values} encoding must reconstruct the dense monthly vector."""
    periods = ["2020-01", "2020-02", "2020-03"]

    vocab = build_term_index(con, periods, tmp_path, min_df=10)
    shard = tmp_path / "terms" / vocab["shards"]["t"]
    entry = json.loads(shard.read_text())["transformer"]

    dense = [0] * len(periods)
    for i, value in enumerate(entry["v"]):
        dense[entry["o"] + i] = value

    assert dense == [5, 0, 9]
    # Leading zeros are dropped, interior zeros survive.
    assert entry["o"] == 0
    assert entry["v"] == [5, 0, 9]


@pytest.mark.integration
def test_vocabulary_stores_only_the_document_frequency(con, tmp_path):
    """The shard key is derivable from the first character and must not be stored."""
    vocab = build_term_index(con, ["2020-01", "2020-02", "2020-03"], tmp_path, min_df=10)

    assert vocab["terms"]["transformer"] == 14
    assert isinstance(vocab["terms"]["graphene"], int)


@pytest.mark.integration
def test_shard_files_are_deterministic(con, tmp_path):
    """Stable URLs plus caching make non-deterministic output a correctness bug."""
    periods = ["2020-01", "2020-02", "2020-03"]

    first_vocab = build_term_index(con, periods, tmp_path / "a", min_df=10)
    second_vocab = build_term_index(con, periods, tmp_path / "b", min_df=10)

    # Identical input must yield an identical hash, or every rebuild renames
    # every shard and busts the cache for no reason.
    assert first_vocab["shards"] == second_vocab["shards"]
    name = first_vocab["shards"]["t"]
    assert (tmp_path / "a" / "terms" / name).read_bytes() == (
        tmp_path / "b" / "terms" / name
    ).read_bytes()

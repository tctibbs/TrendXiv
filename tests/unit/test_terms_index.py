"""Tests for term-index encoding."""

import pytest

from src.pipeline.terms import STOPWORDS, _trim, discovery_candidates, shard_key


@pytest.mark.unit
@pytest.mark.parametrize(
    ("term", "expected"),
    [("transformer", "t"), ("gan", "g"), ("3d", "0"), ("+", "_"), ("zebra", "z")],
)
def test_shard_key(term, expected):
    assert shard_key(term) == expected


@pytest.mark.unit
def test_trim_drops_leading_and_trailing_zeros():
    assert _trim([0, 0, 3, 4, 0, 5, 0, 0]) == (2, [3, 4, 0, 5])


@pytest.mark.unit
def test_trim_of_an_all_zero_vector_is_empty():
    """An all-zero term must produce no stored vector at all, not a run of zeros."""
    assert _trim([0, 0, 0]) == (3, [])


@pytest.mark.unit
def test_trim_preserves_interior_zeros():
    """Interior zeros are real months with no papers and must survive."""
    offset, values = _trim([1, 0, 0, 2])
    assert (offset, values) == (0, [1, 0, 0, 2])


@pytest.mark.unit
def test_discovery_excludes_stopwords_and_rare_terms():
    vocab = {"terms": {"transformer": 5000, "the": 900_000, "obscure": 12}}

    assert discovery_candidates(vocab, min_df=200) == ["transformer"]


@pytest.mark.unit
def test_discovery_excludes_boilerplate_by_corpus_share():
    """A term in >3% of every paper ever published is prose, not a topic.

    "performance" sits at 12% of the corpus and no hand-maintained stoplist can
    keep up with academic boilerplate, so the ceiling is measured instead.
    """
    vocab = {"terms": {"transformer": 29_000, "performance": 379_000}}

    kept = discovery_candidates(vocab, min_df=200, corpus_size=3_148_882)

    assert kept == ["transformer"]


@pytest.mark.unit
def test_corpus_share_ceiling_is_opt_in():
    """Without a corpus size there is no ceiling, so nothing is dropped by share."""
    vocab = {"terms": {"transformer": 29_000, "performance": 379_000}}

    kept = discovery_candidates(vocab, min_df=200)

    assert sorted(kept) == ["performance", "transformer"]


@pytest.mark.unit
def test_common_function_words_are_stopwords():
    for word in ("the", "and", "we", "paper", "results"):
        assert word in STOPWORDS

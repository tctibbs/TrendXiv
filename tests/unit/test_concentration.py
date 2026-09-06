"""Tests for field-specificity scoring."""

import numpy as np
import pytest

from src.analysis.concentration import dominant_groups, jensen_shannon, term_specificity

CORPUS = {"cs": 1_000_000, "math": 500_000, "physics": 400_000, "cond-mat": 300_000,
          "astro-ph": 200_000, "q-bio": 50_000}


@pytest.mark.unit
def test_identical_distributions_have_zero_divergence():
    p = np.array([3.0, 1.0, 1.0])
    assert jensen_shannon(p, p) == pytest.approx(0.0, abs=1e-12)


@pytest.mark.unit
def test_disjoint_distributions_reach_the_upper_bound():
    assert jensen_shannon(np.array([1.0, 0.0]), np.array([0.0, 1.0])) == pytest.approx(1.0)


@pytest.mark.unit
def test_divergence_is_symmetric():
    p = np.array([5.0, 2.0, 1.0])
    q = np.array([1.0, 1.0, 6.0])
    assert jensen_shannon(p, q) == pytest.approx(jensen_shannon(q, p))


@pytest.mark.unit
def test_empty_distribution_is_not_an_error():
    assert jensen_shannon(np.array([0.0, 0.0]), np.array([1.0, 1.0])) == 0.0


@pytest.mark.unit
def test_prose_that_tracks_the_corpus_scores_near_zero():
    """A word used indiscriminately mirrors the corpus and must not look specific."""
    prose = {group: count * 0.06 for group, count in CORPUS.items()}
    assert term_specificity(prose, CORPUS) < 0.01


@pytest.mark.unit
def test_a_single_field_term_scores_high():
    assert term_specificity({"cond-mat": 20_000}, CORPUS) > 0.5


@pytest.mark.unit
def test_specificity_is_measured_against_the_corpus_not_uniform():
    """A term entirely inside the LARGEST field must still register as specific.

    Against a uniform reference this term would look unremarkable simply because
    cs is the biggest archive; against the corpus it is clearly concentrated.
    """
    assert term_specificity({"cs": 100_000}, CORPUS) > 0.25


@pytest.mark.unit
def test_dominant_groups_are_ranked_and_normalised():
    groups = dominant_groups({"cs": 70, "math": 20, "physics": 10}, limit=2)
    assert [g for g, _ in groups] == ["cs", "math"]
    assert groups[0][1] == pytest.approx(0.7)


@pytest.mark.unit
def test_dominant_groups_of_nothing_is_empty():
    assert dominant_groups({}) == []

"""Unit tests for the rising-term board."""

import numpy as np
import pytest
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportions_ztest

from src.analysis.rising import _benjamini_hochberg, deduplicate_ngrams, rising_scores

EXPECTED_COLUMNS = [
    "term",
    "share_recent",
    "share_base",
    "delta_logodds",
    "delta_lo",
    "delta_hi",
    "p_value",
    "q_value",
    "significant",
    "is_new",
    "n_screened",
]


def _null_terms(count: int, seed: int = 7) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Build a block of terms whose share is unchanged between the windows.

    Args:
        count: Number of null terms to generate
        seed: Seed for the count noise

    Returns:
        Tuple of (labels, recent counts, baseline counts) at n=10000 per window
    """
    rng = np.random.default_rng(seed)
    k_recent = rng.binomial(10000, 0.01, size=count)
    k_base = rng.binomial(10000, 0.01, size=count)
    return [f"null term {i}" for i in range(count)], k_recent, k_base


@pytest.mark.unit
class TestRisingScores:
    """Tests for rising_scores."""

    def test_small_counts_are_not_promoted_by_a_big_ratio(self):
        """Test that 1 -> 5 papers stays unflagged while 100 -> 500 is flagged."""
        labels, null_recent, null_base = _null_terms(50)
        terms = ["tiny fluke", "real movement", *labels]
        k_recent = np.concatenate([[5, 500], null_recent])
        k_base = np.concatenate([[1, 100], null_base])

        result = rising_scores(terms, k_recent, 10000, k_base, 10000).set_index("term")

        assert not result.loc["tiny fluke", "significant"]
        assert result.loc["real movement", "significant"]

    def test_small_counts_survive_screening_but_still_fail_the_test(self):
        """Test that the ratio alone cannot carry a term once min_count is relaxed."""
        labels, null_recent, null_base = _null_terms(50)
        terms = ["tiny fluke", "real movement", *labels]
        k_recent = np.concatenate([[5, 500], null_recent])
        k_base = np.concatenate([[1, 100], null_base])

        result = rising_scores(terms, k_recent, 10000, k_base, 10000, min_count=1)
        indexed = result.set_index("term")

        assert indexed.loc["tiny fluke", "q_value"] > 0.10
        assert not indexed.loc["tiny fluke", "significant"]
        assert indexed.loc["real movement", "q_value"] < 1e-10
        assert indexed.loc["real movement", "significant"]

    def test_ranks_by_lower_bound_not_point_estimate(self):
        """Test that a noisy term with the larger point estimate ranks below a solid one."""
        terms = ["noisy", "solid"]
        # "noisy" has the bigger log-odds jump but only a handful of papers behind it.
        result = rising_scores(
            terms,
            k_recent=np.array([30, 600]),
            n_recent=np.array([10000, 10000]),
            k_base=np.array([2, 100]),
            n_base=np.array([10000, 10000]),
        )

        assert result["delta_lo"].is_monotonic_decreasing
        indexed = result.set_index("term")
        assert indexed.loc["noisy", "delta_logodds"] > indexed.loc["solid", "delta_logodds"]
        assert list(result["term"]) == ["solid", "noisy"]

    def test_screened_terms_are_returned_but_not_significant(self):
        """Test that thin terms stay in the frame with significant=False."""
        terms = ["thin", "thick"]
        result = rising_scores(
            terms,
            k_recent=np.array([4, 400]),
            n_recent=10000,
            k_base=np.array([1, 100]),
            n_base=10000,
            min_count=25,
        ).set_index("term")

        assert np.isnan(result.loc["thin", "q_value"])
        assert not result.loc["thin", "significant"]
        assert result.loc["thin", "delta_logodds"] > 0
        assert result.loc["thick", "significant"]

    def test_n_screened_counts_terms_entering_the_correction(self):
        """Test that n_screened reports the size of the multiplicity correction."""
        terms = ["a", "b", "c"]
        result = rising_scores(
            terms,
            k_recent=np.array([100, 100, 2]),
            n_recent=10000,
            k_base=np.array([50, 50, 1]),
            n_base=10000,
            min_count=25,
        )

        assert set(result["n_screened"]) == {2}

    def test_new_term_is_flagged_without_a_percentage(self):
        """Test the is_new rule: no baseline papers and a real recent count."""
        terms = ["brand new", "barely there"]
        result = rising_scores(
            terms,
            k_recent=np.array([40, 3]),
            n_recent=10000,
            k_base=np.array([0, 0]),
            n_base=10000,
            min_count=25,
        ).set_index("term")

        assert result.loc["brand new", "is_new"]
        assert not result.loc["barely there", "is_new"]
        assert np.isfinite(result.loc["brand new", "delta_logodds"])

    def test_declining_term_is_never_flagged_as_rising(self):
        """Test that a significant decline is reported but not called rising."""
        terms = ["falling"]
        result = rising_scores(
            terms,
            k_recent=np.array([100]),
            n_recent=10000,
            k_base=np.array([500]),
            n_base=10000,
        ).iloc[0]

        assert result["delta_logodds"] < 0
        assert result["delta_hi"] < 0
        assert result["q_value"] < 0.01
        assert not result["significant"]

    def test_strata_keep_priors_from_bleeding_across_fields(self):
        """Test that shrinkage borrows strength within a stratum, not across all terms."""
        # Stratum 0 is a busy field at a 30% share; stratum 1 is a rare-term field.
        terms = ["hot a", "hot b", "hot c", "hot thin", "rare a", "rare b"]
        counts = np.array([30, 30, 30, 1, 1, 1])
        n = np.full(6, 100)
        strata = np.array([0, 0, 0, 0, 1, 1])

        stratified = rising_scores(terms, counts, n, counts, n, strata=strata).set_index("term")
        pooled = rising_scores(terms, counts, n, counts, n).set_index("term")

        # A thin term in a busy field is pulled up toward its own field's base
        # rate; the global fit dilutes that pull with the rare field.
        assert stratified.loc["hot thin", "share_recent"] > pooled.loc["hot thin", "share_recent"]
        assert stratified.loc["hot thin", "share_recent"] > 0.01

    def test_p_values_match_the_reference_two_proportion_ztest(self):
        """Test the per-term p-value against statsmodels' pooled two-sided z-test."""
        k_recent = np.array([500, 5, 100, 0, 300])
        k_base = np.array([100, 1, 500, 50, 300])
        n = np.full(5, 10000)

        result = rising_scores(
            [f"t{i}" for i in range(5)], k_recent, n, k_base, n, min_count=0
        ).set_index("term")

        for i in range(5):
            expected = proportions_ztest([k_recent[i], k_base[i]], [n[i], n[i]])[1]
            assert result.loc[f"t{i}", "p_value"] == pytest.approx(expected)

    def test_estimate_and_interval_are_symmetric_under_a_window_swap(self):
        """Test that swapping the windows negates the estimate and preserves the width.

        A log-odds difference is antisymmetric and its standard error must not
        care which window is called recent; an asymmetric error term would show
        up here as two different band widths.
        """
        terms = ["a", "b", "c"]
        k_recent, n_recent = np.array([300, 20, 5]), np.array([10000, 8000, 12000])
        k_base, n_base = np.array([100, 25, 60]), np.array([9000, 8000, 11000])

        # Row order follows delta_lo and therefore differs between the two runs.
        forward = rising_scores(terms, k_recent, n_recent, k_base, n_base).set_index("term")
        forward = forward.loc[terms]
        swapped = rising_scores(terms, k_base, n_base, k_recent, n_recent).set_index("term")
        swapped = swapped.loc[terms]

        assert forward["delta_logodds"].to_numpy() == pytest.approx(
            -swapped["delta_logodds"].to_numpy()
        )
        assert (forward["delta_hi"] - forward["delta_lo"]).to_numpy() == pytest.approx(
            (swapped["delta_hi"] - swapped["delta_lo"]).to_numpy()
        )
        assert forward["p_value"].to_numpy() == pytest.approx(swapped["p_value"].to_numpy())

    def test_screening_uses_the_combined_count(self):
        """Test that a term thin only in the recent window still enters the correction."""
        result = rising_scores(
            ["collapsed", "grown"],
            k_recent=np.array([2, 300]),
            n_recent=10000,
            k_base=np.array([200, 100]),
            n_base=10000,
            min_count=25,
        ).set_index("term")

        assert set(result["n_screened"]) == {2}
        assert np.isfinite(result.loc["collapsed", "q_value"])
        assert not result.loc["collapsed", "significant"]

    def test_returns_documented_schema(self):
        """Test the exact column set, including the absence of a percent column."""
        result = rising_scores(["a"], np.array([50]), 1000, np.array([25]), 1000)

        assert list(result.columns) == EXPECTED_COLUMNS
        assert result["significant"].dtype == bool
        assert result["is_new"].dtype == bool

    def test_empty_input_returns_empty_frame(self):
        """Test that no terms yields an empty frame with the same schema."""
        result = rising_scores([], np.array([]), 1000, np.array([]), 1000)

        assert list(result.columns) == EXPECTED_COLUMNS
        assert len(result) == 0

    def test_mismatched_lengths_raise(self):
        """Test the input validation guard."""
        with pytest.raises(ValueError, match="same length"):
            rising_scores(["a", "b"], np.array([1]), 1000, np.array([1]), 1000)

    def test_mismatched_strata_raise(self):
        """Test the strata validation guard."""
        with pytest.raises(ValueError, match="one label per term"):
            rising_scores(
                ["a", "b"],
                np.array([1, 2]),
                1000,
                np.array([1, 2]),
                1000,
                strata=np.array([0]),
            )

    def test_zero_exposure_term_is_not_screened(self):
        """Test that a term with no reference papers cannot be significant."""
        result = rising_scores(
            ["ghost", "solid"],
            k_recent=np.array([0, 400]),
            n_recent=np.array([0, 10000]),
            k_base=np.array([0, 100]),
            n_base=np.array([0, 10000]),
        ).set_index("term")

        assert result.loc["ghost", "p_value"] == 1.0
        assert not result.loc["ghost", "significant"]
        assert set(result["n_screened"]) == {1}


@pytest.mark.unit
class TestBenjaminiHochberg:
    """Tests for the FDR correction."""

    def test_matches_hand_computed_q_values(self):
        """Test q = min over j >= i of (m / j) * p_j on a worked example."""
        p_values = np.array([0.001, 0.008, 0.039, 0.041, 0.042])

        q_values = _benjamini_hochberg(p_values)

        # m=5: 5*0.001, 5/2*0.008, then 5/3*0.039 and 5/4*0.041 both exceed
        # 5/5*0.042, so the tail is pulled down to 0.042 by monotonicity.
        assert q_values == pytest.approx([0.005, 0.020, 0.042, 0.042, 0.042])

    def test_preserves_input_order(self):
        """Test that q-values are returned against their own p-values."""
        p_values = np.array([0.041, 0.001, 0.042, 0.008, 0.039])

        q_values = _benjamini_hochberg(p_values)

        assert q_values == pytest.approx([0.042, 0.005, 0.042, 0.020, 0.042])

    def test_matches_statsmodels(self):
        """Test agreement with the reference implementation on random p-values."""
        rng = np.random.default_rng(3)
        p_values = rng.uniform(size=200) ** 2

        expected = multipletests(p_values, alpha=0.10, method="fdr_bh")[1]

        assert _benjamini_hochberg(p_values) == pytest.approx(expected)

    def test_empty_input(self):
        """Test that no screened terms produces no q-values."""
        assert _benjamini_hochberg(np.array([])).size == 0

    def test_q_values_are_clipped_to_one(self):
        """Test that large p-values cannot produce a q-value above 1."""
        q_values = _benjamini_hochberg(np.array([0.9, 0.95, 0.99]))

        assert np.all(q_values <= 1.0)


@pytest.mark.unit
class TestDeduplicateNgrams:
    """Tests for deduplicate_ngrams."""

    def test_drops_shorter_ngram_with_similar_score(self):
        """Test that a contained term riding on its container is removed."""
        terms = ["large language model", "language model", "diffusion"]
        scores = [2.00, 1.95, 1.20]

        assert deduplicate_ngrams(terms, scores) == [0, 2]

    def test_keeps_shorter_ngram_when_scores_diverge(self):
        """Test that a differently behaving short term is its own finding."""
        terms = ["large language model", "language model"]
        scores = [2.00, 1.00]

        assert deduplicate_ngrams(terms, scores) == [0, 1]

    def test_requires_contiguous_tokens(self):
        """Test that scattered token matches do not count as containment."""
        terms = ["language based model", "language model"]
        scores = [2.00, 2.00]

        assert deduplicate_ngrams(terms, scores) == [0, 1]

    def test_partial_word_matches_do_not_absorb(self):
        """Test that matching is on whole tokens, not substrings."""
        terms = ["modelling", "model"]
        scores = [1.00, 1.00]

        assert deduplicate_ngrams(terms, scores) == [0, 1]

    def test_equal_length_terms_are_both_kept(self):
        """Test that only strictly shorter terms can be absorbed."""
        terms = ["neural network", "neural network"]
        scores = [1.00, 1.00]

        assert deduplicate_ngrams(terms, scores) == [0, 1]

    def test_zero_scores_are_treated_as_close(self):
        """Test the relative-tolerance guard when both scores are zero."""
        terms = ["large language model", "language model"]
        scores = [0.0, 0.0]

        assert deduplicate_ngrams(terms, scores) == [0]

    def test_case_is_ignored(self):
        """Test that containment is checked on lowercased tokens."""
        terms = ["Large Language Model", "language model"]
        scores = [2.00, 2.05]

        assert deduplicate_ngrams(terms, scores) == [0]

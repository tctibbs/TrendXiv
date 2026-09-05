"""Ranking of rising terms with shrinkage, multiplicity control, and honest bounds.

Google Trends scores a term by raw percent change and calls anything above 5000%
a "Breakout". At arXiv volumes that promotes one paper becoming five to the top of
the board and buries the real movements. Three layers replace it: shrink the shares
toward a fitted prior, test every term and correct for multiplicity with
Benjamini-Hochberg, and rank by the lower confidence bound instead of the point
estimate.
"""

from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy import stats

from src.analysis.intervals import _fit_beta_prior

# Normal quantile for the 95% bounds reported on the log-odds difference.
_Z_95 = 1.96

# Prior mass used when the method-of-moments fit is degenerate: two pseudo-papers
# centred on the stratum's pooled share, weak enough to leave real terms alone.
_FALLBACK_PRIOR_STRENGTH = 2.0


def rising_scores(
    terms: Sequence[str],
    k_recent: np.ndarray,
    n_recent: np.ndarray | int,
    k_base: np.ndarray,
    n_base: np.ndarray | int,
    q: float = 0.10,
    min_count: int = 25,
    strata: np.ndarray | None = None,
) -> pd.DataFrame:
    """Score terms for growth between a baseline window and a recent window.

    Growth is measured as a log-odds difference between shrunk shares, so the
    metric stays finite and symmetric at the boundaries where a percent change
    would explode. Terms are ranked by the lower bound of that difference, which
    demotes flukes without an arbitrary count cutoff.

    Args:
        terms: Term labels, one per row
        k_recent: Matched papers per term in the recent window
        n_recent: Reference-set papers in the recent window, scalar or per term
        k_base: Matched papers per term in the baseline window
        n_base: Reference-set papers in the baseline window, scalar or per term
        q: Benjamini-Hochberg false discovery rate. Defaults to 0.10.
        min_count: Minimum combined matches for a term to enter the multiple
            testing correction. Defaults to 25.
        strata: Optional stratum label per term; the shrinkage prior is fitted
            within each stratum. Defaults to None (one global stratum).

    Returns:
        DataFrame sorted by delta_lo descending with columns term, share_recent,
        share_base, delta_logodds, delta_lo, delta_hi, p_value, q_value,
        significant, is_new, n_screened
    """
    term_list = list(terms)
    n_terms = len(term_list)

    if n_terms == 0:
        return _empty_frame()

    kr = np.asarray(k_recent, dtype=float).reshape(-1)
    kb = np.asarray(k_base, dtype=float).reshape(-1)
    if not (len(kr) == len(kb) == n_terms):
        raise ValueError("terms, k_recent and k_base must have the same length")

    nr = np.broadcast_to(np.asarray(n_recent, dtype=float), (n_terms,)).astype(float)
    nb = np.broadcast_to(np.asarray(n_base, dtype=float), (n_terms,)).astype(float)

    stratum_labels = (
        np.asarray(strata).reshape(-1) if strata is not None else np.zeros(n_terms, dtype=int)
    )
    if len(stratum_labels) != n_terms:
        raise ValueError("strata must have one label per term")

    alpha, beta = _stratum_priors(kr, nr, kb, nb, stratum_labels)

    # Pseudo-counts of the shrunk Beta posterior; logit(a / (a + b)) is log(a / b),
    # so the log-odds difference and its variance both come straight from these.
    a_recent = kr + alpha
    b_recent = np.maximum(nr - kr, 0.0) + beta
    a_base = kb + alpha
    b_base = np.maximum(nb - kb, 0.0) + beta

    share_recent = a_recent / (a_recent + b_recent)
    share_base = a_base / (a_base + b_base)

    delta = np.log(a_recent / b_recent) - np.log(a_base / b_base)
    standard_error = np.sqrt(1.0 / a_recent + 1.0 / b_recent + 1.0 / a_base + 1.0 / b_base)
    delta_lo = delta - _Z_95 * standard_error
    delta_hi = delta + _Z_95 * standard_error

    p_value = _two_proportion_p_values(kr, nr, kb, nb)

    screened = ((kr + kb) >= min_count) & (nr > 0) & (nb > 0)
    q_value = np.full(n_terms, np.nan)
    if screened.any():
        q_value[screened] = _benjamini_hochberg(p_value[screened])

    # A rising board reports growth: declines still get a p-value but are never
    # flagged, so "12 survive at FDR 10%" always means twelve terms going up.
    # Unscreened terms carry a NaN q-value; fill it so the comparison is defined.
    significant = screened & (np.where(screened, q_value, 1.0) <= q) & (delta > 0)

    result = pd.DataFrame(
        {
            "term": term_list,
            "share_recent": share_recent,
            "share_base": share_base,
            "delta_logodds": delta,
            "delta_lo": delta_lo,
            "delta_hi": delta_hi,
            "p_value": p_value,
            "q_value": q_value,
            "significant": significant,
            "is_new": (kb == 0) & (kr >= min_count),
            "n_screened": int(screened.sum()),
        }
    )
    ranked = result.sort_values("delta_lo", ascending=False, na_position="last")
    return ranked.reset_index(drop=True)


def deduplicate_ngrams(terms: Sequence[str], scores: Sequence[float]) -> list[int]:
    """Drop shorter n-grams that are absorbed by a longer, similarly scoring one.

    "language model" inside "large language model" is not an independent finding;
    the two counts move together, which inflates the number of tests and breaks
    the FDR arithmetic. A short term survives only when its score differs enough
    from its container to be telling a different story.

    Args:
        terms: Term labels, whitespace-tokenised for the containment check
        scores: Score per term, on the same scale for all terms

    Returns:
        Indices of the terms to keep, in their original order
    """
    tokens = [tuple(term.lower().split()) for term in terms]
    score_values = np.asarray(scores, dtype=float).reshape(-1)

    keep: list[int] = []
    for i, short_tokens in enumerate(tokens):
        absorbed = any(
            len(short_tokens) < len(long_tokens)
            and _is_contiguous_subsequence(short_tokens, long_tokens)
            and _scores_are_close(score_values[i], score_values[j])
            for j, long_tokens in enumerate(tokens)
            if j != i
        )
        if not absorbed:
            keep.append(i)
    return keep


def _stratum_priors(
    kr: np.ndarray,
    nr: np.ndarray,
    kb: np.ndarray,
    nb: np.ndarray,
    stratum_labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit a Beta prior per stratum and broadcast it back over terms.

    Args:
        kr: Matched papers per term in the recent window
        nr: Reference papers per term in the recent window
        kb: Matched papers per term in the baseline window
        nb: Reference papers per term in the baseline window
        stratum_labels: Stratum label per term

    Returns:
        Tuple of (a, b) arrays holding each term's prior pseudo-counts
    """
    alpha = np.zeros(len(kr), dtype=float)
    beta = np.zeros(len(kr), dtype=float)

    for label in np.unique(stratum_labels):
        in_stratum = stratum_labels == label
        counts = np.concatenate([kr[in_stratum], kb[in_stratum]])
        exposures = np.concatenate([nr[in_stratum], nb[in_stratum]])
        observed = exposures > 0

        total_exposure = exposures[observed].sum()
        pooled = counts[observed].sum() / total_exposure if total_exposure > 0 else 0.0

        fitted_a, fitted_b = _fit_beta_prior(
            counts[observed] / exposures[observed], pooled, exposures=exposures[observed]
        )
        if fitted_a is None or fitted_b is None:
            # Keep the prior positive so every log-odds stays finite, and keep it
            # centred where the stratum's data is rather than at one half.
            guard = 1.0 / (total_exposure + 2.0) if total_exposure > 0 else 0.5
            mean = float(np.clip(pooled, guard, 1.0 - guard))
            fitted_a = _FALLBACK_PRIOR_STRENGTH * mean
            fitted_b = _FALLBACK_PRIOR_STRENGTH * (1.0 - mean)

        alpha[in_stratum] = fitted_a
        beta[in_stratum] = fitted_b

    return alpha, beta


def _two_proportion_p_values(
    kr: np.ndarray,
    nr: np.ndarray,
    kb: np.ndarray,
    nb: np.ndarray,
) -> np.ndarray:
    """Run a two-sided two-proportion z-test per term on the raw counts.

    Args:
        kr: Matched papers per term in the recent window
        nr: Reference papers per term in the recent window
        kb: Matched papers per term in the baseline window
        nb: Reference papers per term in the baseline window

    Returns:
        Array of p-values, 1.0 where a term carries no information
    """
    testable = (nr > 0) & (nb > 0)
    nr_safe = np.where(testable, nr, 1.0)
    nb_safe = np.where(testable, nb, 1.0)

    pooled = (kr + kb) / (nr_safe + nb_safe)
    variance = pooled * (1.0 - pooled) * (1.0 / nr_safe + 1.0 / nb_safe)

    has_variance = testable & (variance > 0)
    variance_safe = np.where(has_variance, variance, 1.0)
    z_stat = (kr / nr_safe - kb / nb_safe) / np.sqrt(variance_safe)

    p_values = 2.0 * stats.norm.sf(np.abs(z_stat))
    return np.where(has_variance, p_values, 1.0)


def _benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    """Convert p-values to Benjamini-Hochberg FDR q-values.

    Args:
        p_values: One p-value per screened term

    Returns:
        Array of q-values in the input order, each clipped to [0, 1]
    """
    p_arr = np.asarray(p_values, dtype=float)
    m = p_arr.size
    if m == 0:
        return p_arr.copy()

    order = np.argsort(p_arr, kind="stable")
    ranks = np.arange(1, m + 1, dtype=float)
    scaled = p_arr[order] * m / ranks
    # Enforce monotonicity: a q-value can never exceed that of a larger p-value.
    q_sorted = np.minimum.accumulate(scaled[::-1])[::-1]

    q_values = np.empty(m, dtype=float)
    q_values[order] = np.clip(q_sorted, 0.0, 1.0)
    return q_values


def _is_contiguous_subsequence(short: tuple[str, ...], long: tuple[str, ...]) -> bool:
    """Check whether one token tuple appears contiguously inside another.

    Args:
        short: Candidate inner token sequence
        long: Container token sequence

    Returns:
        True when short appears as a contiguous run of tokens inside long
    """
    span = len(short)
    return any(long[start : start + span] == short for start in range(len(long) - span + 1))


def _scores_are_close(first: float, second: float, tolerance: float = 0.10) -> bool:
    """Check whether two scores agree to within a relative tolerance.

    Args:
        first: First score
        second: Second score
        tolerance: Allowed relative difference. Defaults to 0.10.

    Returns:
        True when the scores are within tolerance of each other
    """
    scale = max(abs(first), abs(second))
    if scale == 0.0:
        return True
    return abs(first - second) <= tolerance * scale


def _empty_frame() -> pd.DataFrame:
    """Build the empty result frame with the documented columns and dtypes.

    Returns:
        Zero-row DataFrame matching the rising_scores schema
    """
    return pd.DataFrame(
        {
            "term": pd.Series(dtype="object"),
            "share_recent": pd.Series(dtype="float64"),
            "share_base": pd.Series(dtype="float64"),
            "delta_logodds": pd.Series(dtype="float64"),
            "delta_lo": pd.Series(dtype="float64"),
            "delta_hi": pd.Series(dtype="float64"),
            "p_value": pd.Series(dtype="float64"),
            "q_value": pd.Series(dtype="float64"),
            "significant": pd.Series(dtype="bool"),
            "is_new": pd.Series(dtype="bool"),
            "n_screened": pd.Series(dtype="int64"),
        }
    )

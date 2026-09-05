"""Uncertainty quantification for share (proportion) series.

Every share TrendXiv draws is a ratio of matched papers to reference papers in a
period. Those ratios are estimated from small, clustered counts, so each one ships
with an interval band rather than as a bare point.
"""

import numpy as np


def wilson_interval(
    k: np.ndarray,
    n: np.ndarray,
    z: float = 1.96,
    phi: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Wilson score intervals for a series of binomial proportions.

    The Wald interval collapses to zero width at k=0 and k=n and is badly
    miscalibrated whenever k is small, which is the regime a research-term tool
    lives in. The Wilson score interval stays inside [0, 1] and keeps sensible
    coverage down to a single matched paper.

    Args:
        k: Matched paper counts per period (numerator)
        n: Reference-set paper counts per period (denominator)
        z: Normal quantile for the desired coverage. Defaults to 1.96 (95%).
        phi: Overdispersion inflation factor from estimate_overdispersion. The
            effective quantile is z * sqrt(phi). Defaults to 1.0 (pure binomial).

    Returns:
        Tuple of (lower, upper) arrays clipped to [0, 1], with NaN where n == 0
    """
    k_arr = np.asarray(k, dtype=float)
    n_arr = np.asarray(n, dtype=float)

    has_exposure = n_arr > 0
    # Substitute a dummy denominator so the arithmetic below never divides by
    # zero; those entries are overwritten with NaN at the end.
    n_safe = np.where(has_exposure, n_arr, 1.0)

    z_eff = float(z) * np.sqrt(phi)
    z_sq = z_eff**2

    denominator = n_safe + z_sq
    center = (k_arr + z_sq / 2.0) / denominator
    variance = np.maximum(k_arr * (n_safe - k_arr) / n_safe + z_sq / 4.0, 0.0)
    margin = (z_eff / denominator) * np.sqrt(variance)

    lower = np.clip(center - margin, 0.0, 1.0)
    upper = np.clip(center + margin, 0.0, 1.0)

    lower = np.where(has_exposure, lower, np.nan)
    upper = np.where(has_exposure, upper, np.nan)
    return lower, upper


def estimate_overdispersion(k: np.ndarray, n: np.ndarray) -> float:
    """Estimate a quasi-binomial overdispersion factor from a share series.

    Papers are not independent Bernoulli draws: topics cluster in time and one
    lab can post five papers on the same day. The plain binomial variance is
    therefore too small, the bands too tight, and every wiggle looks
    significant. This is the Pearson chi-square of the counts against the pooled
    proportion divided by its degrees of freedom.

    Args:
        k: Matched paper counts per period (numerator)
        n: Reference-set paper counts per period (denominator)

    Returns:
        Inflation factor, floored at 1.0 so the band is never narrower than
        binomial
    """
    k_arr = np.asarray(k, dtype=float)
    n_arr = np.asarray(n, dtype=float)

    has_exposure = n_arr > 0
    k_obs = k_arr[has_exposure]
    n_obs = n_arr[has_exposure]

    degrees_of_freedom = k_obs.size - 1
    if degrees_of_freedom < 1:
        return 1.0

    # Every retained period has n > 0, so the pooled proportion is well defined.
    p_bar = k_obs.sum() / n_obs.sum()
    # A series that is all-zero or all-matched carries no dispersion signal.
    if p_bar <= 0.0 or p_bar >= 1.0:
        return 1.0

    expected = n_obs * p_bar
    residuals = (k_obs - expected) ** 2 / (expected * (1.0 - p_bar))
    phi = float(residuals.sum() / degrees_of_freedom)
    return max(1.0, phi)


def smoothed_share(
    k: np.ndarray,
    n: np.ndarray,
    prior_strength: float | None = None,
) -> np.ndarray:
    """Compute empirical-Bayes shrunk shares for a count series.

    A month with one reference paper and one match is not 100% adoption. Shares
    are shrunk toward a Beta(a, b) prior fitted to the series itself, so thin
    months are pulled to the series average while well-populated months move
    hardly at all.

    Args:
        k: Matched paper counts per period (numerator)
        n: Reference-set paper counts per period (denominator)
        prior_strength: Total prior mass a + b. When given it overrides the
            method-of-moments concentration and only the prior mean is fitted.
            Defaults to None (fit both from the data).

    Returns:
        Array of shrunk shares, (k + a) / (n + a + b), with periods of zero
        exposure taking the prior mean
    """
    k_arr = np.asarray(k, dtype=float)
    n_arr = np.asarray(n, dtype=float)

    has_exposure = n_arr > 0
    total_n = n_arr[has_exposure].sum()
    if total_n <= 0:
        return np.zeros_like(k_arr, dtype=float)

    pooled = k_arr[has_exposure].sum() / total_n
    alpha, beta = _fit_beta_prior(
        k_arr[has_exposure] / n_arr[has_exposure],
        pooled,
        prior_strength,
        exposures=n_arr[has_exposure],
    )
    if alpha is None or beta is None:
        return np.full(k_arr.shape, pooled, dtype=float)

    return (k_arr + alpha) / (n_arr + alpha + beta)


def _fit_beta_prior(
    shares: np.ndarray,
    pooled: float,
    prior_strength: float | None = None,
    exposures: np.ndarray | None = None,
) -> tuple[float, float] | tuple[None, None]:
    """Fit a Beta prior to observed shares by method of moments.

    The prior mean is the unweighted mean of the observed shares, not the
    exposure-weighted pooled proportion: the concentration below is matched
    against the unweighted variance, and mixing the two moments would not
    describe any single Beta.

    Args:
        shares: Observed per-period shares from periods with exposure
        pooled: Pooled proportion, used as the prior mean only when no share was
            observed at all
        prior_strength: Total prior mass a + b, or None to derive it from the
            observed variance
        exposures: Per-period denominators. When given, the binomial sampling
            component is subtracted before matching, which is what makes this a
            beta-binomial fit rather than a Beta fit to noisy point estimates.

    Returns:
        Tuple of (a, b), or (None, None) when the moment match is degenerate
    """
    mean = float(np.mean(shares)) if shares.size else pooled
    if not 0.0 < mean < 1.0:
        return None, None

    if prior_strength is not None:
        if prior_strength <= 0:
            return None, None
        strength = float(prior_strength)
    else:
        if shares.size < 2:
            return None, None
        variance = float(np.var(shares, ddof=1))
        max_variance = mean * (1.0 - mean)

        # The observed spread of monthly shares is Var(prior) + E[p(1-p)/n]:
        # part real dispersion between months, part binomial noise within them.
        # Matching against the raw spread therefore credits sampling noise to the
        # prior, computes far too weak a concentration, and under-shrinks exactly
        # the thin months shrinkage exists for. arXiv's 1991 months carry tens of
        # papers against today's tens of thousands, so this is the normal case
        # here, not an edge case.
        if exposures is not None and exposures.size == shares.size and exposures.size:
            with np.errstate(divide="ignore", invalid="ignore"):
                sampling = np.divide(
                    shares * (1.0 - shares), exposures,
                    out=np.zeros_like(shares, dtype=float),
                    where=exposures > 0,
                )
            excess = variance - float(np.mean(sampling))
        else:
            excess = variance

        if excess >= max_variance:
            # Dispersion at or above the Bernoulli ceiling implies a U-shaped
            # prior with non-positive concentration; nothing to shrink toward.
            return None, None
        if excess <= 0.0:
            # The months are consistent with one constant underlying rate, so the
            # right answer is to shrink hard toward it. Using the total exposure
            # as the prior mass does that without collapsing to a flat series.
            strength = float(np.sum(exposures)) if exposures is not None else 0.0
            if strength <= 0.0:
                return None, None
        else:
            strength = max_variance / excess - 1.0

    return mean * strength, (1.0 - mean) * strength

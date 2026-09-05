"""Unit tests for proportion interval estimation."""

import numpy as np
import pytest
from statsmodels.stats.proportion import proportion_confint

from src.analysis.intervals import (
    estimate_overdispersion,
    smoothed_share,
    wilson_interval,
)


@pytest.mark.unit
class TestWilsonInterval:
    """Tests for wilson_interval."""

    def test_matches_published_value_at_zero_matches(self):
        """Test k=0, n=10 against the published 95% Wilson interval."""
        lower, upper = wilson_interval(np.array([0]), np.array([10]))

        assert lower[0] == pytest.approx(0.0, abs=1e-4)
        assert upper[0] == pytest.approx(0.2775, abs=1e-4)

    def test_matches_published_value_at_full_matches(self):
        """Test k=n=10 against the published 95% Wilson interval."""
        lower, upper = wilson_interval(np.array([10]), np.array([10]))

        assert lower[0] == pytest.approx(0.7225, abs=1e-4)
        assert upper[0] == pytest.approx(1.0, abs=1e-4)

    def test_matches_published_value_at_an_interior_count(self):
        """Test k=5, n=20 against the published 95% Wilson interval.

        The two boundary cases above both have k*(n-k) = 0, so they cannot see
        an error in the variance term. This one can.
        """
        lower, upper = wilson_interval(np.array([5]), np.array([20]))

        assert lower[0] == pytest.approx(0.1119, abs=1e-4)
        assert upper[0] == pytest.approx(0.4687, abs=1e-4)

    def test_matches_reference_implementation_across_the_range(self):
        """Test agreement with statsmodels' Wilson interval on a grid of counts."""
        k = np.array([0, 1, 2, 3, 5, 7, 10, 25, 50, 99, 100])
        n = np.array([10, 10, 10, 7, 20, 20, 10, 100, 100, 100, 100])

        lower, upper = wilson_interval(k, n)
        expected_lower, expected_upper = proportion_confint(k, n, alpha=0.05, method="wilson")

        # z=1.96 rather than the exact 1.959964 quantile costs about 1e-5.
        assert lower == pytest.approx(expected_lower, abs=1e-4)
        assert upper == pytest.approx(expected_upper, abs=1e-4)

    def test_interval_brackets_the_observed_share(self):
        """Test that the band contains the point estimate for interior counts."""
        k = np.array([5, 50, 500])
        n = np.array([100, 1000, 10000])

        lower, upper = wilson_interval(k, n)

        share = k / n
        assert np.all(lower < share)
        assert np.all(upper > share)

    def test_band_narrows_as_exposure_grows(self):
        """Test that a larger reference set yields a tighter band at fixed share."""
        lower, upper = wilson_interval(np.array([5, 50, 500]), np.array([100, 1000, 10000]))

        widths = upper - lower
        assert widths[0] > widths[1] > widths[2]

    def test_zero_exposure_yields_nan(self):
        """Test that periods with no reference papers return NaN, not a divide."""
        lower, upper = wilson_interval(np.array([0, 3]), np.array([0, 100]))

        assert np.isnan(lower[0])
        assert np.isnan(upper[0])
        assert np.isfinite(lower[1])
        assert np.isfinite(upper[1])

    def test_overdispersion_widens_the_band(self):
        """Test that phi inflates the effective quantile."""
        k = np.array([50])
        n = np.array([1000])

        binomial_lo, binomial_hi = wilson_interval(k, n)
        inflated_lo, inflated_hi = wilson_interval(k, n, phi=4.0)

        assert inflated_lo[0] < binomial_lo[0]
        assert inflated_hi[0] > binomial_hi[0]
        # phi=4 doubles z, so the band is close to twice as wide.
        widening = (inflated_hi[0] - inflated_lo[0]) / (binomial_hi[0] - binomial_lo[0])
        assert widening == pytest.approx(2.0, abs=0.1)

    def test_bounds_stay_inside_the_unit_interval(self):
        """Test that clipping holds for extreme counts and wide bands."""
        lower, upper = wilson_interval(np.array([0, 1, 4, 5]), np.array([5, 5, 5, 5]), phi=9.0)

        assert np.all(lower >= 0.0)
        assert np.all(upper <= 1.0)


@pytest.mark.unit
class TestEstimateOverdispersion:
    """Tests for estimate_overdispersion."""

    def test_matches_hand_computed_pearson_chi_square(self):
        """Test the exact quasi-binomial factor on a three-period worked example.

        k=(10, 20, 30) at n=100 pools to p=0.2, so the Pearson terms are
        100/16, 0 and 100/16; over df=2 that is exactly 6.25.
        """
        phi = estimate_overdispersion(np.array([10, 20, 30]), np.full(3, 100))

        assert phi == pytest.approx(6.25)

    def test_clean_binomial_simulation_returns_about_one(self):
        """Test that independent draws show no excess dispersion."""
        rng = np.random.default_rng(11)
        n = np.full(240, 5000)
        k = rng.binomial(n, 0.02)

        phi = estimate_overdispersion(k, n)

        assert 1.0 <= phi < 1.2

    def test_clustered_simulation_returns_inflated_factor(self):
        """Test that period-level clustering is detected as overdispersion."""
        rng = np.random.default_rng(11)
        n = np.full(240, 5000)
        # Each month's true rate wanders, mimicking topic bursts and lab batches.
        mean, sd = 0.02, 0.005
        strength = mean * (1 - mean) / sd**2 - 1
        rates = rng.beta(mean * strength, (1 - mean) * strength, size=240)
        k = rng.binomial(n, rates)

        phi = estimate_overdispersion(k, n)

        assert phi > 1.5

    def test_all_zero_series_returns_one(self):
        """Test the degenerate pooled proportion guard."""
        assert estimate_overdispersion(np.zeros(50), np.full(50, 100)) == 1.0

    def test_single_period_returns_one(self):
        """Test the degrees-of-freedom guard."""
        assert estimate_overdispersion(np.array([3]), np.array([100])) == 1.0

    def test_never_returns_below_one(self):
        """Test that underdispersed data still gets a binomial-width band."""
        # Identical counts every month are far too regular for a binomial.
        phi = estimate_overdispersion(np.full(24, 20), np.full(24, 1000))

        assert phi == 1.0

    def test_empty_exposure_returns_one(self):
        """Test that a series with no reference papers is handled."""
        assert estimate_overdispersion(np.zeros(10), np.zeros(10)) == 1.0


@pytest.mark.unit
class TestSmoothedShare:
    """Tests for smoothed_share."""

    def test_thin_month_is_not_reported_as_certainty(self):
        """Test that a 1-of-1 month is shrunk far away from 100%."""
        k = np.concatenate([[1], np.full(30, 50)])
        n = np.concatenate([[1], np.full(30, 500)])

        shrunk = smoothed_share(k, n)

        assert 0.15 < shrunk[0] < 0.40

    def test_well_populated_month_barely_moves(self):
        """Test that shrinkage is negligible when exposure is large."""
        k = np.concatenate([[1], np.full(30, 50)])
        n = np.concatenate([[1], np.full(30, 500)])

        shrunk = smoothed_share(k, n)

        assert shrunk[1] == pytest.approx(0.1, abs=0.002)

    def test_moment_fit_matches_hand_computed_prior(self):
        """Test the exact beta-binomial moment fit on a two-period example.

        Shares (0.2, 0.4) have mean 0.3 and sample variance 0.02. The binomial
        sampling component is mean(p(1-p)/n) = (0.16/10 + 0.24/10)/2 = 0.02, so
        the excess between-period variance is 0.02 - 0.02 = 0. Two periods of ten
        papers each are entirely consistent with one constant rate, so the prior
        mass is the total exposure (20) and the estimates shrink hard toward the
        mean of 0.3: (k + 6) / (n + 20).
        """
        shrunk = smoothed_share(np.array([2, 4]), np.array([10, 10]))

        assert shrunk == pytest.approx([8.0 / 30.0, 10.0 / 30.0])

    def test_sampling_variance_is_subtracted_before_matching(self):
        """A constant true rate observed through thin months must not look dispersed.

        This is the defect the raw-variance estimator has: it credits binomial
        noise to the prior, computes too weak a concentration, and under-shrinks
        precisely the thin months shrinkage exists for.
        """
        rng = np.random.default_rng(7)
        # 24 thin months then 24 fat ones, all drawn from a single true rate.
        n = np.array([40] * 24 + [40_000] * 24)
        k = rng.binomial(n, 0.02)

        shrunk = smoothed_share(k, n)

        raw_thin = k[:24] / n[:24]
        # The raw shares swing wildly around the truth; the shrunk ones must not.
        assert raw_thin.std() > 0.02
        assert shrunk[:24].std() < 0.006
        assert shrunk[:24].mean() == pytest.approx(0.02, abs=0.004)
        # Well-populated months keep their own estimate.
        assert shrunk[24:].mean() == pytest.approx(0.02, abs=0.001)

    def test_degenerate_moments_fall_back_to_pooled(self):
        """Test the fallback when observed shares carry no variance."""
        k = np.array([5, 10, 20])
        n = np.array([100, 200, 400])

        shrunk = smoothed_share(k, n)

        assert np.allclose(shrunk, 0.05)

    def test_prior_strength_overrides_the_moment_fit(self):
        """Test that a heavier prior pulls harder toward the series mean."""
        k = np.concatenate([[1], np.full(30, 50)])
        n = np.concatenate([[1], np.full(30, 500)])

        weak = smoothed_share(k, n, prior_strength=1.0)
        strong = smoothed_share(k, n, prior_strength=50.0)

        assert strong[0] < weak[0]

    def test_zero_exposure_takes_the_prior_mean(self):
        """Test that an empty month yields the prior mean instead of NaN."""
        k = np.concatenate([[0], np.full(30, 50)])
        n = np.concatenate([[0], np.full(30, 500)])

        shrunk = smoothed_share(k, n, prior_strength=10.0)

        assert shrunk[0] == pytest.approx(np.mean(k[1:] / n[1:]), abs=1e-9)

    def test_series_with_no_exposure_returns_zeros(self):
        """Test the guard for a term with no reference papers at all."""
        assert np.all(smoothed_share(np.zeros(5), np.zeros(5)) == 0.0)

    def test_all_zero_series_falls_back_to_pooled(self):
        """Test that a term with no matches anywhere reports a flat zero share."""
        assert np.all(smoothed_share(np.zeros(12), np.full(12, 500)) == 0.0)

    def test_single_period_falls_back_to_pooled(self):
        """Test that one period gives no second moment to fit."""
        assert smoothed_share(np.array([3]), np.array([100])) == pytest.approx([0.03])

    def test_non_positive_prior_strength_falls_back_to_pooled(self):
        """Test that an unusable prior_strength degrades to the pooled share."""
        k = np.array([1, 50, 60])
        n = np.array([10, 500, 600])

        shrunk = smoothed_share(k, n, prior_strength=0.0)

        assert np.allclose(shrunk, 111 / 1110)

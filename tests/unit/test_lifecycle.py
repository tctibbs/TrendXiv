"""Unit tests for adoption-curve fitting and trend classification."""

import math

import numpy as np
import pytest
from scipy.stats import kendalltau, norm

from src.analysis.lifecycle import BassFit, bass_diffusion, fit_lifecycle, mann_kendall

SATURATION = 0.4
GROWTH_RATE = 0.1
INFLECTION = 60.0
N_PERIODS = 120

# The truncation point sits well before the inflection, where a logistic and an
# exponential are indistinguishable in practice.
TRUNCATED_LENGTH = 45


def logistic_share(
    n_periods: int = N_PERIODS,
    saturation: float = SATURATION,
    rate: float = GROWTH_RATE,
    inflection: float = INFLECTION,
) -> np.ndarray:
    """Create per-period increments of a known logistic adoption curve."""
    t = np.arange(n_periods, dtype=float)
    cumulative = saturation / (1.0 + np.exp(-rate * (t - inflection)))
    return np.diff(cumulative, prepend=0.0)


def noisy(share: np.ndarray, scale: float = 0.05, seed: int = 42) -> np.ndarray:
    """Add proportional noise to a share series."""
    rng = np.random.default_rng(seed)
    return share * (1.0 + rng.normal(0.0, scale, size=share.size))


@pytest.mark.unit
class TestFitLifecycle:
    """Tests for fit_lifecycle."""

    def test_recovers_known_saturation_within_five_percent(self):
        """A fully observed logistic must recover its own M."""
        fit = fit_lifecycle(noisy(logistic_share()))

        assert fit.identifiable
        assert fit.saturation == pytest.approx(SATURATION, rel=0.05)

    def test_recovers_known_inflection_and_rate(self):
        """The fitted inflection and rate must match the simulation."""
        fit = fit_lifecycle(noisy(logistic_share()))

        assert fit.inflection_index == pytest.approx(INFLECTION, rel=0.05)
        assert fit.rate == pytest.approx(GROWTH_RATE, rel=0.1)

    def test_prefers_logistic_for_logistic_data(self):
        """AICc must select the model the data was generated from."""
        fit = fit_lifecycle(noisy(logistic_share()))

        assert fit.model == "logistic"
        assert fit.r_squared > 0.99

    def test_reported_aicc_is_the_small_sample_corrected_score(self):
        """AICc = n*ln(RSS/n) + 2k + 2k(k+1)/(n-k-1) with k = 3 params + variance."""
        share = noisy(logistic_share())
        fit = fit_lifecycle(share)

        t = np.arange(len(share), dtype=float)
        predicted = fit.saturation / (1.0 + np.exp(-fit.rate * (t - fit.inflection_index)))
        residual_sum = float(np.sum((np.cumsum(share) - predicted) ** 2))
        n, k = len(share), 4
        expected = n * math.log(residual_sum / n) + 2 * k + 2 * k * (k + 1) / (n - k - 1)

        assert fit.aicc == pytest.approx(expected, rel=1e-9)

    def test_truncated_series_is_not_identifiable(self):
        """A series cut off before its inflection must refuse to report M."""
        fit = fit_lifecycle(noisy(logistic_share())[:TRUNCATED_LENGTH])

        assert fit.identifiable is False
        assert fit.state == "emerging"

    def test_confidence_interval_brackets_the_inflection(self):
        """The reported interval must contain the fitted inflection."""
        fit = fit_lifecycle(noisy(logistic_share()))

        assert fit.inflection_ci is not None
        low, high = fit.inflection_ci
        assert low <= fit.inflection_index <= high

    def test_completed_adoption_is_plateaued(self):
        """A curve observed past its ceiling is done growing."""
        fit = fit_lifecycle(logistic_share())

        assert fit.state == "plateaued"

    def test_mid_curve_adoption_is_accelerating(self):
        """Just past the inflection the curve is still climbing hard."""
        fit = fit_lifecycle(logistic_share()[:75])

        assert fit.identifiable
        assert fit.state == "accelerating"

    def test_falling_share_is_declining(self):
        """A share trending down is declining whatever the curve fit says."""
        fit = fit_lifecycle(np.linspace(0.02, 0.001, 60))

        assert fit.state == "declining"

    def test_gompertz_data_selects_gompertz(self):
        """AICc must prefer the asymmetric curve for asymmetric data."""
        t = np.arange(N_PERIODS, dtype=float)
        cumulative = SATURATION * np.exp(-np.exp(-GROWTH_RATE * (t - INFLECTION)))
        fit = fit_lifecycle(np.diff(cumulative, prepend=0.0))

        assert fit.model == "gompertz"
        assert fit.saturation == pytest.approx(SATURATION, rel=0.05)

    def test_gompertz_just_past_inflection_is_emerging(self):
        """The Gompertz inflection sits at M/e, well short of half adoption."""
        t = np.arange(N_PERIODS, dtype=float)
        cumulative = SATURATION * np.exp(-np.exp(-GROWTH_RATE * (t - INFLECTION)))
        fit = fit_lifecycle(np.diff(cumulative, prepend=0.0)[:63])

        assert fit.identifiable
        assert fit.state == "emerging"

    def test_one_shot_spike_has_no_explained_variance(self):
        """A flat cumulative curve has nothing for a model to explain."""
        fit = fit_lifecycle(np.array([0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))

        assert fit.r_squared == 0.0
        assert fit.identifiable is False

    def test_short_series_is_not_fitted(self):
        """Too few points to support four parameters means no model."""
        fit = fit_lifecycle(np.array([0.01, 0.02, 0.03]))

        assert fit.model == "none"
        assert fit.identifiable is False
        assert fit.saturation is None
        assert math.isinf(fit.aicc)

    def test_all_zero_series_is_not_fitted(self):
        """A term that never appears has no curve to fit."""
        fit = fit_lifecycle(np.zeros(60))

        assert fit.model == "none"
        assert fit.state == "indeterminate"

    def test_explicit_time_axis_shifts_the_inflection(self):
        """Reporting is in units of t, not in period offsets."""
        share = noisy(logistic_share())
        offset = 1000.0

        fit = fit_lifecycle(share, t=np.arange(len(share), dtype=float) + offset)

        assert fit.inflection_index == pytest.approx(INFLECTION + offset, rel=1e-3)

    def test_misaligned_time_axis_raises(self):
        """t must describe the same periods as share."""
        with pytest.raises(ValueError, match="aligned"):
            fit_lifecycle(np.zeros(10), t=np.arange(9, dtype=float))

    def test_two_dimensional_share_raises(self):
        """The exposure share is a single series."""
        with pytest.raises(ValueError, match="1-D"):
            fit_lifecycle(np.zeros((3, 3)))


@pytest.mark.unit
class TestBassDiffusion:
    """Tests for bass_diffusion."""

    def test_recovers_known_parameters(self):
        """The fit must return the p and q it was simulated from."""
        p, q, m = 0.01, 0.3, 0.5
        share = _bass_share(p, q, m)

        fit = bass_diffusion(share)

        assert fit is not None
        assert fit.p == pytest.approx(p, rel=0.1)
        assert fit.q == pytest.approx(q, rel=0.1)
        assert fit.saturation == pytest.approx(m, rel=0.05)

    def test_reports_peak_when_imitation_dominates(self):
        """With q > p the peak date is the analytical ln(q/p)/(p+q).

        The tolerance is tight on purpose: the noiseless curve is recovered to
        machine precision, so a peak divided by q - p instead of p + q lands
        7% away and must fail here.
        """
        p, q = 0.01, 0.3
        expected_peak = math.log(q / p) / (p + q)

        fit = bass_diffusion(_bass_share(p, q, 0.5))

        assert fit is not None
        assert fit.peak_index == pytest.approx(expected_peak, rel=1e-6)

    def test_peak_is_the_argmax_of_the_fitted_adoption_rate(self):
        """The reported peak must be where the fitted curve adopts fastest."""
        p, q = 0.02, 0.25
        fit = bass_diffusion(_bass_share(p, q, 0.5))
        assert fit is not None

        grid = np.linspace(0.0, 59.0, 600_001)
        decay = np.exp(-(fit.p + fit.q) * grid)
        cumulative = fit.saturation * (1.0 - decay) / (1.0 + (fit.q / fit.p) * decay)

        assert fit.peak_index == pytest.approx(grid[np.argmax(np.diff(cumulative))], rel=1e-4)

    def test_withholds_peak_when_innovation_dominates(self):
        """p > q makes the peak formula negative, so no date is reported."""
        fit = bass_diffusion(_bass_share(0.3, 0.02, 0.5))

        assert fit is not None
        assert fit.p > fit.q
        assert fit.peak_index is None

    def test_short_series_returns_none(self):
        """Three points cannot support a three-parameter fit."""
        assert bass_diffusion(np.array([0.01, 0.02, 0.03])) is None

    def test_empty_adoption_returns_none(self):
        """A series with no adoption has nothing to diffuse."""
        assert bass_diffusion(np.zeros(60)) is None

    def test_is_frozen_dataclass(self):
        """Fits are immutable so pipeline stages cannot mutate them."""
        fit = BassFit(p=0.01, q=0.3, saturation=0.5, peak_index=11.0, r_squared=0.99)

        with pytest.raises(AttributeError):
            fit.p = 0.2


@pytest.mark.unit
class TestMannKendall:
    """Tests for mann_kendall."""

    def test_linear_ramp_is_increasing(self):
        """A pure ramp is the least ambiguous increasing series there is."""
        result = mann_kendall(np.arange(50, dtype=float))

        assert result.trend == "increasing"
        assert result.p_value < 0.01
        assert result.tau == pytest.approx(1.0)
        assert result.slope == pytest.approx(1.0)

    def test_white_noise_has_no_trend(self):
        """Trendless noise must not be flagged."""
        rng = np.random.default_rng(11)

        result = mann_kendall(rng.normal(size=200))

        assert result.trend == "no trend"
        assert result.p_value > 0.05

    def test_decreasing_ramp_is_decreasing(self):
        """A falling ramp must be signed the other way."""
        result = mann_kendall(np.linspace(1.0, 0.0, 40))

        assert result.trend == "decreasing"
        assert result.tau == pytest.approx(-1.0)
        assert result.slope < 0

    def test_tau_matches_scipy_on_an_untied_series(self):
        """The statistic must agree with the reference implementation."""
        rng = np.random.default_rng(3)
        values = rng.normal(size=60)

        result = mann_kendall(values)
        reference = kendalltau(np.arange(60), values)

        assert result.tau == pytest.approx(reference.statistic)

    def test_tie_correction_matches_the_tau_b_reference(self):
        """A heavily tied step series must still match scipy's tau-b."""
        values = np.repeat([1.0, 2.0], 25)

        result = mann_kendall(values)

        assert result.trend == "increasing"
        assert result.tau == pytest.approx(kendalltau(np.arange(50), values).statistic)

    def test_p_value_uses_the_continuity_and_tie_corrections(self):
        """Hand-computed p-value for a 25/25 step, ties and all.

        Every one of the 625 cross-group pairs is concordant and every
        within-group pair is tied, so S = 625 and the two runs of 25 subtract
        2*25*24*55 from the variance. Dropping either correction moves this
        p-value by a factor of 100 (ties) or 5% (continuity).
        """
        values = np.repeat([1.0, 2.0], 25)
        variance = (50 * 49 * 105 - 2 * 25 * 24 * 55) / 18.0
        expected = 2.0 * norm.sf((625 - 1) / math.sqrt(variance))

        result = mann_kendall(values)

        assert variance == pytest.approx(10_625.0)
        assert result.p_value == pytest.approx(expected, rel=1e-9)

    def test_constant_series_has_no_trend(self):
        """Zero variance leaves no trend to detect."""
        result = mann_kendall(np.full(30, 0.05))

        assert result.trend == "no trend"
        assert result.p_value == 1.0
        assert result.slope == 0.0

    def test_too_short_series_has_no_trend(self):
        """Two points are not evidence of a monotonic trend."""
        result = mann_kendall(np.array([1.0, 5.0]))

        assert result.trend == "no trend"
        assert result.p_value == 1.0

    def test_stricter_alpha_withholds_a_weak_trend(self):
        """The significance level must actually gate the verdict."""
        rng = np.random.default_rng(5)
        values = np.arange(30, dtype=float) * 0.01 + rng.normal(0, 0.2, size=30)

        assert mann_kendall(values, alpha=0.5).trend == "increasing"
        assert mann_kendall(values, alpha=1e-6).trend == "no trend"


def _bass_share(p: float, q: float, m: float, n_periods: int = 60) -> np.ndarray:
    """Create per-period increments of a known Bass adoption curve."""
    t = np.arange(n_periods, dtype=float)
    decay = np.exp(-(p + q) * t)
    cumulative = m * (1.0 - decay) / (1.0 + (q / p) * decay)
    return np.diff(cumulative, prepend=0.0)

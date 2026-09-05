"""Adoption-curve fitting and trend classification for term share series."""

import math
import warnings
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.optimize import OptimizeWarning, curve_fit
from scipy.stats import norm

# Number of free parameters in every curve here, plus one for the residual
# variance, which AICc counts.
_N_PARAMETERS = 4

# Two observations beyond the parameter count keeps the AICc correction finite.
MIN_POINTS_FOR_FIT = _N_PARAMETERS + 2
MIN_POINTS_FOR_TREND = 3

# Keeps exponentials finite while the optimiser probes extreme parameters.
_EXP_LIMIT = 500.0

# Share of the fitted saturation that separates the lifecycle states.
_ACCELERATING_FRACTION = 0.5
_PLATEAU_FRACTION = 0.9


@dataclass(frozen=True)
class MannKendallResult:
    """Outcome of a non-parametric monotonic trend test.

    Attributes:
        tau: Kendall rank correlation between value and time
        p_value: Two-sided p-value from the normal approximation
        trend: One of 'increasing', 'decreasing', or 'no trend'
        slope: Theil-Sen slope in value units per time unit
    """

    tau: float
    p_value: float
    trend: str
    slope: float


@dataclass(frozen=True)
class LifecycleFit:
    """Best-fitting adoption curve for a cumulative share series.

    Attributes:
        model: Winning model, one of 'logistic', 'gompertz', or 'none'
        saturation: Fitted ceiling M of the cumulative share, None if unfitted
        inflection_index: Time of the inflection point in units of t
        inflection_ci: 95% confidence interval on the inflection, None if the
            covariance was not estimable
        rate: Fitted growth rate r
        aicc: Small-sample corrected AIC of the winning model
        state: Lifecycle state, one of 'emerging', 'accelerating', 'plateaued',
            'declining', or 'indeterminate'
        identifiable: Whether the inflection point lies inside the observed
            window. When False the caller must not display saturation.
        r_squared: Fraction of cumulative-share variance explained
    """

    model: str
    saturation: float | None
    inflection_index: float | None
    inflection_ci: tuple[float, float] | None
    rate: float | None
    aicc: float
    state: str
    identifiable: bool
    r_squared: float


@dataclass(frozen=True)
class BassFit:
    """Bass diffusion parameters for a cumulative share series.

    Attributes:
        p: Coefficient of innovation, the externally driven adoption rate
        q: Coefficient of imitation, the word-of-mouth adoption rate
        saturation: Fitted ceiling M of the cumulative share
        peak_index: Time of peak adoption in units of t, None when innovation
            dominates imitation and the peak date is meaningless
        r_squared: Fraction of cumulative-share variance explained
    """

    p: float
    q: float
    saturation: float
    peak_index: float | None
    r_squared: float


def fit_lifecycle(share: np.ndarray, t: np.ndarray | None = None) -> LifecycleFit:
    """Fit logistic and Gompertz adoption curves and classify the lifecycle.

    Both models are fitted to the CUMULATIVE share and compared by AICc; the
    winner is reported. Saturation is only trustworthy once the inflection
    point has actually been observed, so a fit whose inflection falls outside
    the window is returned with identifiable=False and must not be displayed.

    Args:
        share: Per-period share of the reference set, k / n
        t: Time coordinate aligned to share. Defaults to period indices.

    Returns:
        The winning fit, or a 'none' model when neither curve converges

    Raises:
        ValueError: If share is not 1-D or t is misaligned
    """
    share_array, time = _validate_series(share, t)
    trend = mann_kendall(share_array)

    if len(share_array) < MIN_POINTS_FOR_FIT:
        return _unfitted(trend)

    cumulative = np.cumsum(share_array)

    candidates = [
        _fit_curve(_logistic, time, cumulative, "logistic"),
        _fit_curve(_gompertz, time, cumulative, "gompertz"),
    ]
    fitted = [candidate for candidate in candidates if candidate is not None]
    if not fitted:
        return _unfitted(trend)

    model, saturation, rate, inflection, inflection_ci, aicc, r_squared = min(
        fitted, key=lambda candidate: candidate[5]
    )

    identifiable = bool(time[0] < inflection < time[-1])
    state = _classify(trend, identifiable, cumulative[-1], saturation)

    return LifecycleFit(
        model=model,
        saturation=saturation,
        inflection_index=inflection,
        inflection_ci=inflection_ci,
        rate=rate,
        aicc=aicc,
        state=state,
        identifiable=identifiable,
        r_squared=r_squared,
    )


def bass_diffusion(share: np.ndarray, t: np.ndarray | None = None) -> BassFit | None:
    """Fit the Bass diffusion model to a cumulative share series.

    Args:
        share: Per-period share of the reference set, k / n
        t: Time coordinate aligned to share. Defaults to period indices.

    Returns:
        The fitted parameters, or None if the fit does not converge

    Raises:
        ValueError: If share is not 1-D or t is misaligned
    """
    share_array, time = _validate_series(share, t)
    if len(share_array) < MIN_POINTS_FOR_FIT:
        return None

    cumulative = np.cumsum(share_array)
    ceiling = float(cumulative[-1])
    if ceiling <= 0:
        return None

    origin = float(time[0])
    guess = [ceiling * 1.5, 0.01, 0.3]
    bounds = ([ceiling * 1e-3, 1e-9, 1e-9], [ceiling * 1e4, 1.0, 10.0])

    def model(time_values: np.ndarray, m: float, p: float, q: float) -> np.ndarray:
        return _bass_cumulative(time_values, m, p, q, origin)

    parameters, _ = _least_squares(model, time, cumulative, guess, bounds)
    if parameters is None:
        return None

    m, p, q = (float(value) for value in parameters)

    # ln(q/p) is negative when innovation dominates imitation, which dates the
    # adoption peak before the series begins. That case is exactly what this
    # diagnostic exists to detect, so report no date rather than a fake one.
    peak_index = origin + math.log(q / p) / (p + q) if q > p else None

    return BassFit(
        p=p,
        q=q,
        saturation=m,
        peak_index=peak_index,
        r_squared=_r_squared(cumulative, model(time, m, p, q)),
    )


def mann_kendall(values: np.ndarray, alpha: float = 0.05) -> MannKendallResult:
    """Test a series for a monotonic trend with the Mann-Kendall statistic.

    Rank-based, so it survives the non-normal heteroscedastic residuals these
    share series actually have. Uses the normal approximation with the standard
    correction for tied values.

    Args:
        values: Series to test, ordered in time
        alpha: Significance level for declaring a trend

    Returns:
        The test statistic, its p-value, the trend direction, and the
        Theil-Sen slope
    """
    series = np.asarray(values, dtype=float).ravel()
    n = len(series)
    if n < MIN_POINTS_FOR_TREND:
        return MannKendallResult(tau=0.0, p_value=1.0, trend="no trend", slope=0.0)

    differences = series[np.newaxis, :] - series[:, np.newaxis]
    upper = np.triu_indices(n, k=1)
    s = float(np.sum(np.sign(differences[upper])))

    _, tie_counts = np.unique(series, return_counts=True)
    tie_correction = float(np.sum(tie_counts * (tie_counts - 1) * (2 * tie_counts + 5)))
    variance = (n * (n - 1) * (2 * n + 5) - tie_correction) / 18.0

    if variance <= 0:
        return MannKendallResult(tau=0.0, p_value=1.0, trend="no trend", slope=0.0)

    # The continuity correction pulls S one unit toward zero before scaling.
    z = (s - np.sign(s)) / math.sqrt(variance)
    p_value = float(2.0 * norm.sf(abs(z)))

    # Time is untied, so only the value ties enter the tau-b denominator.
    pairs = n * (n - 1) / 2.0
    value_ties = float(np.sum(tie_counts * (tie_counts - 1) / 2.0))
    tau = s / math.sqrt((pairs - value_ties) * pairs) if pairs > value_ties else 0.0

    if p_value < alpha and s > 0:
        trend = "increasing"
    elif p_value < alpha and s < 0:
        trend = "decreasing"
    else:
        trend = "no trend"

    return MannKendallResult(
        tau=float(tau),
        p_value=p_value,
        trend=trend,
        slope=_theil_sen_slope(series),
    )


def _validate_series(share: np.ndarray, t: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    """Coerce a share series and its time axis to aligned float arrays.

    Args:
        share: Per-period share of the reference set
        t: Time coordinate aligned to share, or None for period indices

    Returns:
        The share array and its time coordinate

    Raises:
        ValueError: If share is not 1-D or t is misaligned
    """
    share_array = np.asarray(share, dtype=float)
    if share_array.ndim != 1:
        raise ValueError("share must be a 1-D array")

    if t is None:
        return share_array, np.arange(len(share_array), dtype=float)

    time = np.asarray(t, dtype=float)
    if time.shape != share_array.shape:
        raise ValueError("t must be aligned to share")

    return share_array, time


def _logistic(t: np.ndarray, m: float, r: float, t0: float) -> np.ndarray:
    """Evaluate the logistic curve.

    Args:
        t: Time coordinate
        m: Saturation level
        r: Growth rate
        t0: Inflection time, where the curve reaches M / 2

    Returns:
        Curve values at t
    """
    exponent = np.clip(-r * (t - t0), -_EXP_LIMIT, _EXP_LIMIT)
    return m / (1.0 + np.exp(exponent))


def _gompertz(t: np.ndarray, m: float, r: float, t0: float) -> np.ndarray:
    """Evaluate the Gompertz curve.

    Args:
        t: Time coordinate
        m: Saturation level
        r: Growth rate
        t0: Inflection time, where the curve reaches M / e

    Returns:
        Curve values at t
    """
    inner = np.clip(-r * (t - t0), -_EXP_LIMIT, _EXP_LIMIT)
    outer = np.clip(-np.exp(inner), -_EXP_LIMIT, _EXP_LIMIT)
    return m * np.exp(outer)


def _bass_cumulative(t: np.ndarray, m: float, p: float, q: float, origin: float) -> np.ndarray:
    """Evaluate cumulative Bass adoption.

    Args:
        t: Time coordinate
        m: Saturation level
        p: Coefficient of innovation
        q: Coefficient of imitation
        origin: Time at which diffusion starts

    Returns:
        Curve values at t
    """
    exponent = np.clip(-(p + q) * (t - origin), -_EXP_LIMIT, _EXP_LIMIT)
    decay = np.exp(exponent)
    return m * (1.0 - decay) / (1.0 + (q / p) * decay)


def _fit_curve(
    model: Callable[..., np.ndarray],
    time: np.ndarray,
    cumulative: np.ndarray,
    name: str,
) -> tuple[str, float, float, float, tuple[float, float] | None, float, float] | None:
    """Fit one adoption curve and score it.

    Args:
        model: Curve function taking (t, m, r, t0)
        time: Time coordinate
        cumulative: Cumulative share to fit
        name: Model name to report

    Returns:
        Name, saturation, rate, inflection, inflection CI, AICc, and R-squared,
        or None if the fit does not converge
    """
    span = float(time[-1] - time[0]) or 1.0
    ceiling = float(cumulative[-1])
    if ceiling <= 0:
        return None

    guess = [ceiling * 1.5, 4.0 / span, float(time[0]) + span / 2.0]
    bounds = (
        [ceiling * 1e-3, 1e-6, float(time[0]) - 10.0 * span],
        [ceiling * 1e4, 100.0 / span, float(time[-1]) + 10.0 * span],
    )

    parameters, covariance = _least_squares(model, time, cumulative, guess, bounds)
    if parameters is None:
        return None

    m, r, t0 = (float(value) for value in parameters)
    predicted = model(time, m, r, t0)
    residual_sum = float(np.sum((cumulative - predicted) ** 2))

    return (
        name,
        m,
        r,
        t0,
        _inflection_interval(t0, covariance),
        _aicc(residual_sum, len(cumulative)),
        _r_squared(cumulative, predicted),
    )


def _least_squares(
    model: Callable[..., np.ndarray],
    time: np.ndarray,
    cumulative: np.ndarray,
    guess: list[float],
    bounds: tuple[list[float], list[float]],
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Run curve_fit, converting failures into a None result.

    Args:
        model: Curve function
        time: Time coordinate
        cumulative: Cumulative share to fit
        guess: Initial parameter values
        bounds: Lower and upper parameter bounds

    Returns:
        Fitted parameters and their covariance, or (None, None) on failure
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", OptimizeWarning)
            parameters, covariance = curve_fit(
                model,
                time,
                cumulative,
                p0=guess,
                bounds=bounds,
                maxfev=20000,
            )
    except (RuntimeError, ValueError):
        return None, None

    if not np.all(np.isfinite(parameters)):
        return None, None

    return parameters, covariance


def _inflection_interval(t0: float, covariance: np.ndarray | None) -> tuple[float, float] | None:
    """Build a 95% confidence interval for the inflection time.

    Args:
        t0: Fitted inflection time
        covariance: Parameter covariance matrix, inflection last

    Returns:
        Lower and upper bounds, or None if the covariance is not estimable
    """
    if covariance is None or not np.all(np.isfinite(covariance)):
        return None

    variance = float(covariance[2, 2])
    if variance < 0:
        return None

    half_width = 1.96 * math.sqrt(variance)
    return (t0 - half_width, t0 + half_width)


def _aicc(residual_sum: float, n: int) -> float:
    """Compute the small-sample corrected AIC of a least-squares fit.

    Args:
        residual_sum: Residual sum of squares
        n: Number of observations

    Returns:
        AICc of the fit
    """
    # MIN_POINTS_FOR_FIT keeps this denominator positive.
    penalty = 2 * _N_PARAMETERS + (2 * _N_PARAMETERS * (_N_PARAMETERS + 1)) / (
        n - _N_PARAMETERS - 1
    )
    return n * math.log(max(residual_sum, 1e-300) / n) + penalty


def _r_squared(observed: np.ndarray, predicted: np.ndarray) -> float:
    """Compute the coefficient of determination.

    Args:
        observed: Observed values
        predicted: Fitted values

    Returns:
        R-squared, or 0.0 for a series with no variance
    """
    total = float(np.sum((observed - observed.mean()) ** 2))
    if total <= 0:
        return 0.0

    return 1.0 - float(np.sum((observed - predicted) ** 2)) / total


def _theil_sen_slope(series: np.ndarray) -> float:
    """Compute the Theil-Sen median slope against the period index.

    Args:
        series: Series ordered in time

    Returns:
        Median pairwise slope in value units per period
    """
    n = len(series)
    rows, cols = np.triu_indices(n, k=1)
    slopes = (series[cols] - series[rows]) / (cols - rows)

    return float(np.median(slopes))


def _classify(
    trend: MannKendallResult,
    identifiable: bool,
    reached: float,
    saturation: float,
) -> str:
    """Place a fitted series on the lifecycle clock.

    Args:
        trend: Mann-Kendall result for the per-period share
        identifiable: Whether the inflection point was observed
        reached: Cumulative share at the end of the window
        saturation: Fitted saturation level

    Returns:
        The lifecycle state
    """
    if trend.trend == "decreasing":
        return "declining"

    if not identifiable:
        # Saturation is unusable before the inflection, so the only honest call
        # is "still climbing" when the share is rising and nothing otherwise.
        return "emerging" if trend.trend == "increasing" else "indeterminate"

    fraction = reached / saturation if saturation > 0 else 0.0
    if fraction >= _PLATEAU_FRACTION:
        return "plateaued"
    if fraction >= _ACCELERATING_FRACTION:
        return "accelerating"

    return "emerging"


def _unfitted(trend: MannKendallResult) -> LifecycleFit:
    """Build the result used when no curve could be fitted.

    Args:
        trend: Mann-Kendall result for the per-period share

    Returns:
        A LifecycleFit with model 'none' and no reportable parameters
    """
    return LifecycleFit(
        model="none",
        saturation=None,
        inflection_index=None,
        inflection_ci=None,
        rate=None,
        aicc=math.inf,
        state="declining" if trend.trend == "decreasing" else "indeterminate",
        identifiable=False,
        r_squared=0.0,
    )

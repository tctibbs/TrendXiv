"""Seasonal decomposition of monthly arXiv series.

arXiv submissions follow conference deadlines (NeurIPS/ICML/CVPR/ICLR), so a
monthly series carries a repeating within-year pattern on top of its trend.
These helpers split that pattern out so month-over-month comparisons reflect
behaviour rather than the calendar.

All functions are pure: numpy in, numpy out, no I/O.
"""

import calendar
from dataclasses import dataclass

import numpy as np
from statsmodels.tsa.seasonal import STL

# Guards log() against exact zeros in sparse early-history months. Counts and
# shares of interest are far larger than this, so the bias is negligible.
_LOG_EPSILON = 1e-9

# A flat series decomposes into components of size ~1e-16; below this share of
# the series magnitude, component variance is rounding error, not seasonality.
_FLAT_VARIANCE_TOLERANCE = 1e-18

_MONTHS_PER_YEAR = 12


@dataclass(frozen=True)
class SeasonalDecomposition:
    """Result of an STL decomposition of a monthly series.

    When ``used_log`` is True the ``trend``, ``seasonal`` and ``resid``
    components live on the log scale (their seasonal effect is multiplicative
    on the original scale); ``adjusted`` is always on the original scale so it
    can be plotted against the input directly.

    Attributes:
        trend: Slow-moving level of the series
        seasonal: Repeating within-year component
        resid: Remainder after trend and seasonal are removed
        adjusted: Seasonally adjusted series on the ORIGINAL scale
        period: Number of observations per seasonal cycle
        used_log: Whether the decomposition ran on log values
        strength: Hyndman seasonal strength in [0, 1]
    """

    trend: np.ndarray
    seasonal: np.ndarray
    resid: np.ndarray
    adjusted: np.ndarray
    period: int
    used_log: bool
    strength: float


def stl_decompose(
    values: np.ndarray,
    period: int = 12,
    robust: bool = True,
    log: bool = True,
) -> SeasonalDecomposition:
    """Split a monthly series into trend, seasonal and residual components.

    The decomposition runs on log values by default: arXiv series span a ~100x
    dynamic range, over which a deadline effect scales with the level rather
    than adding a fixed number of papers. Series containing non-positive values
    fall back to an additive decomposition, recorded in ``used_log``.

    Args:
        values: Monthly observations, ordered oldest to newest
        period: Observations per seasonal cycle
        robust: Whether to downweight outliers in the STL fit
        log: Whether to decompose on log values when the series allows it

    Returns:
        SeasonalDecomposition with components and the seasonally adjusted series

    Raises:
        ValueError: If the series is shorter than two full periods, or contains
            a non-finite value
    """
    series = np.asarray(values, dtype=float)

    if period < 2:
        raise ValueError(f"period must be at least 2, got {period}")

    if series.size < 2 * period:
        raise ValueError(
            f"STL needs at least two full periods: got {series.size} observations "
            f"for period {period}, need {2 * period}"
        )

    # STL turns a single NaN into an all-NaN decomposition, and the strength of
    # an all-NaN decomposition computes to 0.0, i.e. a confident "not seasonal"
    # verdict on a series nobody actually decomposed. Refuse instead.
    if not np.all(np.isfinite(series)):
        raise ValueError("values must all be finite: found NaN or infinity")

    used_log = bool(log) and bool(np.all(series > 0.0))
    work = np.log(series + _LOG_EPSILON) if used_log else series

    fit = STL(work, period=period, robust=robust).fit()
    trend = np.asarray(fit.trend, dtype=float)
    seasonal = np.asarray(fit.seasonal, dtype=float)
    resid = np.asarray(fit.resid, dtype=float)

    deseasonalized = work - seasonal
    adjusted = np.exp(deseasonalized) - _LOG_EPSILON if used_log else deseasonalized

    return SeasonalDecomposition(
        trend=trend,
        seasonal=seasonal,
        resid=resid,
        adjusted=adjusted,
        period=period,
        used_log=used_log,
        strength=_seasonal_strength(seasonal, resid, float(np.mean(np.square(work)))),
    )


def month_of_year_index(
    values: np.ndarray,
    periods: list[str],
    log: bool = True,
) -> dict[int, float]:
    """Build the deadline fingerprint: a multiplicative index per calendar month.

    The index comes from the STL seasonal component rather than a group-by-month
    mean of the raw series, so long-run growth cannot leak into the fingerprint
    (arXiv grew ~102x since 1992, which would otherwise inflate whichever months
    happen to sit late in the sample).

    Args:
        values: Monthly observations, ordered oldest to newest
        periods: YYYY-MM strings aligned element-wise to values
        log: Whether to decompose on log values when the series allows it

    Returns:
        Mapping of month number 1..12 to a multiplicative index, normalised so
        the geometric mean across the twelve months is exactly 1.0

    Raises:
        ValueError: If periods and values differ in length, a period is
            malformed, the periods are not consecutive months, or the series is
            shorter than two full years
    """
    series = np.asarray(values, dtype=float)
    ordinals = _month_ordinals(periods)

    if ordinals.size != series.size:
        raise ValueError(
            f"values and periods must align: got {series.size} values and {ordinals.size} periods"
        )

    # STL reads array position as time. A gap or a duplicate month would shift
    # every later observation into the wrong calendar slot, smearing the
    # fingerprint across neighbouring months instead of failing.
    if ordinals.size > 1 and not np.all(np.diff(ordinals) == 1):
        raise ValueError("periods must be consecutive months with no gaps or duplicates")

    months = ordinals % _MONTHS_PER_YEAR + 1
    decomposition = stl_decompose(series, period=_MONTHS_PER_YEAR, log=log)

    # Work in log space so averaging across years is a geometric mean, matching
    # the multiplicative reading of the index.
    if decomposition.used_log:
        log_factors = decomposition.seasonal
    else:
        level = float(np.mean(series))
        if level <= 0.0:
            raise ValueError("cannot build a multiplicative index from a non-positive series")
        log_factors = np.log(np.clip(1.0 + decomposition.seasonal / level, _LOG_EPSILON, None))

    # Every month is guaranteed to be observed: the periods are consecutive and
    # stl_decompose already required at least two full years of them.
    month_means = np.array(
        [float(np.mean(log_factors[months == month])) for month in range(1, _MONTHS_PER_YEAR + 1)]
    )

    normalized = np.exp(month_means - month_means.mean())
    return {month: float(normalized[month - 1]) for month in range(1, _MONTHS_PER_YEAR + 1)}


def trading_day_adjust(values: np.ndarray, periods: list[str]) -> np.ndarray:
    """Remove the weekday-count artifact from monthly totals.

    Months hold 20 to 23 weekdays, a ~5% swing in submissions that reflects the
    calendar rather than any change in research activity. Each observation is
    rescaled to the mean weekday count of the given months, so the overall level
    of the series is preserved.

    Args:
        values: Monthly observations, ordered oldest to newest
        periods: YYYY-MM strings aligned element-wise to values

    Returns:
        Array of the same shape with the weekday-count effect divided out

    Raises:
        ValueError: If periods and values differ in length, or a period is malformed
    """
    series = np.asarray(values, dtype=float)
    parsed = [_parse_period(period) for period in periods]

    if len(parsed) != series.size:
        raise ValueError(
            f"values and periods must align: got {series.size} values and {len(parsed)} periods"
        )

    if series.size == 0:
        return series.copy()

    weekdays = np.array([_weekday_count(year, month) for year, month in parsed], dtype=float)
    return series * (weekdays.mean() / weekdays)


def _seasonal_strength(seasonal: np.ndarray, resid: np.ndarray, scale: float) -> float:
    """Compute the Hyndman seasonal strength of a decomposition.

    Args:
        seasonal: Seasonal component
        resid: Residual component
        scale: Mean squared magnitude of the decomposed series

    Returns:
        Strength in [0, 1]: 0 when the seasonal component explains nothing
    """
    detrended_variance = float(np.var(seasonal + resid))
    if detrended_variance <= _FLAT_VARIANCE_TOLERANCE * max(scale, 1.0):
        return 0.0

    return max(0.0, 1.0 - float(np.var(resid)) / detrended_variance)


def _weekday_count(year: int, month: int) -> int:
    """Count Monday-to-Friday days in a calendar month.

    Args:
        year: Four-digit year
        month: Month number 1..12

    Returns:
        Number of weekdays in the month
    """
    first_weekday, days_in_month = calendar.monthrange(year, month)
    return sum(1 for offset in range(days_in_month) if (first_weekday + offset) % 7 < 5)


def _parse_period(period: str) -> tuple[int, int]:
    """Parse a YYYY-MM period label.

    Args:
        period: Period label such as "2024-05"

    Returns:
        Tuple of year and month

    Raises:
        ValueError: If the label is not a well-formed YYYY-MM string
    """
    parts = str(period).split("-")
    if len(parts) != 2:
        raise ValueError(f"period must be formatted YYYY-MM, got {period!r}")

    try:
        year, month = int(parts[0]), int(parts[1])
    except ValueError as exc:
        raise ValueError(f"period must be formatted YYYY-MM, got {period!r}") from exc

    if not 1 <= month <= _MONTHS_PER_YEAR:
        raise ValueError(f"month must be in 1..12, got {period!r}")

    return year, month


def _month_ordinals(periods: list[str]) -> np.ndarray:
    """Convert YYYY-MM labels to a running month count.

    Args:
        periods: Period labels such as ["2024-04", "2024-05"]

    Returns:
        Array of months elapsed since year zero, so consecutive calendar months
        differ by exactly one and ``ordinal % 12 + 1`` recovers the month number
    """
    return np.array(
        [year * _MONTHS_PER_YEAR + month - 1 for year, month in map(_parse_period, periods)],
        dtype=int,
    )

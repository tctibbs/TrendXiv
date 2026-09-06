"""Kleinberg burst detection over binomial exposure counts."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.special import gammaln

MIN_PERIODS_FOR_BURST = 4

# Keeps log(p) and log(1 - p) finite when a state rate is pushed to a boundary.
_PROBABILITY_EPSILON = 1e-12

# Largest exponent that keeps exp() inside float64.
_MAX_RATE_EXPONENT = 700.0


@dataclass(frozen=True)
class Burst:
    """A dated interval where the matched share sat above its baseline rate.

    Attributes:
        start: Period label of the first period in the burst
        end: Period label of the last period in the burst
        level: Highest burst state reached in the interval (1 or greater)
        weight: Total log-likelihood gain of the burst state over baseline
        start_index: Index of the first period in the burst
        end_index: Index of the last period in the burst
    """

    start: str
    end: str
    level: int
    weight: float
    start_index: int
    end_index: int


def kleinberg_bursts(
    k: np.ndarray,
    n: np.ndarray,
    periods: Sequence[str],
    s: float = 2.0,
    gamma: float = 1.0,
    n_states: int = 4,
) -> list[Burst]:
    """Detect bursts in a share series with Kleinberg's batched automaton.

    Models the series as a hidden automaton whose states emit matches at
    geometrically increasing rates p_i = p0 * s**i, where p0 is the pooled
    baseline share. Viterbi decoding trades the fit cost of each state against
    a cost for moving to a higher one, so a burst has to be both large and
    sustained to survive.

    Args:
        k: Matched papers per period
        n: Reference-set papers per period, aligned to k
        periods: Period labels aligned to k, used for the returned dates
        s: Rate multiplier between consecutive states
        gamma: Weight of the state-transition cost
        n_states: Number of automaton states, including the baseline state

    Returns:
        Bursts ordered by start index, empty if the series is too short or
        never leaves the baseline state

    Raises:
        ValueError: If the inputs are misaligned, negative, or k exceeds n
    """
    k_array = np.asarray(k, dtype=np.int64)
    n_array = np.asarray(n, dtype=np.int64)

    if k_array.shape != n_array.shape or k_array.ndim != 1:
        raise ValueError("k and n must be 1-D arrays of the same length")
    if len(periods) != len(k_array):
        raise ValueError("periods must be aligned to k and n")
    if n_states < 2:
        raise ValueError("n_states must be at least 2")
    if s <= 1.0:
        raise ValueError("s must be greater than 1 for states to separate")
    if np.any(k_array < 0) or np.any(n_array < 0):
        raise ValueError("k and n must be non-negative")
    if np.any(k_array > n_array):
        raise ValueError("k must not exceed n in any period")

    n_periods = len(k_array)
    if n_periods < MIN_PERIODS_FOR_BURST:
        return []

    rates = _state_rates(k_array, n_array, s, n_states)
    fit_costs = _fit_costs(k_array, n_array, rates)
    states = _viterbi(fit_costs, gamma, n_periods)

    return _runs_to_bursts(states, fit_costs, periods)


def _state_rates(k: np.ndarray, n: np.ndarray, s: float, n_states: int) -> np.ndarray:
    """Build the expected share of each automaton state.

    Args:
        k: Matched papers per period
        n: Reference-set papers per period
        s: Rate multiplier between consecutive states
        n_states: Number of automaton states

    Returns:
        Array of length n_states with rates strictly inside (0, 1)
    """
    total_n = int(n.sum())
    baseline = float(k.sum()) / total_n if total_n > 0 else 0.0

    # Capping the exponent instead of writing s ** i keeps a large n_states from
    # overflowing to inf, which a zero baseline would turn into nan and poison
    # every fit cost with. States that far above the baseline clip to 1 anyway.
    exponents = np.minimum(np.arange(n_states, dtype=float) * np.log(s), _MAX_RATE_EXPONENT)
    rates = baseline * np.exp(exponents)
    return np.clip(rates, _PROBABILITY_EPSILON, 1.0 - _PROBABILITY_EPSILON)


def _fit_costs(k: np.ndarray, n: np.ndarray, rates: np.ndarray) -> np.ndarray:
    """Compute the binomial negative log-likelihood of each state per period.

    Args:
        k: Matched papers per period
        n: Reference-set papers per period
        rates: Expected share of each state

    Returns:
        Array of shape (n_states, n_periods) of fit costs
    """
    # gammaln instead of factorials: monthly arXiv volumes overflow float64 well
    # before this term matters.
    log_choose = gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)

    log_p = np.log(rates)[:, np.newaxis]
    log_q = np.log1p(-rates)[:, np.newaxis]

    return -(log_choose[np.newaxis, :] + k * log_p + (n - k) * log_q)


def _viterbi(fit_costs: np.ndarray, gamma: float, n_periods: int) -> np.ndarray:
    """Decode the minimum-total-cost state sequence.

    Args:
        fit_costs: Fit costs of shape (n_states, n_periods)
        gamma: Weight of the state-transition cost
        n_periods: Number of time bins

    Returns:
        Array of length n_periods holding the decoded state index per period
    """
    n_states = fit_costs.shape[0]

    # The transition cost scales with log(T), the number of TIME BINS, never
    # log(n_t). Kleinberg's batched model charges a burst for the number of
    # states it climbs against how many opportunities the stream had to climb
    # them. Substituting the per-period volume inflates the cost by roughly 1.9x
    # at arXiv monthly volumes and suppresses nearly every real burst.
    step_cost = gamma * np.log(n_periods)
    state_index = np.arange(n_states)
    transitions = np.maximum(state_index[np.newaxis, :] - state_index[:, np.newaxis], 0) * step_cost

    total_cost = fit_costs[:, 0] + transitions[0, :]
    backpointers = np.zeros((n_periods, n_states), dtype=np.int64)

    for period in range(1, n_periods):
        candidates = total_cost[:, np.newaxis] + transitions
        best_previous = np.argmin(candidates, axis=0)
        backpointers[period] = best_previous
        total_cost = candidates[best_previous, state_index] + fit_costs[:, period]

    states = np.zeros(n_periods, dtype=np.int64)
    states[-1] = int(np.argmin(total_cost))
    for period in range(n_periods - 1, 0, -1):
        states[period - 1] = backpointers[period, states[period]]

    return states


def _runs_to_bursts(
    states: np.ndarray,
    fit_costs: np.ndarray,
    periods: Sequence[str],
) -> list[Burst]:
    """Convert maximal runs of elevated states into bursts.

    Args:
        states: Decoded state index per period
        fit_costs: Fit costs of shape (n_states, n_periods)
        periods: Period labels aligned to the state sequence

    Returns:
        Bursts ordered by start index
    """
    bursts: list[Burst] = []
    start: int | None = None

    for index in range(len(states) + 1):
        elevated = index < len(states) and states[index] >= 1

        if elevated and start is None:
            start = index
        elif not elevated and start is not None:
            end = index - 1
            level = int(states[start : end + 1].max())
            weight = float(
                np.sum(fit_costs[0, start : end + 1] - fit_costs[level, start : end + 1])
            )
            bursts.append(
                Burst(
                    start=periods[start],
                    end=periods[end],
                    level=level,
                    weight=weight,
                    start_index=start,
                    end_index=end,
                )
            )
            start = None

    return bursts

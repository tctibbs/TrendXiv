"""Unit tests for Kleinberg burst detection."""

import math

import numpy as np
import pytest
from scipy.stats import binom

from src.analysis.burst import Burst, kleinberg_bursts

N_PERIODS = 60
REFERENCE_VOLUME = 10_000
BASELINE_SHARE = 0.01
STEP_START = 24
STEP_END = 36


def make_periods(count: int = N_PERIODS) -> list[str]:
    """Create monthly period labels starting at 2000-01."""
    return [f"{2000 + index // 12}-{index % 12 + 1:02d}" for index in range(count)]


def make_counts(share: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Create exact exposure counts for a per-period share."""
    n = np.full(len(share), REFERENCE_VOLUME, dtype=np.int64)
    return np.rint(share * n).astype(np.int64), n


@pytest.fixture
def periods() -> list[str]:
    """Monthly period labels for the synthetic series."""
    return make_periods()


@pytest.fixture
def flat_share() -> np.ndarray:
    """A trendless share series at the baseline rate."""
    return np.full(N_PERIODS, BASELINE_SHARE)


@pytest.fixture
def step_share(flat_share: np.ndarray) -> np.ndarray:
    """A flat series stepped up 5x for twelve months, then flat again."""
    share = flat_share.copy()
    share[STEP_START:STEP_END] = BASELINE_SHARE * 5
    return share


@pytest.mark.unit
class TestKleinbergBursts:
    """Tests for kleinberg_bursts."""

    def test_step_function_yields_one_burst_at_the_step(self, step_share, periods):
        """A 5x step must produce exactly one burst starting at the step."""
        k, n = make_counts(step_share)

        bursts = kleinberg_bursts(k, n, periods)

        assert len(bursts) == 1
        assert abs(bursts[0].start_index - STEP_START) <= 1
        assert abs(bursts[0].end_index - (STEP_END - 1)) <= 1

    def test_burst_reports_period_labels_of_its_endpoints(self, step_share, periods):
        """Burst dates must be the labels at the burst's own indices."""
        k, n = make_counts(step_share)

        burst = kleinberg_bursts(k, n, periods)[0]

        assert burst.start == periods[burst.start_index]
        assert burst.end == periods[burst.end_index]

    def test_burst_level_and_weight_are_positive(self, step_share, periods):
        """A detected burst sits above baseline, so both must be positive."""
        k, n = make_counts(step_share)

        burst = kleinberg_bursts(k, n, periods)[0]

        assert burst.level >= 1
        assert burst.weight > 0

    def test_flat_series_yields_no_bursts(self, flat_share, periods):
        """A series that never leaves its baseline must produce no bursts."""
        k, n = make_counts(flat_share)

        assert kleinberg_bursts(k, n, periods) == []

    def test_noisy_step_still_yields_one_burst(self, step_share, periods):
        """Binomial sampling noise must not fragment a real burst."""
        rng = np.random.default_rng(7)
        n = np.full(N_PERIODS, REFERENCE_VOLUME, dtype=np.int64)
        k = rng.binomial(n, step_share)

        bursts = kleinberg_bursts(k, n, periods)

        assert len(bursts) == 1
        assert abs(bursts[0].start_index - STEP_START) <= 1

    def test_growing_corpus_does_not_create_a_burst(self, flat_share, periods):
        """Raw counts rising with the corpus must not register as a burst."""
        n = np.rint(1000 * 1.02 ** np.arange(N_PERIODS)).astype(np.int64)
        k = np.rint(flat_share * n).astype(np.int64)

        assert kleinberg_bursts(k, n, periods) == []

    def test_larger_spike_reaches_a_higher_level(self, step_share, periods):
        """A 20x spike must be graded above a 5x step."""
        modest_k, n = make_counts(step_share)
        extreme_share = step_share.copy()
        extreme_share[STEP_START:STEP_END] = BASELINE_SHARE * 20
        extreme_k, _ = make_counts(extreme_share)

        modest = kleinberg_bursts(modest_k, n, periods)[0]
        extreme = kleinberg_bursts(extreme_k, n, periods)[0]

        assert extreme.level > modest.level

    def test_two_separated_spikes_yield_two_bursts(self, flat_share, periods):
        """Runs separated by baseline periods must not be merged."""
        share = flat_share.copy()
        share[10:16] = BASELINE_SHARE * 8
        share[40:46] = BASELINE_SHARE * 8
        k, n = make_counts(share)

        bursts = kleinberg_bursts(k, n, periods)

        assert len(bursts) == 2
        assert bursts[0].end_index < bursts[1].start_index

    def test_higher_gamma_suppresses_a_marginal_burst(self, flat_share, periods):
        """Raising the transition cost must make detection stricter."""
        share = flat_share.copy()
        share[30:34] = BASELINE_SHARE * 3
        n = np.full(N_PERIODS, 300, dtype=np.int64)
        k = np.rint(share * n).astype(np.int64)

        assert kleinberg_bursts(k, n, periods, gamma=1.0)
        assert kleinberg_bursts(k, n, periods, gamma=10.0) == []

    def test_transition_cost_scales_with_the_period_count(self, flat_share, periods):
        """The threshold is gamma*log(T) over TIME BINS, never log(n_t).

        A two-state automaton enters state 1 over exactly the elevated window
        when that window's binomial log-likelihood gain beats one upward
        transition, so the gamma at which detection flips pins log(T) itself.
        Substituting the per-period volume moves the flip by log(400)/log(60).
        """
        share = flat_share.copy()
        share[30:34] = BASELINE_SHARE * 2
        n = np.full(N_PERIODS, 400, dtype=np.int64)
        k = np.rint(share * n).astype(np.int64)

        baseline = k.sum() / n.sum()
        window = slice(30, 34)
        gain = float(
            np.sum(
                binom.logpmf(k[window], n[window], 2 * baseline)
                - binom.logpmf(k[window], n[window], baseline)
            )
        )
        critical_gamma = gain / math.log(N_PERIODS)

        detected = kleinberg_bursts(k, n, periods, gamma=critical_gamma * 0.95, n_states=2)
        assert [(burst.start_index, burst.end_index) for burst in detected] == [(30, 33)]
        assert kleinberg_bursts(k, n, periods, gamma=critical_gamma * 1.05, n_states=2) == []

    def test_baseline_is_the_pooled_corpus_rate(self, periods):
        """The baseline is total k over total n, not the mean of the shares.

        Thirty low-volume months at a 5% share sit 9x above the 0.54% pooled
        corpus rate and belong in the top state. Averaging the per-period
        shares would put the baseline at 2.75% and grade the same months as a
        marginal level-1 wobble.
        """
        n = np.concatenate([np.full(30, 100), np.full(30, 10_000)]).astype(np.int64)
        share = np.concatenate([np.full(30, 0.05), np.full(30, 0.005)])
        k = np.rint(share * n).astype(np.int64)

        bursts = kleinberg_bursts(k, n, periods)

        assert len(bursts) == 1
        assert (bursts[0].start_index, bursts[0].end_index) == (0, 29)
        assert bursts[0].level == 3

    def test_short_series_returns_no_bursts(self):
        """Fewer than four periods cannot support a burst model."""
        k = np.array([1, 50, 1])
        n = np.array([100, 100, 100])

        assert kleinberg_bursts(k, n, make_periods(3)) == []

    def test_all_zero_matches_yields_no_bursts(self, periods):
        """A term that never appears must not burst."""
        k = np.zeros(N_PERIODS, dtype=np.int64)
        n = np.full(N_PERIODS, REFERENCE_VOLUME, dtype=np.int64)

        assert kleinberg_bursts(k, n, periods) == []

    def test_saturated_baseline_yields_no_bursts(self, periods):
        """A term matching every paper leaves no room above baseline."""
        n = np.full(N_PERIODS, REFERENCE_VOLUME, dtype=np.int64)

        assert kleinberg_bursts(n, n, periods) == []

    def test_burst_is_hashable_dataclass(self):
        """Bursts are frozen so callers can key artifacts by them."""
        burst = Burst("2020-01", "2020-06", 2, 12.5, 0, 5)

        assert hash(burst) == hash(Burst("2020-01", "2020-06", 2, 12.5, 0, 5))


@pytest.mark.unit
class TestKleinbergBurstsValidation:
    """Tests for input validation in kleinberg_bursts."""

    def test_mismatched_lengths_raise(self, periods):
        """k and n must describe the same periods."""
        with pytest.raises(ValueError, match="same length"):
            kleinberg_bursts(np.ones(5), np.ones(6), periods)

    def test_misaligned_periods_raise(self):
        """Period labels must be aligned to the counts."""
        with pytest.raises(ValueError, match="aligned"):
            kleinberg_bursts(np.ones(5), np.ones(5), make_periods(4))

    def test_matches_exceeding_reference_raise(self):
        """k above n is not a valid exposure table."""
        with pytest.raises(ValueError, match="must not exceed"):
            kleinberg_bursts(np.array([1, 2, 3, 400]), np.full(4, 100), make_periods(4))

    def test_negative_counts_raise(self):
        """Counts are non-negative by construction."""
        with pytest.raises(ValueError, match="non-negative"):
            kleinberg_bursts(np.array([-1, 2, 3, 4]), np.full(4, 100), make_periods(4))

    def test_single_state_raises(self, periods):
        """An automaton needs a baseline state and at least one burst state."""
        with pytest.raises(ValueError, match="n_states"):
            kleinberg_bursts(np.ones(N_PERIODS), np.full(N_PERIODS, 100), periods, n_states=1)

    def test_non_separating_scaling_raises(self, periods):
        """s <= 1 collapses the states onto the baseline rate."""
        with pytest.raises(ValueError, match="greater than 1"):
            kleinberg_bursts(np.ones(N_PERIODS), np.full(N_PERIODS, 100), periods, s=1.0)

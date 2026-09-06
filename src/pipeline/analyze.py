"""Run the analysis package over the built cube and term index.

Everything here is offline. The browser receives finished numbers and draws
them; it computes no statistics of its own. That keeps a single implementation
of each method under test in Python rather than a second, untested one in
JavaScript.

Every function in this module excludes the provisional trailing window. A
partial month is not evidence of a decline, and letting one into a burst,
changepoint or rising computation manufactures exactly the false signal the
product exists to avoid.
"""

from __future__ import annotations

import logging

import numpy as np

from src.analysis.burst import kleinberg_bursts
from src.analysis.concentration import dominant_groups, time_adjusted_specificity
from src.analysis.lifecycle import fit_lifecycle, mann_kendall
from src.analysis.rising import deduplicate_ngrams, rising_scores
from src.analysis.seasonality import month_of_year_index, trading_day_adjust
from src.pipeline.terms import discovery_candidates

logger = logging.getLogger(__name__)

#: Months compared against the preceding baseline when scoring rising terms.
RECENT_WINDOW = 12
BASELINE_WINDOW = 36

#: A category needs this many months of history before a seasonal decomposition
#: is meaningful; STL itself requires two full periods.
MIN_SEASONAL_MONTHS = 60

#: A term needs this many papers over the whole corpus before a burst in it is
#: worth reporting.
MIN_BURST_PAPERS = 200

#: Minimum field-specificity for a term to enter burst detection or the term
#: discovery views. Academic prose ("enabling", "leveraging", "outperforms")
#: bursts exactly as hard as real topics did after 2023 and sits at a similar
#: corpus share, so neither frequency nor growth separates them -- but prose
#: tracks the corpus's own field distribution and a topic does not.
MIN_SPECIFICITY = 0.12

#: Window for the local level that overdispersion is measured against.
LOCAL_TREND_MONTHS = 13

#: Ceiling on the shipped overdispersion factor. Beyond this the band stops
#: informing and just swallows the chart.
MAX_OVERDISPERSION = 12.0


def group_month_matrix(cube: dict, taxonomy_group: dict, periods: list[str]) -> dict:
    """Aggregate the category cube into papers per archive group per month.

    Args:
        cube: Category cube.
        taxonomy_group: Category code to archive group.
        periods: Shared period axis.

    Returns:
        Archive group to a monthly paper-count array.
    """
    out: dict[str, np.ndarray] = {}
    for code, entry in cube.items():
        group = taxonomy_group.get(code)
        if group is None:
            continue
        arr = out.setdefault(group, np.zeros(len(periods), dtype=float))
        arr += np.asarray(entry["primary"], dtype=float)
    return out


def load_group_profiles(con) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
    """Read per-term archive-group usage and the corpus's own group profile.

    Args:
        con: Connection to the ingested working set.

    Returns:
        Tuple of (term -> {group: papers}, corpus {group: papers}).
    """
    profiles: dict[str, dict[str, float]] = {}
    for term, group, df in con.execute(
        "SELECT term, grp, sum(df) FROM term_group GROUP BY 1, 2"
    ).fetchall():
        profiles.setdefault(term, {})[group] = float(df)

    corpus = {
        group: float(df)
        for group, df in con.execute(
            "SELECT grp, sum(df) FROM term_group GROUP BY 1"
        ).fetchall()
    }
    logger.info("group profiles: %d terms over %d archive groups", len(profiles), len(corpus))
    return profiles, corpus


def specific_terms(
    candidates: dict[str, np.ndarray],
    profiles: dict[str, dict[str, float]],
    group_month: dict[str, np.ndarray],
    minimum: float = MIN_SPECIFICITY,
) -> dict[str, np.ndarray]:
    """Drop candidates whose field mix is explained by when they were used.

    Args:
        candidates: Term to monthly count vector.
        profiles: Term to per-group paper counts.
        group_month: Archive group to monthly paper counts.
        minimum: Specificity floor.

    Returns:
        The subset of candidates that are genuinely field-specific.
    """
    kept = {
        term: vector
        for term, vector in candidates.items()
        if time_adjusted_specificity(profiles.get(term, {}), vector, group_month) >= minimum
    }
    logger.info("specificity: %d of %d terms are field-specific (>= %.2f)",
                len(kept), len(candidates), minimum)
    return kept


def overdispersion_factor(cube: dict, totals: list[int], cutoff: int) -> float:
    """Estimate how much wider than binomial the real spread is.

    Papers are not independent draws: topics cluster and one group posts five
    papers at once, so a plain binomial band is too tight and everything looks
    significant. Measured across the largest categories and shipped for the
    browser to widen its intervals by sqrt(phi).

    Args:
        cube: Category cube.
        totals: Corpus papers per month.
        cutoff: Index of the first provisional bucket.

    Returns:
        Overdispersion factor, at least 1.0.
    """
    denom = np.asarray(totals[:cutoff], dtype=float)
    ranked = sorted(cube, key=lambda c: sum(cube[c]["any"]), reverse=True)[:40]
    factors = []
    for code in ranked:
        counts = np.asarray(cube[code]["any"][:cutoff], dtype=float)
        usable = (denom >= 400) & (counts > 0)
        if usable.sum() < 120:
            continue

        share = counts[usable] / denom[usable]
        # Compare each month against its LOCAL level, not against the pooled
        # rate. A category's share moves by orders of magnitude across 35 years,
        # and measuring against one global number scores that trend as
        # overdispersion: doing so returned phi=35, which would widen every band
        # sixfold. What we want is the excess wobble around the current level.
        expected = _centred_mean(share, LOCAL_TREND_MONTHS)
        n_used = denom[usable]
        variance = expected * (1.0 - expected) / n_used
        keep = variance > 0
        if keep.sum() < 60:
            continue
        residual = (share[keep] - expected[keep]) ** 2 / variance[keep]
        factors.append(float(residual.sum() / keep.sum()))

    phi = float(np.median(factors)) if factors else 1.0
    logger.info("overdispersion: phi=%.2f across %d categories", phi, len(factors))
    return float(np.clip(phi, 1.0, MAX_OVERDISPERSION))


def _centred_mean(values: np.ndarray, window: int) -> np.ndarray:
    """Centred moving average, used as the local expectation."""
    half = window // 2
    return np.array([
        values[max(0, i - half):min(len(values), i + half + 1)].mean()
        for i in range(len(values))
    ])


def seasonal_artifact(cube: dict, periods: list[str], cutoff: int) -> dict:
    """Compute month-of-year seasonal indices for every eligible category.

    Args:
        cube: Category cube.
        periods: Shared period axis.
        cutoff: Index of the first provisional bucket.

    Returns:
        Mapping of category code to ``{"month_index": {...}, "amplitude": float}``.
    """
    out: dict[str, dict] = {}
    # Eight years is enough to estimate twelve monthly factors without letting
    # a decade-old publishing culture dominate the current one.
    start = max(0, cutoff - 96)
    window = periods[start:cutoff]
    for code, entry in cube.items():
        values = np.asarray(entry["any"][start:cutoff], dtype=float)
        if len(values) < MIN_SEASONAL_MONTHS or (values > 0).sum() < MIN_SEASONAL_MONTHS:
            continue
        try:
            # Months hold between 20 and 23 weekdays, about a 5% swing that is
            # calendar arithmetic rather than behaviour. Left in, it lands
            # directly on the fingerprint this artifact exists to draw.
            index = month_of_year_index(trading_day_adjust(values, window), window)
        except (ValueError, np.linalg.LinAlgError) as error:
            logger.debug("seasonality skipped for %s: %s", code, error)
            continue
        factors = list(index.values())
        out[code] = {
            "month_index": {str(month): round(value, 4) for month, value in index.items()},
            "amplitude": round(max(factors) / min(factors), 3),
            "peak_month": int(max(index, key=index.get)),
        }
    logger.info("seasonality: %d categories", len(out))
    return out


def _window_counts(vector: np.ndarray, end: int, length: int) -> np.ndarray:
    """Slice a count vector to the ``length`` months ending at ``end``."""
    return vector[max(0, end - length):end]


def rising_artifact(
    term_vectors: dict[str, np.ndarray],
    totals: list[int],
    cutoff: int,
    q: float = 0.10,
    limit: int = 40,
) -> dict:
    """Score every discovery candidate and keep those surviving FDR control.

    Args:
        term_vectors: Term to monthly document-frequency vector.
        totals: Corpus papers per month.
        cutoff: Index of the first provisional bucket.
        q: Target false-discovery rate.
        limit: Maximum rows to ship.

    Returns:
        Artifact with the surviving rows and the number of terms screened.
    """
    denom = np.asarray(totals, dtype=float)
    n_recent = float(_window_counts(denom, cutoff, RECENT_WINDOW).sum())
    n_base = float(_window_counts(denom, cutoff - RECENT_WINDOW, BASELINE_WINDOW).sum())

    terms = sorted(term_vectors)
    k_recent = np.array(
        [_window_counts(term_vectors[t], cutoff, RECENT_WINDOW).sum() for t in terms], dtype=float
    )
    k_base = np.array(
        [_window_counts(term_vectors[t], cutoff - RECENT_WINDOW, BASELINE_WINDOW).sum()
         for t in terms], dtype=float
    )

    scored = rising_scores(terms, k_recent, n_recent, k_base, n_base, q=q)
    survivors = scored[scored["significant"]].copy()
    survivors = survivors.sort_values("delta_lo", ascending=False)

    keep = deduplicate_ngrams(
        survivors["term"].tolist(), survivors["delta_lo"].tolist()
    )
    survivors = survivors.iloc[keep].head(limit)

    # Each row carries its own sparkline. Without this the board asks the browser
    # for the vocabulary and one term shard per distinct first letter just to
    # draw twenty 90px sparklines, which turned a 240 KB first paint into 8 MB
    # over a dozen serial round trips.
    window = RECENT_WINDOW + BASELINE_WINDOW
    start = max(0, cutoff - window)
    spark_denom = denom[start:cutoff]
    rows = []
    for row in survivors.itertuples():
        counts = term_vectors[row.term][start:cutoff]
        with np.errstate(divide="ignore", invalid="ignore"):
            shares = np.divide(
                counts, spark_denom,
                out=np.zeros_like(counts, dtype=float), where=spark_denom > 0,
            )
        rows.append({
            "term": row.term,
            "share_recent": float(row.share_recent),
            "share_base": float(row.share_base),
            "delta_logodds": float(row.delta_logodds),
            "delta_lo": float(row.delta_lo),
            "q_value": float(row.q_value),
            "is_new": bool(row.is_new),
            "spark": [round(float(v), 7) for v in shares],
        })
    logger.info("rising: %d terms screened, %d survive at q=%.2f", len(terms), len(rows), q)
    return {"q": q, "n_screened": int(len(terms)), "rows": rows,
            "recent_months": RECENT_WINDOW, "baseline_months": BASELINE_WINDOW}


def burst_artifact(
    term_vectors: dict[str, np.ndarray],
    totals: list[int],
    periods: list[str],
    cutoff: int,
    limit: int = 60,
    profiles: dict[str, dict[str, float]] | None = None,
) -> dict:
    """Detect bursts for the most active terms.

    Args:
        term_vectors: Term to monthly document-frequency vector.
        totals: Corpus papers per month.
        periods: Shared period axis.
        cutoff: Index of the first provisional bucket.
        limit: Maximum terms to report.
        profiles: Per-term archive-group counts, used to label each burst with
            the field it belongs to.

    Returns:
        Artifact mapping each term to its detected burst intervals.
    """
    denom = np.asarray(totals[:cutoff], dtype=float)
    window = periods[:cutoff]

    # Score every eligible term, not a shortlist of the most voluminous ones.
    # Ranking candidates by raw volume before detection meant the pool was the
    # most common words in the corpus: "transformer" sat at rank 1157, "gan" at
    # 3902 and "bert" at 4444, so the single most famous burst in modern research
    # was excluded before the algorithm ever saw it, and the timeline filled up
    # with whichever generic words happened to be frequent.
    out = []
    for term in sorted(term_vectors):
        counts = np.asarray(term_vectors[term][:cutoff], dtype=float)
        if counts.sum() < MIN_BURST_PAPERS:
            continue
        try:
            bursts = kleinberg_bursts(counts, denom, window)
        except (ValueError, FloatingPointError) as error:
            logger.debug("burst skipped for %s: %s", term, error)
            continue
        if not bursts:
            continue
        strongest = max(bursts, key=lambda b: b.weight)
        row = {
            "term": term,
            "start": strongest.start,
            "end": strongest.end,
            "level": int(strongest.level),
            # Normalised by the term's own exposure: the raw Kleinberg weight
            # scales with volume, so ranking on it is a popularity ranking.
            "weight": round(float(strongest.weight) / max(1.0, counts.sum()) * 1000, 2),
        }
        if profiles:
            top = dominant_groups(profiles.get(term, {}), limit=2)
            if top:
                row["field"] = top[0][0]
                row["field_share"] = round(top[0][1], 3)
        out.append(row)
    out.sort(key=lambda row: row["weight"], reverse=True)
    selected = _balance_by_era(out, limit)
    logger.info("bursts: %d terms burst, %d shown across %d decades",
                len(out), len(selected), len({r["start"][:3] for r in selected}))
    return {"rows": selected}


def _balance_by_era(rows: list[dict], limit: int) -> list[dict]:
    """Take the strongest bursts from each decade rather than overall.

    Burst weight scales with the volume behind it, and arXiv published roughly
    a hundred times more papers in 2025 than in 1995, so a straight ranking
    hands 47 of 60 rows to the 2020s and the timeline stops being a timeline.
    Filling per decade first keeps the whole 35 years legible; any unused
    allowance falls back to the global ranking.

    Args:
        rows: Detected bursts, already sorted by descending weight.
        limit: Total rows to return.

    Returns:
        The selected rows, still ordered by weight.
    """
    decades = sorted({row["start"][:3] for row in rows})
    if not decades:
        return []

    per_decade = max(1, limit // len(decades))
    selected: list[dict] = []
    seen: set[str] = set()
    for decade in decades:
        for row in (r for r in rows if r["start"][:3] == decade):
            if len(selected) >= limit:
                break
            selected.append(row)
            seen.add(row["term"])
            if sum(1 for r in selected if r["start"][:3] == decade) >= per_decade:
                break

    for row in rows:
        if len(selected) >= limit:
            break
        if row["term"] not in seen:
            selected.append(row)
            seen.add(row["term"])

    selected.sort(key=lambda row: row["weight"], reverse=True)
    return selected


def lifecycle_artifact(
    term_vectors: dict[str, np.ndarray],
    totals: list[int],
    cutoff: int,
    limit: int = 60,
) -> dict:
    """Fit adoption curves for the most active terms.

    Saturation is reported only when the fitted inflection lies inside the
    observed window; a pre-inflection logistic fit is confidently wrong.

    Args:
        term_vectors: Term to monthly document-frequency vector.
        totals: Corpus papers per month.
        cutoff: Index of the first provisional bucket.
        limit: Maximum terms to report.

    Returns:
        Artifact with one row per term.
    """
    denom = np.asarray(totals[:cutoff], dtype=float)
    ranked = sorted(term_vectors, key=lambda t: term_vectors[t][:cutoff].sum(), reverse=True)

    rows = []
    for term in ranked[:limit]:
        counts = np.asarray(term_vectors[term][:cutoff], dtype=float)
        share = np.divide(counts, denom, out=np.zeros_like(counts), where=denom > 0)
        recent = share[-60:]
        if recent.sum() <= 0:
            continue
        try:
            fit = fit_lifecycle(np.cumsum(share))
            trend = mann_kendall(recent)
        except (ValueError, RuntimeError) as error:
            logger.debug("lifecycle skipped for %s: %s", term, error)
            continue
        rows.append({
            "term": term,
            "state": fit.state,
            "identifiable": bool(fit.identifiable),
            "saturation": round(float(fit.saturation), 6) if fit.identifiable else None,
            "model": fit.model,
            "trend": trend.trend,
            "share_now": round(float(share[-12:].mean()), 6),
            "growth": round(float(trend.slope * 12), 8),
        })
    logger.info("lifecycle: %d terms fitted", len(rows))
    return {"rows": rows}


def load_term_vectors(
    vocab: dict, shards_dir, length: int, min_df: int = 400, corpus_size: int | None = None
) -> dict:
    """Read the sharded term index back into memory for analysis.

    Args:
        vocab: Vocabulary artifact.
        shards_dir: Directory holding ``t-<shard>.json``.
        length: Period axis length.
        min_df: Lifetime document-frequency floor for analysis candidates.
        corpus_size: Total papers, used to exclude boilerplate by corpus share.

    Returns:
        Mapping of term to a dense monthly count vector.
    """
    import json

    candidates = set(discovery_candidates(vocab, min_df=min_df, corpus_size=corpus_size))
    vectors: dict[str, np.ndarray] = {}
    for path in sorted(shards_dir.glob("t-*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for term, entry in payload.items():
            if term not in candidates:
                continue
            vector = np.zeros(length, dtype=float)
            vector[entry["o"]:entry["o"] + len(entry["v"])] = entry["v"]
            vectors[term] = vector
    logger.info("loaded %d term vectors for analysis", len(vectors))
    return vectors

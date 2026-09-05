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
from src.analysis.seasonality import month_of_year_index
from src.pipeline.terms import discovery_candidates

logger = logging.getLogger(__name__)

#: Months compared against the preceding baseline when scoring rising terms.
RECENT_WINDOW = 12
BASELINE_WINDOW = 36

#: A category needs this many months of history before a seasonal decomposition
#: is meaningful; STL itself requires two full periods.
MIN_SEASONAL_MONTHS = 60

#: Minimum field-specificity for a term to enter burst detection or the term
#: discovery views. Academic prose ("enabling", "leveraging", "outperforms")
#: bursts exactly as hard as real topics did after 2023 and sits at a similar
#: corpus share, so neither frequency nor growth separates them -- but prose
#: tracks the corpus's own field distribution and a topic does not.
MIN_SPECIFICITY = 0.12


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
            index = month_of_year_index(values, window)
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

    rows = [
        {
            "term": row.term,
            "share_recent": float(row.share_recent),
            "share_base": float(row.share_base),
            "delta_logodds": float(row.delta_logodds),
            "delta_lo": float(row.delta_lo),
            "q_value": float(row.q_value),
            "is_new": bool(row.is_new),
        }
        for row in survivors.itertuples()
    ]
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
    ranked = sorted(term_vectors, key=lambda t: term_vectors[t][:cutoff].sum(), reverse=True)

    out = []
    for term in ranked[:limit * 4]:
        counts = np.asarray(term_vectors[term][:cutoff], dtype=float)
        if counts.sum() < 200:
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
    logger.info("bursts: %d terms with a detected burst", len(out))
    return {"rows": out[:limit]}


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

"""How narrowly a term belongs to a field.

Frequency cannot separate a research topic from academic prose. Both
"transformer" and "enabling" surged after 2023, both sit at a similar share of
the corpus, and both produce a textbook burst. What separates them is *where*
they are used: "graphene" is almost entirely condensed matter, while "enabling"
is spread across every archive in proportion to that archive's size.

The measure is the Jensen-Shannon divergence between a term's distribution over
top-level archive groups and the corpus's own distribution over those groups.
Comparing against the corpus rather than against uniform matters: arXiv is not
evenly divided, so a term used indiscriminately still looks "concentrated in
cs" simply because cs is large.
"""

from __future__ import annotations

import numpy as np


def _kl(p: np.ndarray, q: np.ndarray) -> float:
    """Kullback-Leibler divergence, skipping zero-probability terms."""
    mask = p > 0
    return float(np.sum(p[mask] * np.log2(p[mask] / q[mask])))


def jensen_shannon(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence in bits, bounded to [0, 1].

    Args:
        p: A distribution; need not be normalised.
        q: The reference distribution; need not be normalised.

    Returns:
        Divergence in [0, 1], where 0 means identical to the reference.
    """
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    if p.sum() <= 0 or q.sum() <= 0:
        return 0.0
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    return float(np.clip(0.5 * _kl(p, m) + 0.5 * _kl(q, m), 0.0, 1.0))


def term_specificity(term_counts: dict[str, float], corpus_counts: dict[str, float]) -> float:
    """Score how field-specific a term is against a fixed reference.

    Prefer :func:`time_adjusted_specificity` for real corpora. This variant
    compares against one static reference and therefore confounds field with
    time: on arXiv, any term that became common recently looks "concentrated in
    computer science" purely because that is where recent growth happened.

    Args:
        term_counts: Papers using the term, by archive group.
        corpus_counts: Papers in the corpus, by archive group.

    Returns:
        Specificity in [0, 1]. A term tracking the reference scores near 0.
    """
    groups = sorted(corpus_counts)
    p = np.array([term_counts.get(g, 0.0) for g in groups], dtype=float)
    q = np.array([corpus_counts.get(g, 0.0) for g in groups], dtype=float)
    return jensen_shannon(p, q)


def expected_group_mix(
    monthly_counts: np.ndarray,
    group_month: dict[str, np.ndarray],
) -> dict[str, float]:
    """Field mix a term would show if it were used in proportion to field activity.

    This is the null hypothesis a specific term has to beat: given *when* the
    term was used, how would its field mix look if it carried no field
    preference at all? Weighting the corpus's own monthly field mix by the
    term's monthly usage is what removes the time confound.

    Args:
        monthly_counts: The term's papers per month.
        group_month: Archive group to that group's papers per month.

    Returns:
        Expected papers per archive group.
    """
    weights = np.asarray(monthly_counts, dtype=float)
    total = weights.sum()
    if total <= 0:
        return {group: 0.0 for group in group_month}
    weights = weights / total

    per_month_total = np.sum([counts for counts in group_month.values()], axis=0)
    expected: dict[str, float] = {}
    for group, counts in group_month.items():
        with np.errstate(divide="ignore", invalid="ignore"):
            share = np.divide(
                counts, per_month_total,
                out=np.zeros_like(counts, dtype=float),
                where=per_month_total > 0,
            )
        expected[group] = float(np.sum(weights * share))
    return expected


def time_adjusted_specificity(
    term_counts: dict[str, float],
    monthly_counts: np.ndarray,
    group_month: dict[str, np.ndarray],
) -> float:
    """Score field-specificity after removing the time confound.

    Args:
        term_counts: Papers using the term, by archive group.
        monthly_counts: The term's papers per month.
        group_month: Archive group to that group's papers per month.

    Returns:
        Specificity in [0, 1]. Academic prose scores near 0 even when it surged
        recently in one field; a term genuinely tied to a field scores high.
    """
    return term_specificity(term_counts, expected_group_mix(monthly_counts, group_month))


def dominant_groups(
    term_counts: dict[str, float], limit: int = 3
) -> list[tuple[str, float]]:
    """Return the archive groups a term is most used in, with their shares.

    Args:
        term_counts: Papers using the term, by archive group.
        limit: Maximum groups to return.

    Returns:
        ``(group, share)`` pairs, largest first.
    """
    total = sum(term_counts.values())
    if total <= 0:
        return []
    ranked = sorted(term_counts.items(), key=lambda item: item[1], reverse=True)
    return [(group, count / total) for group, count in ranked[:limit]]

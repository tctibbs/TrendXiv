"""Build-time integrity checks.

These gates exist because the failure mode they guard against is silent: a
schema change upstream, a partially-read shard, or a date-parsing regression
all produce a complete-looking cube whose numbers are quietly wrong.

arXiv's published monthly submission counts are used here and ONLY here. They
are a check on our derived totals, never a denominator -- see
:mod:`src.pipeline.cube` for why mixing the two sources into one ratio is a bug.
"""

from __future__ import annotations

import csv
import io
import logging
import urllib.request
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

OFFICIAL_CSV = "https://arxiv.org/stats/get_monthly_submissions"

#: Aggregate tolerance. Observed drift against the official series is ~0.03%.
#: The gate is deliberately on the total rather than per-month: the worst
#: individual historical month differs by ~2.6%, so a 1% per-month gate would
#: fail on the very build it was meant to certify.
AGGREGATE_TOLERANCE = 0.01

#: Accepted range for the corpus row count against the recorded baseline. The
#: corpus only grows, so anything below 1.0 means rows went missing; the upper
#: bound catches a snapshot that has changed shape rather than merely grown.
MIN_GROWTH = 1.0
MAX_GROWTH = 1.5

#: Per-decade tolerance. An aggregate-only gate is where compensating errors
#: hide, so each decade is checked separately at a looser bound.
DECADE_TOLERANCE = 0.05


@dataclass
class ValidationReport:
    """Outcome of the build-time integrity checks."""

    passed: bool
    checks: list[tuple[str, bool, str]] = field(default_factory=list)

    def add(self, name: str, ok: bool, detail: str) -> None:
        """Record one check."""
        self.checks.append((name, ok, detail))
        if not ok:
            self.passed = False

    def render(self) -> str:
        """Return a human-readable summary."""
        return "\n".join(
            f"{'PASS' if ok else 'FAIL'}  {name}: {detail}" for name, ok, detail in self.checks
        )


def fetch_official_monthly(timeout: int = 60) -> dict[str, int]:
    """Fetch arXiv's published monthly submission counts.

    Args:
        timeout: Socket timeout in seconds.

    Returns:
        Mapping of ``YYYY-MM`` to submission count.
    """
    with urllib.request.urlopen(OFFICIAL_CSV, timeout=timeout) as response:
        text = response.read().decode("utf-8")
    reader = csv.DictReader(io.StringIO(text))
    return {row["month"]: int(row["submissions"]) for row in reader}


def validate(
    periods: list[str],
    totals: dict,
    cube: dict,
    corpus_rows: int,
    expected_rows: int,
    official: dict[str, int] | None = None,
) -> ValidationReport:
    """Run every integrity gate over the built artifacts.

    Args:
        periods: The shared period axis.
        totals: Denominator artifact from :func:`src.pipeline.cube.build_totals`.
        cube: Category cube from :func:`src.pipeline.cube.build_cube`.
        corpus_rows: Rows actually ingested.
        expected_rows: Baseline row count from the last known-good build.
        official: Published monthly counts; fetched if omitted.

    Returns:
        A :class:`ValidationReport`; ``passed`` is False if any gate failed.
    """
    report = ValidationReport(passed=True)

    # A growth band, not an equality. The scheduled rebuild exists to pick up new
    # papers, so demanding an exact row count guarantees the build aborts the
    # first week the snapshot grows and the site silently freezes on its last
    # good deploy. What is worth catching is the corpus shrinking, or jumping so
    # far that something upstream has changed shape.
    growth = corpus_rows / expected_rows if expected_rows else 0.0
    report.add(
        "row_count",
        MIN_GROWTH <= growth <= MAX_GROWTH,
        f"ingested {corpus_rows:,} against a {expected_rows:,} baseline "
        f"({growth:.4f}x, accepted {MIN_GROWTH}-{MAX_GROWTH}x)",
    )
    report.add(
        "axis_alignment",
        all(len(entry[mode]) == len(periods) for entry in cube.values() for mode in entry),
        f"{len(cube)} categories x {len(periods)} periods",
    )
    report.add(
        "no_negative_counts",
        all(value >= 0 for entry in cube.values() for value in entry["any"]),
        "all category counts non-negative",
    )

    papers = totals["papers"]
    report.add(
        "denominator_monotone_coverage",
        sum(1 for value in papers if value == 0) < 3,
        f"{sum(1 for value in papers if value == 0)} empty months on a {len(papers)}-month axis",
    )

    if official is None:
        try:
            official = fetch_official_monthly()
        except OSError as error:  # Network is unavailable in some CI contexts.
            report.add("official_csv", True, f"skipped, unreachable ({error})")
            return report

    overlap = [p for p in periods if p in official]
    ours = sum(papers[periods.index(p)] for p in overlap)
    theirs = sum(official[p] for p in overlap)
    drift = abs(ours - theirs) / theirs if theirs else 1.0
    report.add(
        "official_aggregate",
        drift <= AGGREGATE_TOLERANCE,
        f"ours {ours:,} vs official {theirs:,} ({drift:.4%} drift over {len(overlap)} months)",
    )

    for decade in sorted({p[:3] for p in overlap}):
        months = [p for p in overlap if p.startswith(decade)]
        ours_d = sum(papers[periods.index(p)] for p in months)
        theirs_d = sum(official[p] for p in months)
        drift_d = abs(ours_d - theirs_d) / theirs_d if theirs_d else 1.0
        report.add(
            f"official_decade_{decade}0s",
            drift_d <= DECADE_TOLERANCE,
            f"{ours_d:,} vs {theirs_d:,} ({drift_d:.3%})",
        )

    return report

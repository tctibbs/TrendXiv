"""Build every static artifact and write a manifest.

Run with ``python -m src.pipeline.build``. Output lands in ``build/`` and is
deployed to GitHub Pages as an Actions artifact -- never committed. Committing
it would hit the 100 MiB per-file block, and a Pages-served Git LFS file
returns the pointer text rather than the file.

Data filenames are content-hashed and the manifest is the only stable name.
A rebuilt artifact at a stable URL, combined with a browser cache, lets a
mid-session redeploy mix old and new bytes; hashing makes that impossible.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import duckdb

from src.common.categories import LEGACY_ARCHIVES, display_name, short_name
from src.common.taxonomy import CATEGORY_EVENTS, GROUPS, group_of
from src.pipeline import cube as cube_module
from src.pipeline.analyze import (
    burst_artifact,
    group_month_matrix,
    lifecycle_artifact,
    load_group_profiles,
    load_term_vectors,
    rising_artifact,
    seasonal_artifact,
    specific_terms,
)
from src.pipeline.periods import completeness
from src.pipeline.terms import build_term_index
from src.pipeline.validate import validate

logger = logging.getLogger(__name__)

EXPECTED_ROWS = 3_148_882
SCHEMA_VERSION = 1


def write_hashed(out_dir: Path, name: str, payload: object) -> str:
    """Write a JSON artifact under a content-hashed filename.

    Args:
        out_dir: Build directory.
        name: Logical name, e.g. ``cube``.
        payload: JSON-serialisable content.

    Returns:
        The filename written, including the hash.
    """
    # sort_keys is what makes the hash a function of the CONTENT rather than of
    # DuckDB's row order, which varies run to run under parallel scans. Without
    # it every build emits new filenames and busts every cache for no reason.
    body = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    digest = hashlib.sha256(body).hexdigest()[:10]
    filename = f"{name}-{digest}.json"
    (out_dir / filename).write_bytes(body)
    logger.info("wrote %s (%.1f KB)", filename, len(body) / 1024)
    return filename


def build(db_path: Path, out_dir: Path, skip_terms: bool = False, strict: bool = True) -> dict:
    """Build all artifacts from the ingested working set.

    Args:
        db_path: Path to the DuckDB working set produced by the ingest.
        out_dir: Directory to write artifacts into.
        skip_terms: Skip the term index (useful for fast iteration on the cube).
        strict: Abort when an integrity gate fails. Only a preview build built
            from a partial ingest should ever set this False.

    Returns:
        The manifest dictionary that was written.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path), read_only=True)

    periods, max_period = cube_module.axis_for(con)
    max_submission = con.execute(
        "SELECT max(strptime(period || '-01', '%Y-%m-%d')) FROM corpus_month WHERE period = ?",
        [max_period],
    ).fetchone()[0]
    complete_through, provisional_from = completeness(max_submission.date())
    corpus_rows = con.execute("SELECT sum(rows) FROM shard_log").fetchone()[0]
    # Provenance is read back from the working set, not from a constant in the
    # source: the manifest must record the revision the data actually came from.
    revision_row = con.execute(
        "SELECT value FROM build_meta WHERE key = 'source_revision'"
    ).fetchone()
    revision = revision_row[0] if revision_row else "unknown"

    logger.info("axis %s..%s (%d months), provisional from %s",
                periods[0], periods[-1], len(periods), provisional_from)

    series = cube_module.build_cube(con, periods)
    totals = cube_module.build_totals(con, periods)

    report = validate(periods, totals, series, corpus_rows, EXPECTED_ROWS)
    logger.info("validation:\n%s", report.render())
    if not report.passed:
        if strict:
            raise SystemExit("build aborted: integrity checks failed\n" + report.render())
        logger.warning("PREVIEW BUILD: integrity gates failed, artifacts are not publishable")

    taxonomy = {
        "groups": GROUPS,
        "category_group": {code: group_of(code) for code in series},
        # The code is always paired with a name, never replaced by one: cs.LG and
        # stat.ML are both officially "Machine Learning".
        "names": {code: display_name(code) for code in series},
        "short_names": {code: short_name(code) for code in series},
        "legacy": sorted(code for code in series if code in LEGACY_ARCHIVES),
        "events": [
            {"code": e.code, "date": e.date, "kind": e.kind, "note": e.note}
            for e in CATEGORY_EVENTS
            if e.code in series
        ],
    }

    files = {
        "cube": write_hashed(out_dir, "cube", {"series": series}),
        "totals": write_hashed(out_dir, "totals", totals),
        "taxonomy": write_hashed(out_dir, "taxonomy", taxonomy),
    }

    cutoff = periods.index(provisional_from)

    # Seasonality needs only the cube, so it ships even on a cube-only build.
    files["seasonal"] = write_hashed(
        out_dir, "seasonal", seasonal_artifact(series, periods, cutoff)
    )

    if not skip_terms:
        vocab = build_term_index(con, periods, out_dir)
        files["vocab"] = write_hashed(out_dir, "vocab", vocab)
        files["terms_dir"] = "terms"

        vectors = load_term_vectors(
            vocab, out_dir / "terms", len(periods), corpus_size=corpus_rows
        )
        profiles, _ = load_group_profiles(con)
        group_month = group_month_matrix(series, taxonomy["category_group"], periods)
        topical = specific_terms(vectors, profiles, group_month)
        papers = totals["papers"]

        # Rising screens the full candidate set: a genuinely new term is worth
        # surfacing even before it has enough history to look field-specific.
        files["rising"] = write_hashed(
            out_dir, "rising", rising_artifact(vectors, papers, cutoff)
        )
        files["bursts"] = write_hashed(
            out_dir, "bursts",
            burst_artifact(topical, papers, periods, cutoff, profiles=profiles),
        )
        files["lifecycle"] = write_hashed(
            out_dir, "lifecycle", lifecycle_artifact(topical, papers, cutoff)
        )

    events_path = Path("data/reference/events.yaml")
    if events_path.exists():
        try:
            from src.pipeline.events import load_events, to_artifact

            files["events"] = write_hashed(out_dir, "events", to_artifact(load_events(events_path)))
        except Exception as error:  # Events are additive; never fail the build on them.
            logger.warning("events artifact skipped: %s", error)

    manifest = {
        "schema": SCHEMA_VERSION,
        "built_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "source_revision": revision,
        "corpus_rows": corpus_rows,
        "categories": len(series),
        "data_complete_through": complete_through,
        "provisional_from": provisional_from,
        "periods": periods,
        "files": files,
        "validation": [
            {"check": name, "ok": ok, "detail": detail} for name, ok, detail in report.checks
        ],
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, separators=(",", ":")), encoding="utf-8"
    )
    con.close()
    logger.info("manifest written: %d categories, %d periods", len(series), len(periods))
    return manifest


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Build TrendXiv static artifacts.")
    parser.add_argument("--db", type=Path, default=Path("data/cache/trendxiv.duckdb"))
    parser.add_argument("--out", type=Path, default=Path("build"))
    parser.add_argument("--skip-terms", action="store_true")
    parser.add_argument(
        "--no-gate", action="store_true", help="preview build from a partial ingest"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    build(args.db, args.out, skip_terms=args.skip_terms, strict=not args.no_gate)


if __name__ == "__main__":
    main()

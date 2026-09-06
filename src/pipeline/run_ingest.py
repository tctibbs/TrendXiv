"""CLI wrapper around the snapshot ingest: ``python -m src.pipeline.run_ingest``."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.pipeline.ingest import ingest_all


def main() -> None:
    """Run the ingest, resuming from any shards already recorded."""
    parser = argparse.ArgumentParser(description="Ingest the arXiv metadata snapshot.")
    parser.add_argument("--db", type=Path, default=Path("data/cache/trendxiv.duckdb"))
    parser.add_argument("--cache", type=Path, default=Path("data/cache"))
    parser.add_argument("--keep-shards", action="store_true")
    parser.add_argument(
        "--revision", help="ingest a specific upstream commit instead of the latest"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    total = ingest_all(
        args.db, args.cache, keep_shards=args.keep_shards, revision=args.revision
    )
    logging.info("ingest complete: %d new rows", total)


if __name__ == "__main__":
    main()

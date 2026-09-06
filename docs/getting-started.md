# Getting started

[Back to the README](../README.md) &middot; [Development](development.md) &middot; [Data contract](data-contract.md)

You need Python 3.12 and about 3 GB of bandwidth for the first run.

```bash
uv venv --python 3.12 .venv && source .venv/bin/activate
pip install -e ".[dev]"

python -m src.pipeline.run_ingest          # about 15 minutes, once
python -m src.pipeline.build --out web/data
python -m http.server 8777 -d web          # http://127.0.0.1:8777
```

The ingest downloads the arXiv snapshot one shard at a time and throws each away
after reading it, so peak disk stays around 350 MB rather than 3 GB. It writes a
DuckDB working set to `data/cache/`, and every later build reads that instead of
the network.

## Rebuilding

Once the working set exists, a rebuild takes seconds:

```bash
python -m src.pipeline.build --out web/data
```

Add `--skip-terms` if you are only touching the frontend. It builds the category
cube and the seasonal artifact and skips the keyword index, which is most of the
time.

Re-running the ingest is cheap when nothing upstream has moved. It compares the
snapshot revision against the one already ingested and does nothing if they
match. When the revision has moved it clears the working set and starts over,
which is what makes the weekly rebuild pick up new papers instead of quietly
republishing the same numbers.

## Where things end up

`web/data/` holds the built artifacts. They are gitignored on purpose: they are
deployed as a Pages artifact, and a Pages-served Git LFS file hands back the
pointer text rather than the file. [The data contract](data-contract.md) describes their shape.

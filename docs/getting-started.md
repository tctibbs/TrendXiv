# Getting started

[Back to the README](../README.md) &middot; [Development](development.md) &middot; [Data contract](data-contract.md)

You need Python 3.12. The first run pulls about 3 GB.

```bash
uv venv --python 3.12 .venv && source .venv/bin/activate
pip install -e ".[dev]"

python -m src.pipeline.run_ingest          # about 15 minutes, once
python -m src.pipeline.build --out web/data
python -m http.server 8777 -d web          # http://127.0.0.1:8777
```

The ingest reads the arXiv snapshot one shard at a time and deletes each one
after it is done, so you never need 3 GB free, only about 350 MB. What it keeps
is a DuckDB working set in `data/cache/`, and every later build reads that
instead of the network.

## Rebuilding

Once the working set exists, rebuilds take seconds:

```bash
python -m src.pipeline.build --out web/data
```

Add `--skip-terms` if you are only changing the frontend. It builds the category
cube and skips the keyword index, which is most of the work.

Running the ingest again is cheap. It checks the snapshot revision first and
stops if nothing has moved. If the revision has changed it wipes the working set
and starts over, which is how the weekly rebuild picks up new papers rather than
republishing the same numbers forever.

## Where things end up

Built artifacts land in `web/data/`, which is gitignored. They ship as a Pages
artifact rather than in the repo, partly because a Pages-served Git LFS file
hands back the pointer text instead of the file. [The data
contract](data-contract.md) describes their shape.

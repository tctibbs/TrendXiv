# TrendXiv

**[See it live](https://tctibbs.github.io/TrendXiv/)**

Which research topics are actually growing on arXiv, and which ones only look like
they are. Built from all 3,148,882 papers going back to 1991.

There is no server, no database and no API key. A weekly GitHub Actions job turns
the whole arXiv corpus into a handful of static JSON files, GitHub Pages serves
them, and the browser downloads about 240 KB and draws the charts itself. The
whole thing runs on free tiers.

## Why it isn't just a line chart

arXiv grew about 102 times over between 1992 and today and is still adding roughly
27% more papers each year. Plot raw keyword counts and you are mostly plotting that
growth, so almost everything slopes up, including topics that are quietly shrinking.
TrendXiv shows **share of the corpus** by default, names the denominator on the axis,
and puts the raw numerator in every tooltip.

Beyond that:

| | What it does | Why it beats the naive version |
|---|---|---|
| **Wilson intervals** | Every series carries a band, with an overdispersion correction | Wald intervals degenerate at k=0, exactly the regime a term tool lives in |
| **Kleinberg bursts** | Dated intervals, not a wiggly line | "diffusion entered a level-3 burst 2022-04 → 2023-11" is a claim you can check |
| **Rising board** | Empirical-Bayes shrinkage + Benjamini-Hochberg FDR | 1 paper → 5 papers is not a "+400% breakout" |
| **Deadline fingerprint** | Month-of-year seasonal index per field via STL | cs.CV peaks in March (CVPR); math.AP is nearly flat |
| **Lifecycle fits** | Logistic/Gompertz with an identifiability gate | Saturation is never reported from a pre-inflection fit |
| **Provisional masking** | Trailing incomplete months hatched and excluded from inference | An 87% "collapse" in the current month is a calendar artifact |

## Architecture

```
Weekly GitHub Actions job (unmetered on public repos)
  │
  ├─ ingest    DuckDB reads the pinned arXiv snapshot shard by shard.
  │            Peak disk stays ~350 MB; the network is touched once.
  │            Categories are canonicalised AND deduplicated per paper.
  │
  ├─ cube      category × month counts in three attribution modes
  ├─ terms     term × month document frequency, sharded by first letter
  ├─ analyze   seasonality, bursts, rising terms, lifecycle fits
  └─ validate  gates against arXiv's published monthly totals
  │
  ▼
build/  ──►  actions/upload-pages-artifact  ──►  GitHub Pages
```

Artifacts are **never committed**. A Pages-served Git LFS file returns the pointer
text, not the file, and a 100 MiB per-file block ends the discussion anyway.

Everything statistical happens offline in Python and is unit-tested. The browser
receives finished numbers and draws them; it computes no statistics of its own.

## Quick start

```bash
uv venv --python 3.12 .venv && source .venv/bin/activate
pip install -e ".[dev]"

python -m src.pipeline.run_ingest      # ~15 min, ~3 GB transferred, once
python -m src.pipeline.build --out web/data
python -m http.server 8777 -d web      # open http://127.0.0.1:8777
```

Iterating on the frontend only? `--skip-terms` builds the category cube and the
seasonal artifact in about a second.

## Data notes, stated plainly

Papers are bucketed by their **v1 submission date**, never `update_date`: the latter
piles revisions of old papers into recent months and rewrites history on every build.

Category counts default to **any-listing** attribution, matching arXiv's own `cat:`
queries, so a number here can be checked against arxiv.org. Primary-only and
fractional modes ship alongside.

Three limits no free data source can fix, surfaced in the UI rather than buried:

1. **Category membership is current, not historical.** A 2010 paper cross-listed into
   cs.LG in 2024 counts in cs.LG's 2010 bucket, so today's taxonomy is back-projected
   onto history.
2. **Text is the latest revision's.** A 2016 paper revised in 2025 can register a 2025
   term in 2016.
3. **arXiv is not science.** Preprinting is near-universal in high-energy physics and
   weak in chemistry and clinical medicine, so cross-field comparisons partly measure
   preprinting culture.

## Development

```bash
ruff check src tests
pytest -q
```

## Licence

MIT. arXiv metadata is CC0. TrendXiv is not affiliated with or endorsed by arXiv or
Cornell University.

# Development

[Back to the README](../README.md) &middot; [Getting started](getting-started.md) &middot; [Data contract](data-contract.md)

```bash
ruff check src tests
pytest -q
```

Coverage has to clear 80%. It currently sits near 88%.

## How the code is laid out

`src/pipeline/` is the ETL. It reads the snapshot, builds the cube and the term
index, runs the analysis and writes the artifacts.

`src/analysis/` is pure functions over numpy arrays: Wilson intervals, STL
seasonality, Kleinberg bursts, rising-term scoring, lifecycle fits, field
specificity. No I/O, no network, no globals. That is what makes it testable
against synthetic series with known answers.

`web/` is the site. Vanilla ES modules and hand-rolled SVG, no framework and no
chart library. Every mark the design needs is a custom mark in any library
anyway, so a library would have cost around 480 KB and still left the hard parts
to write.

## Two rules worth keeping

**Statistics stay in Python.** The browser receives finished numbers and draws
them. One implementation under test beats two that drift.

**Analysis never sees smoothed data.** A moving average injects autocorrelation,
so a changepoint finder run on it finds changepoints in the smoother. Smoothing
is a presentation step and happens last.

## Testing

Golden tests over synthetic series, not smoke tests. A step function has to
produce a burst at the step. A pure sinusoid has to deseasonalise flat. A
simulated logistic has to recover its saturation. A trendless series has to stay
unflagged.

`tests/integration/` builds real artifacts from a small fixture and checks the
things that only break end to end: attribution modes summing correctly, alias
pairs collapsing, content hashes staying stable across identical builds.

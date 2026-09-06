# Development

[Back to the README](../README.md) &middot; [Getting started](getting-started.md) &middot; [Data contract](data-contract.md)

```bash
ruff check src tests
pytest -q
```

Coverage has to clear 80%. It sits near 88%.

## The layout

`src/pipeline/` reads the snapshot and writes the artifacts.

`src/analysis/` is pure functions over numpy arrays: intervals, seasonality,
bursts, rising terms, lifecycle fits, field specificity. No I/O and no globals,
which is what makes it testable against synthetic series with known answers.

`web/` is the site. Vanilla ES modules and hand-rolled SVG. Direct labels, event
flags and hatched regions are custom marks in any charting library anyway, so
one would have cost around 480 KB and still left the hard parts to write.

## Two rules

**Statistics stay in Python.** The browser gets finished numbers. One
implementation beats two that drift apart.

**Analysis never sees smoothed data.** A moving average adds autocorrelation, so
a changepoint finder run on it finds changepoints in the smoother. Smoothing is
a presentation step and happens last.

## Tests

Golden tests over synthetic series, not smoke tests. A step function has to
produce a burst at the step. A sinusoid has to deseasonalise flat. A simulated
logistic has to recover its saturation. A trendless series has to stay unflagged.

`tests/integration/` builds real artifacts from a small fixture and checks what
only breaks end to end: attribution modes summing correctly, alias pairs
collapsing, hashes staying stable across identical builds.

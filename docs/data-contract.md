# Data contract

[Back to the README](../README.md) &middot; [Getting started](getting-started.md) &middot; [Development](development.md)

`python -m src.pipeline.build` writes everything the site needs into one folder,
and that folder is deployed as a Pages artifact. None of it is committed.

Filenames carry a content hash, so a file that changes gets a new name. That is
what stops a reader holding a fresh manifest alongside a cached copy of last
week's data. `manifest.json` is the only stable name, and the site reads it
first to find everything else.

## manifest.json

```json
{
  "schema": 1,
  "built_at": "2026-09-06T13:24:00+00:00",
  "source_revision": "80dbeac57a...",
  "corpus_rows": 3148882,
  "categories": 152,
  "data_complete_through": "2026-07",
  "provisional_from": "2026-08",
  "overdispersion": 2.275,
  "periods": ["1991-07", "...", "2026-08"],
  "files": { "cube": "cube-<hash>.json", "terms_shards": { "a": "t-a-<hash>.json" } },
  "validation": [{ "check": "row_count", "ok": true, "detail": "..." }]
}
```

`periods` is the shared month axis. Every series elsewhere is an array lined up
against it, so nothing else has to carry dates.

Months from `provisional_from` onward are still filling up. They are drawn
hatched and left out of every calculation.

`overdispersion` widens the uncertainty bands. Papers are not independent draws,
so a plain binomial interval claims more precision than the data supports.

## cube

```json
{ "series": { "cs.LG": { "any": [...], "primary": [...], "frac": [...] } } }
```

Three ways of counting the same paper:

- `any` counts it in every category it lists. This matches arXiv's own `cat:`
  search, so the number is checkable. It is the default.
- `primary` counts only the first category listed.
- `frac` splits one paper across its categories, so the shares sum to one. Used
  for the stacked view.

## totals

```json
{ "papers": [...], "listings": [...] }
```

`papers` is the denominator for every share. It comes from the same snapshot as
the numerator, never from arXiv's published CSV. Mixing the two puts sources
that disagree by up to 2.6% in a month into one ratio, and that disagreement
drifts, so it reads as signal.

## terms

Sharded by first character, listed in `manifest.files.terms_shards`.

```json
{ "transformer": { "o": 180, "v": [1, 0, 3] } }
```

`o` is where the vector starts on the period axis, since leading and trailing
zeros are dropped. `v` counts papers that mention the term at least once, not
mentions. Abstracts have grown longer over 35 years, so counting mentions would
bake that drift into every series.

`vocab` holds `terms` as a plain term to count map, plus `min_df` and
`below_threshold_count`, which let the site say a word is below the indexing
threshold rather than silently drawing zeros.

## Everything else

`taxonomy` carries category names, groups and dated events like the astro-ph
split. `seasonal` holds a month-of-year index per category. `bursts` and
`rising` hold the detected intervals and the fastest growing words, with
sparkline points included so the board does not need the term index. `events`
is the hand-curated timeline.

## Rules worth keeping

1. The denominator always comes from this snapshot. arXiv's published CSV is a
   build-time check, nothing more.
2. Bucket on the first version's timestamp. Never `update_date`.
3. Provisional months stay out of bursts, rising terms, seasonality and
   forecasts.
4. Missing is not zero. An absent entry means unindexed. A `0` means the month
   really had no papers.
5. Smoothing is presentation only, applied after everything else.

## One known gap

The data files are hashed, but `index.html`, `styles.css` and `js/*.js` are not,
and Pages serves them with a ten minute cache it will not let us override. So
for ten minutes after a deploy someone can hold new data with old code.

It degrades safely: unknown manifest keys read as absent and a missing shard
comes back empty, so search finds nothing rather than showing wrong numbers.
Fixing it properly means propagating a version to imported modules, which
without a bundler needs dynamic imports keyed off `import.meta.url`. Until then,
bump `schema` on any breaking change so old code can tell.

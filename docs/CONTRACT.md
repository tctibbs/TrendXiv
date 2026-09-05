# TrendXiv artifact contract (v1)

All artifacts are emitted to `build/` by `python -m src.pipeline.build` and deployed
to GitHub Pages via `actions/upload-pages-artifact`. **Nothing here is committed to git.**

Filenames are content-hashed: `<name>-<sha256[:10]>.<ext>`. `manifest.json` is the only
stable name and is the single source of truth the frontend reads first.

## manifest.json
{
  "schema": 1,
  "built_at": "2026-09-05T00:00:00Z",
  "source_revision": "<hf revision sha>",
  "corpus_rows": 3148882,
  "data_complete_through": "2026-07-31",   // last FULLY complete month
  "snapshot_max_date": "2026-08-27",
  "provisional_from": "2026-08-01",        // buckets >= this are hatched + excluded from inference
  "periods": ["1991-07", ..., "2026-08"],  // shared month axis, index = position in every series array
  "files": {
    "cube":     "cube-<hash>.json",
    "totals":   "totals-<hash>.json",
    "events":   "events-<hash>.json",
    "taxonomy": "taxonomy-<hash>.json",
    "rising":   "rising-<hash>.json",
    "bursts":   "bursts-<hash>.json",
    "seasonal": "seasonal-<hash>.json",
    "vocab":    "vocab-<hash>.json",
    "terms_shard_pattern": "terms/t-{shard}-<hash>.json"
  }
}

## cube-<hash>.json   (category x month)
{
  "periods_len": 423,
  "series": {
    "cs.LG": {"any": [<int> x periods_len], "primary": [...], "frac": [<float> x N]},
    ...
  }
}
`any`      = paper lists this category anywhere (matches arXiv `cat:` semantics)  <- DEFAULT
`primary`  = paper's primary category only
`frac`     = 1/k fractional attribution (used only for the 100%-stacked landscape)

## totals-<hash>.json  (the denominator - derived from the SAME snapshot, never the official CSV)
{"any": [...], "primary": [...]}   // primary == corpus paper count per month

## terms/t-<shard>-<hash>.json    (shard = first char a-z, "0" for digit, "_" other)
{"transformer": {"o": 180, "v": [1,0,3,...]}, ...}
  o = offset into manifest.periods where this term's vector starts
  v = document-frequency counts (papers mentioning term >=1x), NOT term occurrences

## vocab-<hash>.json
{"terms": {"transformer": {"df": 91234, "s": "t"}}, "below_threshold": ["...", ...]}
  s = shard key.  below_threshold lets the UI say "indexed only above 10 papers"
  rather than silently rendering zeros.

## Rules that must not be violated
1. Denominator is ALWAYS from this snapshot. The official arXiv CSV is a build-time
   integrity check only. (Cross-source shares drift up to 2.6%/month.)
2. Bucket on versions[1].created (v1). Never update_date.
3. Provisional buckets are excluded from every inference path (burst/rising/STL/forecast).
4. Missing != zero. Absent series entry means unindexed; 0 means genuinely zero papers.
5. Smoothing is presentation-only, applied after all inference.

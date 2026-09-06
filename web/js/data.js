/* Artifact loading and the value transforms the UI offers.
 *
 * All statistics are precomputed offline in Python; this file does arithmetic
 * only - shares, indexing, and presentation smoothing. Nothing here may be
 * used to derive an inferential claim.
 */

const BASE = new URL('./data/', new URL('.', import.meta.url)).pathname.replace(/\/js\/$/, '/');

export const store = {
  manifest: null, cube: null, totals: null, taxonomy: null,
  events: null, vocab: null, shards: new Map(),
};

async function json(path) {
  const response = await fetch(path, { cache: 'default' });
  if (!response.ok) throw new Error(`${path}: ${response.status}`);
  return response.json();
}

/** Load the manifest and everything the landing view needs. */
export async function boot(base = 'data') {
  const manifest = await json(`${base}/manifest.json`);
  const at = (name) => `${base}/${manifest.files[name]}`;
  const [cube, totals, taxonomy] = await Promise.all([
    json(at('cube')), json(at('totals')), json(at('taxonomy')),
  ]);
  store.manifest = manifest;
  store.cube = cube.series;
  store.totals = totals;
  store.taxonomy = taxonomy;
  store.base = base;
  if (manifest.files.events) {
    store.events = await json(at('events')).catch(() => null);
  }
  return manifest;
}

/** Lazily fetch the vocabulary, needed only once the user searches. */
export async function loadVocab() {
  if (store.vocab) return store.vocab;
  const { manifest, base } = store;
  if (!manifest.files.vocab) return null;
  store.vocab = await json(`${base}/${manifest.files.vocab}`);
  return store.vocab;
}

/** Fetch one term shard, memoised. One ordinary fetch, no query engine. */
export async function loadShard(key) {
  if (store.shards.has(key)) return store.shards.get(key);
  const promise = json(`${store.base}/terms/t-${key}.json`).catch(() => ({}));
  store.shards.set(key, promise);
  return promise;
}

/** Expand a stored {o, v} sparse vector to the full period axis. */
export function expand(entry, length) {
  const out = new Array(length).fill(0);
  if (!entry) return out;
  for (let i = 0; i < entry.v.length; i++) out[entry.o + i] = entry.v[i];
  return out;
}

/** Look up a term's monthly document-frequency vector. */
export async function termSeries(term) {
  const key = term.toLowerCase();
  const vocab = await loadVocab();
  if (!vocab?.terms?.[key]) return null;
  const shard = await loadShard(shardKey(key));
  return expand(shard[key], store.manifest.periods.length);
}

/** Shard a term belongs to. Mirrors src/pipeline/terms.py::shard_key. */
export function shardKey(term) {
  const head = term[0];
  if (head >= 'a' && head <= 'z') return head;
  return head >= '0' && head <= '9' ? '0' : '_';
}

/**
 * Convert counts to the requested display mode.
 *
 * The denominator is ALWAYS the corpus total from the same snapshot - never
 * the row-sum of whatever series the user happens to have selected. That
 * mistake makes "the cs.LG share" change when a third checkbox is ticked, so
 * two people looking at the same series see different numbers.
 */
export function transform(counts, mode, { attribution = 'any' } = {}) {
  const totals = store.totals;
  const denom = attribution === 'any' ? totals.papers : totals.papers;
  switch (mode) {
    case 'count':
      return counts.slice();
    case 'share':
      return counts.map((k, i) => (denom[i] > 0 ? k / denom[i] : null));
    case 'yoy':
      return counts.map((k, i) => {
        if (i < 12) return null;
        const a = denom[i] > 0 ? k / denom[i] : null;
        const b = denom[i - 12] > 0 ? counts[i - 12] / denom[i - 12] : null;
        return a === null || !b ? null : (a - b) / b;
      });
    default:
      return counts.slice();
  }
}

/**
 * Centred moving average, for presentation only.
 *
 * Centred rather than trailing: a trailing window shifts every feature right by
 * (w-1)/2, which at a 6-month setting misdates a topic's takeoff by 2.5 months.
 */
export function smooth(values, window) {
  if (!window || window < 2) return values;
  const half = Math.floor(window / 2);
  return values.map((_, i) => {
    let sum = 0;
    let count = 0;
    for (let j = Math.max(0, i - half); j <= Math.min(values.length - 1, i + half); j++) {
      const v = values[j];
      if (v !== null && v !== undefined && !Number.isNaN(v)) { sum += v; count++; }
    }
    return count ? sum / count : null;
  });
}

/** Wilson score interval, mirrored from src/analysis/intervals.py. */
export function wilson(k, n, z = 1.96) {
  if (!n) return [null, null];
  const p = k / n;
  const z2 = z * z;
  const d = 1 + z2 / n;
  const centre = (p + z2 / (2 * n)) / d;
  const halfWidth = (z * Math.sqrt((p * (1 - p)) / n + z2 / (4 * n * n))) / d;
  return [Math.max(0, centre - halfWidth), Math.min(1, centre + halfWidth)];
}

/** Index of the first provisional bucket. */
export function provisionalIndex() {
  return store.manifest.periods.indexOf(store.manifest.provisional_from);
}

const EVENT_KIND_LABEL = {
  created: 'created', split_from: 'split off', renamed: 'renamed', aliased_to: 'merged',
};

/**
 * Events resolved onto period indices, filtered to the active scope.
 *
 * Two sources are merged. Curated events give a chart its context, but taxonomy
 * events are a correctness feature: a category's first months are a cliff, not a
 * trend, because arXiv never reclassifies existing papers into a new category.
 * Cosmology appears to leap from nothing to nine percent of arXiv in January
 * 2009, and without the annotation that reads as a discovery rather than as a
 * filing change. Taxonomy events are given top weight so they are never the ones
 * dropped when labels collide.
 */
export function eventsForScope(keys, limit = 4) {
  const periods = store.manifest.periods;
  const active = new Set(keys);

  const curated = (store.events?.events ?? []).filter(
    (e) => !e.scope?.length || e.scope.some((s) => active.has(s)),
  );

  const taxonomy = (store.taxonomy?.events ?? [])
    .filter((e) => active.has(e.code))
    .map((e) => ({
      date: e.date,
      short: `${e.code} ${EVENT_KIND_LABEL[e.kind] ?? e.kind}`,
      label: `${e.code} ${EVENT_KIND_LABEL[e.kind] ?? e.kind}: ${e.note}`,
      weight: 99,
      taxonomy: true,
    }));

  return [...taxonomy, ...curated]
    .map((e) => ({ ...e, index: periods.indexOf(e.date) }))
    .filter((e) => e.index >= 0)
    .sort((a, b) => (b.weight ?? 3) - (a.weight ?? 3))
    .slice(0, limit)
    .sort((a, b) => a.index - b.index);
}

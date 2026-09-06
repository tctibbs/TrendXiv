/* TrendXiv application shell. */

import {
  boot, store, transform, smooth, wilson, provisionalIndex, eventsForScope,
  loadVocab, termSeries, loadShard, expand, shardKey,
} from './data.js';
import {
  lineChart, sparkline, fingerprint, burstTimeline, seriesColor, formatValue, formatPeriod,
} from './chart.js';

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

const state = {
  selection: [],        // [{kind:'category'|'term', key, label}]
  mode: 'share',
  smoothWindow: 3,
  attribution: 'any',
  seasonal: 'raw',
};

/** A month with fewer papers than this cannot support a stable share. */
const MIN_DENOMINATOR = 400;

/** A series matching fewer papers than this over the whole window is drawn faintly. */
const MIN_TOTAL_MATCHES = 25;

const MODE_LABEL = {
  share: 'Share of everything arXiv published that month',
  count: 'Papers per month',
  index: 'Growth since the first month with data, starting at 100',
  yoy: 'Change in share against the same month a year earlier',
};

/** Full name for a category code, falling back to the code itself. */
const nameOf = (code) => store.taxonomy?.names?.[code] ?? code;

/** Compact name for on-chart labels, where horizontal space is tight. */
const shortNameOf = (code) => store.taxonomy?.short_names?.[code] ?? nameOf(code);

/**
 * Build a selectable item for a category.
 *
 * The name leads and the code follows, never one without the other: cs.LG and
 * stat.ML are both officially called "Machine Learning", so a name alone is
 * genuinely ambiguous, while a code alone is unreadable to everyone else.
 */
const categoryItem = (code) => ({
  kind: 'category', key: code, label: nameOf(code), short: shortNameOf(code), code,
});

const termItem = (term) => ({
  kind: 'term', key: term.toLowerCase(), label: term.toLowerCase(), short: term.toLowerCase(),
});

/* ---------------------------------------------------------------- URL state */

function readUrl() {
  const params = new URLSearchParams(location.search);
  const q = params.get('q');
  if (q) {
    state.selection = q.split(',').filter(Boolean).map((raw) => {
      const key = decodeURIComponent(raw);
      return store.cube[key] ? categoryItem(key) : termItem(key);
    });
  }
  state.mode = params.get('mode') ?? state.mode;
  const sm = params.get('smooth');
  if (sm !== null) state.smoothWindow = Number(sm);
}

function writeUrl() {
  const params = new URLSearchParams();
  if (state.selection.length) {
    params.set('q', state.selection.map((s) => encodeURIComponent(s.key)).join(','));
  }
  if (state.mode !== 'share') params.set('mode', state.mode);
  if (state.smoothWindow !== 3) params.set('smooth', String(state.smoothWindow));
  const qs = params.toString();
  history.replaceState(null, '', qs ? `?${qs}` : location.pathname);
}

/* -------------------------------------------------------------- series data */

const seriesCache = new Map();

async function rawCounts(item) {
  const cacheKey = `${item.kind}:${item.key}`;
  if (seriesCache.has(cacheKey)) return seriesCache.get(cacheKey);
  let counts;
  if (item.kind === 'category') {
    counts = store.cube[item.key]?.[state.attribution] ?? null;
  } else {
    counts = await termSeries(item.key);
  }
  seriesCache.set(cacheKey, counts);
  return counts;
}

/** Build render-ready series, including Wilson bands on share mode. */
async function buildSeries(selection) {
  const periods = store.manifest.periods;
  const cutoff = provisionalIndex();
  const out = [];
  for (const [i, item] of selection.entries()) {
    const counts = await rawCounts(item);
    if (!counts) { out.push(null); continue; }
    const values = smooth(transform(counts, state.mode, state), state.smoothWindow);
    const entry = {
      key: item.key, label: item.short ?? item.label, fullLabel: item.label,
      code: item.code, values, color: seriesColor(i), counts, kind: item.kind,
    };
    // Two masks, both about not asserting more than the data supports.
    // 1. The incomplete trailing months are never drawn.
    // 2. A share computed from a handful of papers is noise, not signal: in
    //    1994 arXiv published a few hundred papers a month, so one cs.CL paper
    //    renders as a 19% spike that dwarfs the entire modern era.
    const drawable = (j) => {
      if (j >= cutoff) return false;
      if (state.mode === 'count') return true;
      const denom = store.totals.papers[j];
      return Boolean(denom) && denom >= MIN_DENOMINATOR;
    };
    entry.values = entry.values.map((v, j) => (drawable(j) ? v : null));

    // The band has to obey the same mask as the line it belongs to. A Wilson
    // interval over a 50-paper month reaches most of the way to 100%, and since
    // the y-scale is fitted to the bands as well as the lines, one unmasked
    // month flattens the entire modern era against the axis.
    if (state.mode === 'share' && !state.smoothWindow) {
      const denom = store.totals.papers;
      const bands = counts.map((k, j) => (drawable(j) ? wilson(k, denom[j]) : [null, null]));
      entry.lo = bands.map((b) => b[0]);
      entry.hi = bands.map((b) => b[1]);
    }
    entry.sparse = counts.slice(0, cutoff).reduce((a, b) => a + b, 0) < MIN_TOTAL_MATCHES;
    out.push(entry);
  }
  return out.filter(Boolean);
}

/* --------------------------------------------------------------------- hints */

const hint = document.createElement('div');
hint.className = 'hint';
hint.setAttribute('role', 'tooltip');

/**
 * Show a short explanation for any element carrying data-hint.
 *
 * Bound once on the document so controls rendered later are covered without
 * rewiring, and on focus as well as hover so the keyboard path works too.
 */
function wireHints() {
  document.body.appendChild(hint);

  const show = (target) => {
    const text = target.getAttribute('data-hint');
    if (!text) return;
    hint.textContent = text;
    hint.classList.add('on');
    const box = target.getBoundingClientRect();
    const width = hint.offsetWidth;
    const height = hint.offsetHeight;
    // Prefer above; flip below when there is no room at the top of the viewport.
    const above = box.top > height + 12;
    hint.style.top = `${above ? box.top - height - 8 : box.bottom + 8}px`;
    hint.style.left = `${Math.max(8,
      Math.min(window.innerWidth - width - 8, box.left + box.width / 2 - width / 2))}px`;
  };
  const hide = () => hint.classList.remove('on');

  document.addEventListener('pointerover', (event) => {
    const target = event.target.closest('[data-hint]');
    if (target) show(target); else hide();
  });
  document.addEventListener('focusin', (event) => {
    const target = event.target.closest('[data-hint]');
    if (target) show(target);
  });
  document.addEventListener('focusout', hide);
  document.addEventListener('pointerleave', hide);
  window.addEventListener('scroll', hide, { passive: true });
}

/* ------------------------------------------------------------------ tooltip */

const tooltip = document.createElement('div');
tooltip.className = 'tooltip';

function showTooltip(host, index, event, series) {
  if (index === null) { tooltip.classList.remove('on'); return; }
  const periods = store.manifest.periods;
  const denom = store.totals.papers[index];
  const rows = series
    .map((s) => ({ s, v: s.values[index], raw: s.counts?.[index] }))
    .filter((r) => r.v !== null && r.v !== undefined && !Number.isNaN(r.v))
    .sort((a, b) => b.v - a.v)
    .slice(0, 6);
  if (!rows.length) { tooltip.classList.remove('on'); return; }

  tooltip.innerHTML = `<div class="tt-date">${formatPeriod(periods[index])}</div>${rows
    .map((r) => `<div class="tt-row">
        <span class="swatch" style="background:${r.s.color}"></span>
        <span class="name">${escape(r.s.fullLabel ?? r.s.label)}</span>
        <span class="val">${formatValue(r.v, state.mode)}</span>
      </div>${
        state.mode === 'share' && r.raw !== undefined
          ? `<div class="tt-row"><span class="swatch" style="opacity:0"></span>
               <span class="raw">${r.raw.toLocaleString()} of ${denom?.toLocaleString() ?? '?'} papers</span></div>`
          : ''
      }`)
    .join('')}`;

  const rect = host.getBoundingClientRect();
  tooltip.classList.add('on');
  const tw = tooltip.offsetWidth;
  let left = event.clientX - rect.left + 14;
  if (left + tw > rect.width) left = event.clientX - rect.left - tw - 14;
  tooltip.style.left = `${Math.max(0, left)}px`;
  tooltip.style.top = `${Math.max(0, event.clientY - rect.top - 12)}px`;
}

const escape = (s) => String(s).replace(/[&<>"]/g, (c) =>
  ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

/* ------------------------------------------------------------------ explorer */

let currentSeries = [];

async function renderExplorer() {
  const host = $('#explorer-chart');
  if (!state.selection.length) {
    host.innerHTML = '<p class="notice">Search for a field or a word above to start comparing.</p>';
    $('#explorer-chips').innerHTML = '';
    return;
  }
  currentSeries = await buildSeries(state.selection);
  host.style.position = 'relative';
  if (!host.contains(tooltip)) host.appendChild(tooltip);

  const events = eventsForScope(state.selection.map((s) => s.key), 4);
  lineChart(host, {
    periods: store.manifest.periods,
    series: currentSeries,
    provisionalFrom: provisionalIndex(),
    events,
    mode: state.mode,
    yZero: state.mode !== 'yoy',
    ariaLabel: `${MODE_LABEL[state.mode]} for ${state.selection.map((s) => s.label).join(', ')}`,
    onHover: (i, ev) => showTooltip(host, i, ev, currentSeries),
  });
  host.appendChild(tooltip);

  $('#explorer-sub').textContent = state.smoothWindow
    ? `${MODE_LABEL[state.mode]}, averaged over ${state.smoothWindow} months`
    : `${MODE_LABEL[state.mode]}, month by month, with a 95% confidence band`;
  renderChips();
  renderDataTable();
  writeUrl();
}

function renderChips() {
  const host = $('#explorer-chips');
  host.innerHTML = '';
  state.selection.forEach((item, i) => {
    const chip = document.createElement('span');
    chip.className = 'chip';
    chip.innerHTML = `<span class="swatch" style="background:${seriesColor(i)}"></span>
      <span>${escape(item.short ?? item.label)}</span>${
        item.code ? `<span class="chip-code">${escape(item.code)}</span>` : ''
      }`;
    if (item.code) chip.title = `${item.label} (${item.code})`;
    const close = document.createElement('button');
    close.type = 'button';
    close.setAttribute('aria-label', `Remove ${item.label}`);
    close.textContent = '×';
    close.onclick = () => { state.selection.splice(i, 1); renderExplorer(); };
    chip.appendChild(close);
    // Hover-to-dim: isolates one line without removing the others' context.
    chip.onpointerenter = () => {
      currentSeries.forEach((s, j) => { s.dim = j !== i; });
      redrawOnly();
    };
    chip.onpointerleave = () => {
      currentSeries.forEach((s) => { s.dim = false; });
      redrawOnly();
    };
    host.appendChild(chip);
  });
}

function redrawOnly() {
  const host = $('#explorer-chart');
  lineChart(host, {
    periods: store.manifest.periods,
    series: currentSeries,
    provisionalFrom: provisionalIndex(),
    events: eventsForScope(state.selection.map((s) => s.key), 4),
    mode: state.mode,
    yZero: state.mode !== 'yoy',
    onHover: (i, ev) => showTooltip(host, i, ev, currentSeries),
  });
  host.appendChild(tooltip);
}

/** The hidden table doubles as a screen-reader affordance and a no-JS fallback. */
function renderDataTable() {
  const table = $('#explorer-table');
  const periods = store.manifest.periods;
  const step = Math.max(1, Math.floor(periods.length / 40));
  const head = `<tr><th>Month</th>${currentSeries.map((s) => `<th>${escape(s.label)}</th>`).join('')}</tr>`;
  const rows = periods.filter((_, i) => i % step === 0).map((p, idx) => {
    const i = idx * step;
    return `<tr><td>${p}</td>${currentSeries
      .map((s) => `<td>${formatValue(s.values[i], state.mode)}</td>`).join('')}</tr>`;
  }).join('');
  table.innerHTML = `<caption>${MODE_LABEL[state.mode]}</caption>${head}${rows}`;
}

function addSelection(item) {
  if (state.selection.some((s) => s.key === item.key)) return;
  if (state.selection.length >= 8) state.selection.shift();
  state.selection.push(item);
  renderExplorer();
}

/* -------------------------------------------------------------------- search */

let searchTimer;

async function runSearch(query) {
  const box = $('#search-results');
  const q = query.trim().toLowerCase();
  if (q.length < 2) { box.innerHTML = ''; return; }

  // Match on the name as well as the code, so "machine learning" finds cs.LG
  // for a reader who has never seen an arXiv code.
  const categories = Object.keys(store.cube)
    .filter((c) => c.toLowerCase().includes(q) || nameOf(c).toLowerCase().includes(q))
    .sort((a, b) => sum(store.cube[b].any) - sum(store.cube[a].any))
    .slice(0, 5)
    .map((c) => ({ ...categoryItem(c), df: sum(store.cube[c].any) }));

  const vocab = await loadVocab();
  let terms = [];
  if (vocab) {
    const exact = vocab.terms[q];
    const prefix = Object.keys(vocab.terms)
      .filter((t) => t.startsWith(q) && t !== q)
      .sort((a, b) => vocab.terms[b] - vocab.terms[a])
      .slice(0, 7);
    terms = [...(exact ? [q] : []), ...prefix]
      .map((t) => ({ ...termItem(t), df: vocab.terms[t] }));
  }

  const items = [...categories, ...terms].slice(0, 10);
  if (!items.length) {
    const belowThreshold = vocab && vocab.min_df;
    box.innerHTML = `<div class="result"><span class="term" style="color:var(--ink-muted)">
      Nothing indexed under that name.${
        belowThreshold
          ? ` Words used in fewer than ${vocab.min_df} papers are left out of the index.`
          : ''
      }
    </span></div>`;
    return;
  }

  box.innerHTML = '';
  for (const item of items) {
    const row = document.createElement('button');
    row.type = 'button';
    row.className = 'result';
    row.innerHTML = `<span class="term">${escape(item.short ?? item.label)}</span>${
      item.code ? `<span class="chip-code">${escape(item.code)}</span>` : ''
    }<span class="df">${item.df.toLocaleString()} papers</span>`;
    const counts = item.kind === 'category'
      ? store.cube[item.key].any
      : expand((await loadShard(shardKey(item.key)))[item.key], store.manifest.periods.length);
    row.prepend(sparkline(smooth(counts, 6), { color: 'var(--accent)' }));
    row.onclick = () => {
      addSelection(item);
      $('#search-input').value = '';
      box.innerHTML = '';
    };
    box.appendChild(row);
  }
}

const sum = (arr) => arr.reduce((a, b) => a + b, 0);

/* ---------------------------------------------------------------- landing UI */

function renderHero() {
  const periods = store.manifest.periods;
  const totals = store.totals.papers;
  const host = $('#hero-chart');
  host.style.position = 'relative';
  const series = [{
    key: 'arxiv', label: 'All arXiv', fullLabel: 'All of arXiv',
    values: smooth(totals, 3), counts: totals,
    color: 'var(--ink)', width: 1.8,
  }];
  const cutoff = provisionalIndex();
  series[0].values = series[0].values.map((v, i) => (i >= cutoff ? null : v));
  lineChart(host, {
    periods, series, provisionalFrom: cutoff,
    events: eventsForScope([], 5), mode: 'count', height: 320,
    ariaLabel: 'Monthly arXiv submissions since 1991',
    onHover: (i, ev) => showTooltip(host, i, ev, series),
  });
  host.appendChild(tooltip);

  const last12 = totals.slice(cutoff - 12, cutoff);
  const prev12 = totals.slice(cutoff - 24, cutoff - 12);
  const yoy = (sum(last12) - sum(prev12)) / sum(prev12);
  $('#stat-papers').textContent = store.manifest.corpus_rows.toLocaleString();
  $('#stat-rate').textContent = Math.round(sum(last12) / 12).toLocaleString();
  $('#stat-yoy').textContent = `${yoy > 0 ? '+' : ''}${(yoy * 100).toFixed(1)}%`;
  $('#stat-cats').textContent = store.manifest.categories.toLocaleString();
}

function renderLandscape() {
  const groups = store.taxonomy.groups;
  const map = store.taxonomy.category_group;
  const periods = store.manifest.periods;
  const byGroup = {};
  for (const [code, entry] of Object.entries(store.cube)) {
    const g = map[code];
    if (!g) continue;
    byGroup[g] ??= new Array(periods.length).fill(0);
    for (let i = 0; i < periods.length; i++) byGroup[g][i] += entry.frac[i];
  }
  const cutoff = provisionalIndex();
  const top = Object.entries(byGroup)
    .sort((a, b) => sum(b[1].slice(cutoff - 36, cutoff)) - sum(a[1].slice(cutoff - 36, cutoff)))
    .slice(0, 8);

  const series = top.map(([g, values], i) => ({
    key: g, label: groups[g] ?? g, fullLabel: groups[g] ?? g, counts: values,
    values: smooth(values.map((v, j) => (store.totals.papers[j] ? v / store.totals.papers[j] : null)), 6)
      .map((v, j) => (j >= cutoff ? null : v)),
    color: seriesColor(i),
  }));
  const host = $('#landscape-chart');
  host.style.position = 'relative';
  lineChart(host, {
    periods, series, provisionalFrom: cutoff, mode: 'share', height: 340,
    ariaLabel: 'Share of arXiv submissions by broad area',
    onHover: (i, ev) => showTooltip(host, i, ev, series),
  });
  host.appendChild(tooltip);
}

async function renderFingerprints() {
  const host = $('#fingerprints');
  const picks = ['cs.CL', 'cs.LG', 'cs.CV', 'cs.RO', 'math.AP', 'astro-ph.GA']
    .filter((c) => store.cube[c]);
  host.innerHTML = '';
  const seasonal = store.manifest.files.seasonal
    ? await fetch(`${store.base}/${store.manifest.files.seasonal}`).then((r) => r.json()).catch(() => null)
    : null;

  picks.forEach((code, i) => {
    const index = seasonal?.[code]?.month_index ?? monthIndexFallback(store.cube[code].any);
    const card = document.createElement('div');
    card.className = 'multiple';
    const peakMonth = Object.entries(index).sort((a, b) => b[1] - a[1])[0];
    const names = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
    const amp = Math.max(...Object.values(index)) / Math.min(...Object.values(index));
    card.innerHTML = `<h4>${escape(shortNameOf(code))}</h4>
      <div class="sub">${escape(code)} &middot; PEAKS IN
        ${names[Number(peakMonth[0]) - 1].toUpperCase()} &middot; ${amp.toFixed(2)}x SWING</div>
      <div class="fp"></div>`;
    host.appendChild(card);
    fingerprint($('.fp', card), index, { color: seriesColor(i), size: 190 });
  });
}

/** Crude month-of-year index used until the Python seasonal artifact exists. */
function monthIndexFallback(counts) {
  const periods = store.manifest.periods;
  const cutoff = provisionalIndex();
  const start = Math.max(0, cutoff - 96);
  const byMonth = Array.from({ length: 12 }, () => []);
  for (let i = start; i < cutoff; i++) {
    const trailing = counts.slice(Math.max(0, i - 6), i + 6).filter((v) => v > 0);
    if (trailing.length < 6 || !counts[i]) continue;
    const local = trailing.reduce((a, b) => a + b, 0) / trailing.length;
    if (local > 0) byMonth[Number(periods[i].slice(5)) - 1].push(counts[i] / local);
  }
  const raw = byMonth.map((xs) => (xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : 1));
  const geo = Math.exp(raw.reduce((a, b) => a + Math.log(b || 1), 0) / 12);
  return Object.fromEntries(raw.map((v, i) => [i + 1, v / geo]));
}

async function renderBursts() {
  const host = $('#burst-timeline');
  const file = store.manifest.files.bursts;
  if (!file) {
    host.innerHTML = `<p class="notice">This build does not include the word index that burst
      detection runs over.</p>`;
    return;
  }
  const data = await fetch(`${store.base}/${file}`).then((r) => r.json()).catch(() => null);
  if (!data?.rows?.length) {
    host.innerHTML = '<p class="notice">No bursts found in this build.</p>';
    return;
  }
  const rows = [...data.rows]
    .sort((a, b) => store.manifest.periods.indexOf(a.start) - store.manifest.periods.indexOf(b.start));
  burstTimeline(host, rows, store.manifest.periods, (term) => {
    addSelection(termItem(term));
    $('#explorer').scrollIntoView({ behavior: 'smooth' });
  });
}

async function renderRising() {
  const host = $('#rising-board');
  const file = store.manifest.files.rising;
  if (!file) {
    host.innerHTML = `<p class="notice">This build does not include the rising board, which is
      produced by the analysis step of the pipeline.</p>`;
    return;
  }
  const data = await fetch(`${store.base}/${file}`).then((r) => r.json()).catch(() => null);
  if (!data?.rows?.length) {
    host.innerHTML = '<p class="notice">No word grew by enough to clear the bar this time.</p>';
    return;
  }
  host.innerHTML = `
    <table class="board">
      <thead><tr><th>Term</th><th>Last ${data.recent_months + data.baseline_months} months</th>
        <th class="num">Share now</th><th class="num">Share before</th>
        <th class="num">Log-odds</th><th></th></tr></thead>
      <tbody></tbody></table>`;
  const body = $('tbody', host);
  for (const row of data.rows.slice(0, 20)) {
    const tr = document.createElement('tr');
    tr.innerHTML = `<td class="term-cell">${escape(row.term)}</td>
      <td class="spark"></td>
      <td class="num">${(row.share_recent * 100).toFixed(2)}%</td>
      <td class="num" style="color:var(--ink-faint)">${(row.share_base * 100).toFixed(2)}%</td>
      <td class="num">${row.delta_logodds > 0 ? '+' : ''}${row.delta_logodds.toFixed(2)}</td>
      <td>${row.is_new ? '<span class="tag new">New</span>' : ''}</td>`;
    const counts = await termSeries(row.term);
    if (counts) {
      // Scoped to the recent window: over the full 35-year axis every recent
      // riser is a flat line with a spike at the right edge, which tells the
      // reader nothing about how the rise actually unfolded.
      const window = data.recent_months + data.baseline_months;
      const recent = smooth(counts.slice(0, provisionalIndex()), 3).slice(-window);
      $('.spark', tr).appendChild(sparkline(recent, { color: 'var(--s2)', width: 90 }));
    }
    $('.term-cell', tr).onclick = () => {
      addSelection(termItem(row.term));
      $('#explorer').scrollIntoView({ behavior: 'smooth' });
    };
    body.appendChild(tr);
  }
}

/* --------------------------------------------------------------------- wiring */

/** Push current state onto the segmented controls. */
function syncControls() {
  $$('#mode-control button').forEach((b) =>
    b.setAttribute('aria-pressed', String(b.dataset.mode === state.mode)));
  $$('#smooth-control button').forEach((b) =>
    b.setAttribute('aria-pressed', String(Number(b.dataset.smooth) === state.smoothWindow)));
}

function wireControls() {
  $$('#mode-control button').forEach((button) => {
    button.onclick = () => {
      state.mode = button.dataset.mode;
      $$('#mode-control button').forEach((b) =>
        b.setAttribute('aria-pressed', String(b === button)));
      renderExplorer();
    };
  });
  $$('#smooth-control button').forEach((button) => {
    button.onclick = () => {
      state.smoothWindow = Number(button.dataset.smooth);
      $$('#smooth-control button').forEach((b) =>
        b.setAttribute('aria-pressed', String(b === button)));
      renderExplorer();
    };
  });
  // Theme is a viewer preference, so it lives in localStorage and never in the
  // URL: a shared permalink must not force the sender's theme on the recipient.
  $('#theme-toggle').onclick = () => {
    const root = document.documentElement;
    const dark = root.getAttribute('data-theme') === 'dark'
      || (!root.hasAttribute('data-theme')
          && matchMedia('(prefers-color-scheme: dark)').matches);
    const next = dark ? 'light' : 'dark';
    root.setAttribute('data-theme', next);
    try { localStorage.setItem('trendxiv-theme', next); } catch { /* private mode */ }
    renderHero();
    renderLandscape();
    renderFingerprints();
    renderBursts();
    if (state.selection.length) redrawOnly();
  };

  const input = $('#search-input');
  input.oninput = () => {
    clearTimeout(searchTimer);
    searchTimer = setTimeout(() => runSearch(input.value), 110);
  };
  input.onkeydown = (event) => {
    if (event.key === 'Escape') { $('#search-results').innerHTML = ''; input.blur(); }
    if (event.key === 'Enter') {
      const first = $('#search-results .result');
      if (first) first.click();
    }
  };
  document.addEventListener('click', (event) => {
    if (!event.target.closest('.search')) $('#search-results').innerHTML = '';
  });
  window.addEventListener('resize', debounce(() => {
    if (state.selection.length) redrawOnly();
    renderHero();
    renderLandscape();
  }, 180));
}

function debounce(fn, ms) {
  let t;
  return (...args) => { clearTimeout(t); t = setTimeout(() => fn(...args), ms); };
}

function renderQuickPicks() {
  const host = $('#quick-picks');
  const picks = [
    { label: 'The machine learning fields', q: ['cs.LG', 'cs.CV', 'cs.CL'] },
    { label: 'Core physics', q: ['hep-th', 'astro-ph.GA', 'cond-mat.mtrl-sci'] },
    { label: 'The rise of language models', q: ['transformer', 'attention', 'llm'] },
    { label: 'Ways of generating things', q: ['diffusion', 'gan', 'autoencoder'] },
    { label: 'Quantum everything', q: ['quant-ph', 'qubit', 'entanglement'] },
  ];
  host.innerHTML = '';
  for (const pick of picks) {
    const button = document.createElement('button');
    button.className = 'chip';
    button.style.cursor = 'pointer';
    button.textContent = pick.label;
    button.onclick = () => {
      state.selection = pick.q.map((key) => (store.cube[key] ? categoryItem(key) : termItem(key)));
      renderExplorer();
      $('#explorer').scrollIntoView({ behavior: 'smooth', block: 'start' });
    };
    host.appendChild(button);
  }
}

async function main() {
  try {
    const manifest = await boot('data');
    const stamp = $('#build-stamp');
    stamp.textContent = `LAST UPDATED ${manifest.built_at.slice(0, 10)}`;
    // The gap between the build date and the newest complete month is real and
    // worth being able to find, just not worth the masthead space.
    stamp.title =
      `Complete through ${manifest.data_complete_through}. `
      + `Built from arXiv snapshot ${manifest.source_revision.slice(0, 10)}.`;
    $$('.skeleton').forEach((n) => n.classList.remove('skeleton'));
    readUrl();
    // A shared link carries mode and smoothing, so the controls have to follow
    // the restored state rather than their markup defaults.
    syncControls();
    renderHero();
    renderLandscape();
    renderQuickPicks();
    wireControls();
    wireHints();
    if (!state.selection.length) {
      state.selection = ['cs.LG', 'cs.CV', 'cs.CL'].filter((c) => store.cube[c]).map(categoryItem);
    }
    await renderExplorer();
    renderFingerprints();
    renderBursts();
    renderRising();
  } catch (error) {
    document.body.insertAdjacentHTML('afterbegin',
      `<div class="wrap"><p class="notice">The data did not load: ${escape(error.message)}.
       Run <code>python -m src.pipeline.build</code>, then serve the <code>web/</code>
       folder.</p></div>`);
    console.error(error);
  }
}

main();

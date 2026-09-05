/* Minimal SVG chart primitives.
 *
 * Hand-rolled rather than library-backed: every mark this product needs -
 * direct labels with collision avoidance, event flags, hatched provisional
 * regions, uncertainty bands - is a custom mark in any charting library, so a
 * library would cost ~480 KB and still leave the hard parts to write. This is
 * ~14 KB and does exactly what the design calls for.
 */

const NS = 'http://www.w3.org/2000/svg';
const SERIES_VARS = ['--s1', '--s2', '--s3', '--s4', '--s5', '--s6', '--s7', '--s8'];

export const seriesColor = (i) => `var(${SERIES_VARS[i % SERIES_VARS.length]})`;

function el(name, attrs = {}, parent = null) {
  const node = document.createElementNS(NS, name);
  for (const [key, value] of Object.entries(attrs)) {
    if (value !== null && value !== undefined) node.setAttribute(key, value);
  }
  if (parent) parent.appendChild(node);
  return node;
}

/** Nice round tick values covering [lo, hi]. */
function ticks(lo, hi, count = 5) {
  if (!(hi > lo)) return [lo];
  const raw = (hi - lo) / count;
  const mag = 10 ** Math.floor(Math.log10(raw));
  const norm = raw / mag;
  const step = (norm >= 7.5 ? 10 : norm >= 3.5 ? 5 : norm >= 1.5 ? 2 : 1) * mag;
  const out = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-9; v += step) {
    out.push(Number(v.toFixed(10)));
  }
  return out;
}

/** Year gridline positions for a monthly axis, thinned to fit the width. */
function yearTicks(periods, width) {
  const years = [];
  periods.forEach((p, i) => {
    if (p.endsWith('-01')) years.push({ index: i, label: p.slice(0, 4) });
  });
  const every = Math.max(1, Math.ceil(years.length / Math.max(3, Math.floor(width / 62))));
  return years.filter((_, i) => i % every === 0);
}

export function formatValue(v, mode) {
  if (v === null || v === undefined || Number.isNaN(v)) return '—';
  if (mode === 'share') return `${(v * 100).toFixed(v * 100 < 1 ? 2 : 1)}%`;
  if (mode === 'per1k') return `${(v * 1000).toFixed(1)}`;
  return v >= 1000 ? Math.round(v).toLocaleString() : String(Math.round(v));
}

export function formatPeriod(p) {
  const [y, m] = p.split('-');
  const names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
  return `${names[Number(m) - 1]} ${y}`;
}

/**
 * Draw a multi-series line chart.
 *
 * @param {HTMLElement} host Container; cleared and re-rendered.
 * @param {object} spec
 * @param {string[]} spec.periods Month labels, the shared x axis.
 * @param {Array} spec.series [{key,label,values,lo,hi,color}] - lo/hi optional bands.
 * @param {number} spec.provisionalFrom Index from which data is incomplete.
 * @param {Array} spec.events [{index,short,label}] timeline annotations.
 * @param {string} spec.mode Value formatting mode.
 * @param {function} spec.onHover Called with (periodIndex|null).
 */
export function lineChart(host, spec) {
  const {
    periods, series, provisionalFrom = Infinity, events = [],
    mode = 'share', height: fixedHeight, onHover, yZero = true,
  } = spec;

  host.textContent = '';
  const width = Math.max(320, host.clientWidth || 800);
  // Never a hardcoded height: a fixed aspect changes the perceived slope of
  // every trend as the viewport changes.
  const height = fixedHeight ?? Math.min(0.52 * window.innerHeight, Math.max(260, 0.40 * width));
  const isNarrow = width < 620;
  const m = { top: 16, right: isNarrow ? 12 : 116, bottom: 26, left: 4 };

  const svg = el('svg', {
    viewBox: `0 0 ${width} ${height}`, role: 'img',
    'aria-label': spec.ariaLabel || 'Time series chart',
  }, host);

  const iw = width - m.left - m.right;
  const ih = height - m.top - m.bottom;
  const x = (i) => m.left + (periods.length < 2 ? 0 : (i / (periods.length - 1)) * iw);

  let lo = Infinity;
  let hi = -Infinity;
  for (const s of series) {
    for (let i = 0; i < s.values.length; i++) {
      const vals = [s.values[i], s.lo?.[i], s.hi?.[i]];
      for (const v of vals) {
        if (v === null || v === undefined || Number.isNaN(v)) continue;
        if (v < lo) lo = v;
        if (v > hi) hi = v;
      }
    }
  }
  if (!Number.isFinite(lo)) { lo = 0; hi = 1; }
  if (yZero) lo = Math.min(0, lo);
  if (hi === lo) hi = lo + 1;
  hi += (hi - lo) * 0.08;

  const y = (v) => m.top + ih - ((v - lo) / (hi - lo)) * ih;

  // Gridlines: horizontal only. No spines, no tick marks, no rotated text.
  const yt = ticks(lo, hi, isNarrow ? 3 : 5);
  for (const t of yt) {
    const py = y(t);
    el('line', { x1: m.left, x2: m.left + iw, y1: py, y2: py,
      class: t === 0 ? 'zero-line' : 'grid-line' }, svg);
    // Labels sit inside the plot area, which saves a 60px left margin.
    el('text', { x: m.left + 2, y: py - 4, class: 'axis-label' }, svg)
      .textContent = formatValue(t, mode);
  }

  for (const { index, label } of yearTicks(periods, iw)) {
    el('text', { x: x(index), y: height - 8, class: 'axis-label', 'text-anchor': 'middle' }, svg)
      .textContent = label;
  }

  // Provisional region: the last buckets are incomplete, and an unmarked
  // partial month reads as a collapse (arXiv's own 2026-09 sits 87% below
  // 2026-08 for this reason alone).
  if (provisionalFrom < periods.length) {
    const defs = el('defs', {}, svg);
    const pat = el('pattern', { id: 'hatch', width: 5, height: 5,
      patternUnits: 'userSpaceOnUse', patternTransform: 'rotate(45)' }, defs);
    el('line', { x1: 0, y1: 0, x2: 0, y2: 5, stroke: 'var(--rule-strong)', 'stroke-width': 1.2 }, pat);
    const px = x(provisionalFrom);
    el('rect', { x: px, y: m.top, width: Math.max(0, m.left + iw - px), height: ih,
      fill: 'url(#hatch)', opacity: 0.35 }, svg);
    if (m.left + iw - px > 46) {
      el('text', { x: px + 4, y: m.top + 11, class: 'event-flag' }, svg).textContent = 'PARTIAL';
    }
  }

  // Events always draw their rule; only the LABEL is dropped when two would
  // overlap, so information is lost from the text layer and never from the chart.
  const labelled = [];
  for (const ev of events) {
    if (ev.index < 0 || ev.index >= periods.length) continue;
    const ex = x(ev.index);
    el('line', { x1: ex, x2: ex, y1: m.top, y2: m.top + ih, class: 'event-rule' }, svg);
    // IBM Plex Mono at 10px with 0.06em tracking runs ~6.8px per character;
    // the extra 12px is the minimum gap that still reads as two labels.
    const room = ev.short.length * 6.8 + 12;
    const clash = labelled.some((px) => Math.abs(px - ex) < room);
    const anchor = ex > m.left + iw - 60 ? 'end' : 'start';
    const t = el('text', {
      x: ex + (anchor === 'end' ? -4 : 4), y: m.top + 10,
      class: 'event-flag', 'text-anchor': anchor,
    }, svg);
    if (!clash) { t.textContent = ev.short; labelled.push(ex); }
    el('title', {}, t).textContent = `${ev.label} (${periods[ev.index]})`;
  }

  const line = (values) => {
    let d = '';
    let pen = false;
    values.forEach((v, i) => {
      if (v === null || v === undefined || Number.isNaN(v)) { pen = false; return; }
      d += `${pen ? 'L' : 'M'}${x(i).toFixed(2)},${y(v).toFixed(2)}`;
      pen = true;
    });
    return d;
  };

  series.forEach((s, si) => {
    const color = s.color ?? seriesColor(si);
    if (s.lo && s.hi) {
      let top = '';
      let bot = '';
      let started = false;
      for (let i = 0; i < s.lo.length; i++) {
        const a = s.hi[i];
        const b = s.lo[i];
        if (a === null || b === null || Number.isNaN(a) || Number.isNaN(b)) continue;
        top += `${started ? 'L' : 'M'}${x(i).toFixed(2)},${y(a).toFixed(2)}`;
        bot = `L${x(i).toFixed(2)},${y(b).toFixed(2)}${bot}`;
        started = true;
      }
      if (started) el('path', { d: `${top}${bot}Z`, fill: color, class: 'band' }, svg);
    }
    el('path', {
      d: line(s.values), fill: 'none', stroke: color, 'stroke-width': s.width ?? 2,
      'stroke-linejoin': 'round', 'stroke-linecap': 'round', opacity: s.dim ? 0.22 : 1,
    }, svg);
  });

  // Direct labels at the right terminus. The legend-swatch-to-line eye jump is
  // the largest readability tax on a multi-series chart.
  if (!isNarrow) {
    const placed = [];
    series.forEach((s, si) => {
      let last = -1;
      for (let i = s.values.length - 1; i >= 0; i--) {
        const v = s.values[i];
        if (v !== null && v !== undefined && !Number.isNaN(v)) { last = i; break; }
      }
      if (last < 0) return;
      let py = y(s.values[last]);
      // Iterative nudge with a 15px floor keeps labels legible when lines converge.
      for (let guard = 0; guard < 60; guard++) {
        const clash = placed.find((p) => Math.abs(p - py) < 15);
        if (clash === undefined) break;
        py = clash + (py >= clash ? 15 : -15);
      }
      py = Math.max(m.top + 9, Math.min(m.top + ih, py));
      placed.push(py);
      const label = el('text', {
        x: m.left + iw + 8, y: py + 4, class: 'series-label',
        fill: s.color ?? seriesColor(si), opacity: s.dim ? 0.3 : 1,
      }, svg);
      label.textContent = s.label.length > 15 ? `${s.label.slice(0, 14)}…` : s.label;
    });
  }

  const focusLine = el('line', {
    y1: m.top, y2: m.top + ih, class: 'grid-line', stroke: 'var(--ink-faint)', opacity: 0,
  }, svg);
  const dots = series.map((s, si) => el('circle', {
    r: 3.5, fill: s.color ?? seriesColor(si), stroke: 'var(--bg)', 'stroke-width': 1.5, opacity: 0,
  }, svg));

  if (onHover) {
    const toIndex = (event) => {
      const rect = host.getBoundingClientRect();
      const px = ((event.clientX - rect.left) / rect.width) * width;
      const i = Math.round(((px - m.left) / iw) * (periods.length - 1));
      return Math.max(0, Math.min(periods.length - 1, i));
    };
    const move = (event) => {
      const i = toIndex(event);
      focusLine.setAttribute('x1', x(i));
      focusLine.setAttribute('x2', x(i));
      focusLine.setAttribute('opacity', 0.5);
      series.forEach((s, si) => {
        const v = s.values[i];
        const ok = v !== null && v !== undefined && !Number.isNaN(v);
        dots[si].setAttribute('opacity', ok && !s.dim ? 1 : 0);
        if (ok) { dots[si].setAttribute('cx', x(i)); dots[si].setAttribute('cy', y(v)); }
      });
      onHover(i, event);
    };
    host.onpointermove = move;
    host.onpointerdown = move;
    host.onpointerleave = () => {
      focusLine.setAttribute('opacity', 0);
      dots.forEach((d) => d.setAttribute('opacity', 0));
      onHover(null);
    };
  }
  return svg;
}

/** A 60x18 sparkline. Used in search results, which is what makes the input feel like data. */
export function sparkline(values, { width = 62, height = 18, color = 'currentColor' } = {}) {
  const svg = el('svg', { viewBox: `0 0 ${width} ${height}`, width, height, 'aria-hidden': 'true' });
  const clean = values.filter((v) => v !== null && !Number.isNaN(v));
  if (!clean.length) return svg;
  const lo = Math.min(0, ...clean);
  const hi = Math.max(...clean) || 1;
  const d = values.map((v, i) => {
    const px = (i / Math.max(1, values.length - 1)) * width;
    const py = height - 1 - ((v - lo) / (hi - lo || 1)) * (height - 2);
    return `${i ? 'L' : 'M'}${px.toFixed(1)},${py.toFixed(1)}`;
  }).join('');
  el('path', { d, fill: 'none', stroke: color, 'stroke-width': 1.4, 'stroke-linejoin': 'round' }, svg);
  return svg;
}

/**
 * Radial month-of-year plot - the "Deadline Fingerprint".
 *
 * @param {HTMLElement} host Container.
 * @param {object} index Map of month number (1-12) to multiplicative index.
 * @param {object} opts {color, size, label}
 */
export function fingerprint(host, index, { color = 'var(--s1)', size = 200 } = {}) {
  host.textContent = '';
  const svg = el('svg', { viewBox: `0 0 ${size} ${size}`, role: 'img',
    'aria-label': 'Month-of-year seasonal index' }, host);
  const cx = size / 2;
  const cy = size / 2;
  const rMax = size / 2 - 24;
  const values = Array.from({ length: 12 }, (_, i) => index[i + 1] ?? 1);
  const peak = Math.max(1.02, ...values);
  const scale = (v) => (v / peak) * rMax;
  const angle = (i) => (i / 12) * 2 * Math.PI - Math.PI / 2;

  // The unit circle is the "no seasonality" reference; without it the shape has no meaning.
  el('circle', { cx, cy, r: scale(1), fill: 'none', stroke: 'var(--rule-strong)',
    'stroke-dasharray': '2 3' }, svg);

  const pts = values.map((v, i) => {
    const a = angle(i);
    return [cx + Math.cos(a) * scale(v), cy + Math.sin(a) * scale(v)];
  });
  el('path', {
    d: `${pts.map((p, i) => `${i ? 'L' : 'M'}${p[0].toFixed(1)},${p[1].toFixed(1)}`).join('')}Z`,
    fill: color, 'fill-opacity': 0.18, stroke: color, 'stroke-width': 1.8, 'stroke-linejoin': 'round',
  }, svg);

  const names = ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'];
  values.forEach((v, i) => {
    const a = angle(i);
    const lr = rMax + 11;
    const t = el('text', {
      x: cx + Math.cos(a) * lr, y: cy + Math.sin(a) * lr + 3,
      class: 'event-flag', 'text-anchor': 'middle',
      fill: v === Math.max(...values) ? color : 'var(--ink-faint)',
    }, svg);
    t.textContent = names[i];
    el('title', {}, t).textContent = `${(v * 100 - 100).toFixed(0)}% vs typical month`;
  });
  return svg;
}

/**
 * Horizontal burst timeline - one bar per term, sorted by onset.
 *
 * A line chart shows one term well and ten terms badly. This shows sixty: each
 * bar carries a start date, an end date and a magnitude, so a field's whole
 * history reads at a glance.
 *
 * @param {HTMLElement} host Container.
 * @param {Array} rows [{term,start,end,level,weight}]
 * @param {string[]} periods Shared period axis.
 * @param {function} onPick Called with a term when a row is clicked.
 */
export function burstTimeline(host, rows, periods, onPick) {
  host.textContent = '';
  const width = Math.max(320, host.clientWidth || 800);
  const isNarrow = width < 620;
  const labelW = isNarrow ? 96 : 150;
  const rowH = 19;
  const m = { top: 8, right: isNarrow ? 12 : 76, bottom: 24, left: labelW };
  const height = m.top + rows.length * rowH + m.bottom;

  const svg = el('svg', { viewBox: `0 0 ${width} ${height}`, role: 'img',
    'aria-label': 'Detected burst intervals by term' }, host);
  const iw = width - m.left - m.right;
  const x = (i) => m.left + (i / Math.max(1, periods.length - 1)) * iw;

  for (const { index, label } of yearTicks(periods, iw)) {
    el('line', { x1: x(index), x2: x(index), y1: m.top, y2: m.top + rows.length * rowH,
      class: 'grid-line' }, svg);
    el('text', { x: x(index), y: height - 8, class: 'axis-label', 'text-anchor': 'middle' }, svg)
      .textContent = label;
  }

  rows.forEach((row, i) => {
    const y = m.top + i * rowH;
    const a = periods.indexOf(row.start);
    const b = periods.indexOf(row.end);
    if (a < 0 || b < 0) return;

    const group = el('g', { style: 'cursor:pointer' }, svg);
    el('rect', { x: 0, y, width, height: rowH, fill: 'transparent' }, group);
    const label = el('text', {
      x: m.left - 8, y: y + rowH / 2 + 4, class: 'axis-label',
      'text-anchor': 'end', fill: 'var(--ink)',
    }, group);
    label.textContent = row.term.length > (isNarrow ? 13 : 20)
      ? `${row.term.slice(0, isNarrow ? 12 : 19)}…` : row.term;

    // Thickness encodes burst level, so intensity reads without a colour ramp.
    const thickness = 3 + row.level * 2.4;
    el('rect', {
      x: x(a), y: y + (rowH - thickness) / 2, rx: thickness / 2,
      width: Math.max(2.5, x(b) - x(a)), height: thickness,
      fill: seriesColor(row.level - 1),
    }, group);

    if (!isNarrow && row.field) {
      el('text', {
        x: x(b) + 7, y: y + rowH / 2 + 3.5, class: 'event-flag',
        fill: 'var(--ink-faint)',
      }, group).textContent = row.field;
    }

    const title = el('title', {}, group);
    title.textContent =
      `${row.term}: level ${row.level} burst, ${row.start} to ${row.end}` +
      (row.field ? ` — ${Math.round(row.field_share * 100)}% ${row.field}` : '');
    if (onPick) group.onclick = () => onPick(row.term);
  });
  return svg;
}

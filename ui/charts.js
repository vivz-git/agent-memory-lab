/**
 * charts.js — Lightweight canvas chart renderer for Agent Memory Lab.
 * No external charting dependencies. Uses vanilla Canvas 2D API.
 *
 * Exports:
 *   window.Charts.drawLineChart(canvasId, series, opts)
 *   window.Charts.drawTimeline(canvasId, events, opts)
 */

'use strict';

window.Charts = (() => {

  /* ── Theme-aware colour helpers ─────────────────── */
  function getCssVar(name) {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  }

  const DECISION_COLORS = {
    add:    () => getCssVar('--col-add'),
    retain: () => getCssVar('--col-retain'),
    delete: () => getCssVar('--col-delete'),
    reject: () => getCssVar('--col-reject'),
  };

  const SERIES_COLORS = [
    '#b08050', // accent copper
    '#c0392b', // research red
    '#1e5fa8', // academic blue
    '#2e7d5a', // forest green
  ];

  /* ── DPR-aware canvas setup ─────────────────────── */
  function setupCanvas(canvas) {
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    const w = rect.width  || canvas.clientWidth  || 600;
    const h = rect.height || canvas.clientHeight || 200;
    canvas.width  = w * dpr;
    canvas.height = h * dpr;
    const ctx = canvas.getContext('2d');
    ctx.scale(dpr, dpr);
    return { ctx, w, h };
  }

  /* ── Line chart ─────────────────────────────────── */
  function drawLineChart(canvasId, series, opts = {}) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;

    const { ctx, w, h } = setupCanvas(canvas);

    const pad = { top: 16, right: 16, bottom: 32, left: 44 };
    const innerW = w - pad.left - pad.right;
    const innerH = h - pad.top - pad.bottom;

    const steps  = opts.steps || series[0]?.data.map((_, i) => i + 1) || [];
    const minY   = opts.minY ?? 0;
    const maxY   = opts.maxY ?? 1;
    const yTicks = opts.yTicks ?? 5;

    const textColor   = getCssVar('--text-muted');
    const borderColor = getCssVar('--border');
    const bgColor     = getCssVar('--bg-card');

    // Background
    ctx.fillStyle = bgColor;
    ctx.fillRect(0, 0, w, h);

    // Grid lines
    ctx.strokeStyle = borderColor;
    ctx.lineWidth = 1;
    for (let i = 0; i <= yTicks; i++) {
      const y = pad.top + innerH - (i / yTicks) * innerH;
      ctx.beginPath();
      ctx.moveTo(pad.left, y);
      ctx.lineTo(pad.left + innerW, y);
      ctx.stroke();

      // Y labels
      ctx.fillStyle = textColor;
      ctx.font = `11px ${getCssVar('--font-mono') || 'monospace'}`;
      ctx.textAlign = 'right';
      const val = minY + (i / yTicks) * (maxY - minY);
      ctx.fillText(val.toFixed(opts.yFixed ?? 2), pad.left - 6, y + 4);
    }

    // X axis labels
    const xLabelInterval = Math.ceil(steps.length / 6);
    ctx.fillStyle = textColor;
    ctx.font = `11px ${getCssVar('--font-mono') || 'monospace'}`;
    ctx.textAlign = 'center';
    steps.forEach((s, i) => {
      if (i % xLabelInterval === 0 || i === steps.length - 1) {
        const x = pad.left + (i / (steps.length - 1)) * innerW;
        ctx.fillText(s, x, h - 8);
      }
    });

    // X axis label text
    if (opts.xLabel) {
      ctx.fillStyle = textColor;
      ctx.font = `10px ${getCssVar('--font-ui') || 'sans-serif'}`;
      ctx.textAlign = 'center';
      ctx.fillText(opts.xLabel, pad.left + innerW / 2, h - 2);
    }

    // Series lines
    series.forEach((s, si) => {
      const color = s.color || SERIES_COLORS[si % SERIES_COLORS.length];
      ctx.strokeStyle = color;
      ctx.lineWidth = s.lineWidth ?? 2;
      ctx.lineJoin = 'round';
      ctx.lineCap  = 'round';

      if (s.dashed) {
        ctx.setLineDash([6, 4]);
      } else {
        ctx.setLineDash([]);
      }

      ctx.beginPath();
      s.data.forEach((v, i) => {
        const x = pad.left + (i / (s.data.length - 1)) * innerW;
        const y = pad.top + innerH - ((v - minY) / (maxY - minY)) * innerH;
        if (i === 0) ctx.moveTo(x, y);
        else         ctx.lineTo(x, y);
      });
      ctx.stroke();

      // Area fill (subtle)
      if (s.fill) {
        ctx.setLineDash([]);
        ctx.beginPath();
        s.data.forEach((v, i) => {
          const x = pad.left + (i / (s.data.length - 1)) * innerW;
          const y = pad.top + innerH - ((v - minY) / (maxY - minY)) * innerH;
          if (i === 0) ctx.moveTo(x, y);
          else         ctx.lineTo(x, y);
        });
        ctx.lineTo(pad.left + innerW, pad.top + innerH);
        ctx.lineTo(pad.left, pad.top + innerH);
        ctx.closePath();
        ctx.fillStyle = color + '18'; // ~10% opacity
        ctx.fill();
      }

      ctx.setLineDash([]);
    });
  }

  /* ── Timeline bar ───────────────────────────────── */
  function drawTimeline(canvasId, events, opts = {}) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;

    const { ctx, w, h } = setupCanvas(canvas);

    const pad  = { top: 16, right: 12, bottom: 28, left: 48 };
    const innerW = w - pad.left - pad.right;
    const innerH = h - pad.top  - pad.bottom;

    const bgColor   = getCssVar('--bg-card');
    const textColor = getCssVar('--text-muted');
    const borderColor = getCssVar('--border');

    ctx.fillStyle = bgColor;
    ctx.fillRect(0, 0, w, h);

    if (!events || !events.length) return;

    const barW = Math.max(6, Math.floor(innerW / events.length) - 3);

    events.forEach((ev, i) => {
      const x = pad.left + (i / events.length) * innerW + 2;
      const color = DECISION_COLORS[ev.decision]?.() || getCssVar('--border-strong');

      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.roundRect(x, pad.top + 4, barW, innerH - 4, 3);
      ctx.fill();
    });

    // Step labels at start / end
    ctx.fillStyle = textColor;
    ctx.font = `10px ${getCssVar('--font-mono') || 'monospace'}`;
    ctx.textAlign = 'left';
    ctx.fillText(`t=${events[0].step}`, pad.left, h - 8);
    ctx.textAlign = 'right';
    ctx.fillText(`t=${events[events.length - 1].step}`, pad.left + innerW, h - 8);

    // Y label
    ctx.save();
    ctx.translate(14, pad.top + innerH / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillStyle = textColor;
    ctx.font = `10px ${getCssVar('--font-ui') || 'sans-serif'}`;
    ctx.textAlign = 'center';
    ctx.fillText('Events', 0, 0);
    ctx.restore();
  }

  /* ── Render legend DOM ──────────────────────────── */
  function renderLegend(containerId, series) {
    const el = document.getElementById(containerId);
    if (!el) return;
    el.innerHTML = '';
    series.forEach((s, si) => {
      const color = s.color || SERIES_COLORS[si % SERIES_COLORS.length];
      const entry = document.createElement('span');
      entry.className = 'legend-entry';
      entry.setAttribute('role', 'listitem');
      entry.innerHTML = `<span class="legend-line" style="background:${color}${s.dashed ? ';background:repeating-linear-gradient(90deg,'+color+' 0,'+color+' 4px,transparent 4px,transparent 8px)' : ''}"></span>${s.label}`;
      el.appendChild(entry);
    });
  }

  /* ── Public API ─────────────────────────────────── */
  return { drawLineChart, drawTimeline, renderLegend };
})();

/**
 * app.js — Agent Memory Lab UI Application Logic
 *
 * Handles:
 *   - Tab navigation
 *   - Theme toggle
 *   - Demo preset loading (ADD / RETAIN / DELETE / REJECT / Error states)
 *   - Run button (simulated async execution with loading overlay)
 *   - Memory bank table rendering + filtering
 *   - Timeline and chart rendering
 *   - Adaptive read rejection toggle
 *   - Toast notifications
 *   - Keyboard navigation
 *   - Reduced motion support
 *
 * Backend integration:
 *   Replace `simulateEpisode()` with fetch('/api/run_episode', {...})
 *   Replace `loadMemoryBank()` with fetch('/api/memory_bank')
 *   Replace `loadMetrics()` with fetch('/api/metrics')
 */

'use strict';

/* ── State ──────────────────────────────────────── */
const state = {
  activePanel: 'run',
  adaptiveEnabled: true,
  lastResult: null,
  theme: localStorage.getItem('aml-theme') || 'light',
};

/* ── DOM helpers ────────────────────────────────── */
const $ = (id) => document.getElementById(id);
const $$ = (sel) => document.querySelectorAll(sel);

function show(el) { if (el) { el.hidden = false; el.removeAttribute('hidden'); } }
function hide(el) { if (el) { el.hidden = true; } }

/* ── Theme ──────────────────────────────────────── */
function applyTheme(t) {
  state.theme = t;
  document.documentElement.setAttribute('data-theme', t);
  $('theme-icon').textContent = t === 'dark' ? '\u25D0' : '\u25D1';
  localStorage.setItem('aml-theme', t);
}
applyTheme(state.theme);
$('theme-toggle').addEventListener('click', () => applyTheme(state.theme === 'light' ? 'dark' : 'light'));

/* ── Tab Navigation ─────────────────────────────── */
function switchPanel(name) {
  $$('.nav-tab').forEach(btn => {
    const isActive = btn.dataset.panel === name;
    btn.classList.toggle('active', isActive);
    btn.setAttribute('aria-selected', isActive ? 'true' : 'false');
  });
  $$('.panel').forEach(p => {
    const isActive = p.id === `panel-${name}`;
    p.classList.toggle('active', isActive);
    isActive ? p.removeAttribute('hidden') : p.setAttribute('hidden', '');
  });
  state.activePanel = name;

  // Lazy-render charts/memory when switching to those panels
  if (name === 'metrics') renderMetrics();
  if (name === 'memory')  renderMemory();
}

$$('.nav-tab').forEach(btn => {
  btn.addEventListener('click', () => switchPanel(btn.dataset.panel));
  btn.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); switchPanel(btn.dataset.panel); }
  });
});

/* ── Adaptive toggle ────────────────────────────── */
const adaptiveBtn = $('adaptive-toggle');
adaptiveBtn.addEventListener('click', () => {
  state.adaptiveEnabled = !state.adaptiveEnabled;
  adaptiveBtn.setAttribute('aria-checked', state.adaptiveEnabled ? 'true' : 'false');
  toast(state.adaptiveEnabled ? 'Adaptive Read Rejection enabled' : 'Adaptive Read Rejection disabled');
});

/* ── Toast ──────────────────────────────────────── */
let toastTimer;
function toast(msg, type = '') {
  const el = $('toast');
  el.textContent = msg;
  el.className = 'toast show' + (type ? ' ' + type : '');
  show(el);
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => {
    el.classList.remove('show');
    setTimeout(() => hide(el), 300);
  }, 3000);
}

/* ── Loading overlay ────────────────────────────── */
function setLoading(on) {
  const overlay = $('loading-overlay');
  const runBtn  = $('run-btn');
  if (on) {
    show(overlay);
    overlay.removeAttribute('aria-hidden');
    runBtn.setAttribute('aria-busy', 'true');
    show(runBtn.querySelector('.btn-spinner'));
  } else {
    hide(overlay);
    overlay.setAttribute('aria-hidden', 'true');
    runBtn.setAttribute('aria-busy', 'false');
    hide(runBtn.querySelector('.btn-spinner'));
  }
}

/* ── Simulate backend episode call ─────────────── */
function simulateEpisode() {
  /* BACKEND INTEGRATION POINT:
     Replace this with:
       const resp = await fetch('/api/run_episode', {
         method: 'POST',
         headers: { 'Content-Type': 'application/json' },
         body: JSON.stringify({
           query: $('query-input').value,
           environment: $('env-select').value,
           addition_policy: $('add-policy').value,
           deletion_policy: $('del-policy').value,
           adaptive_read_rejection: state.adaptiveEnabled,
         }),
       });
       return resp.json();
  */
  return new Promise((resolve, reject) => {
    const delay = 900 + Math.random() * 600;
    setTimeout(() => {
      const addPolicy = $('add-policy').value;
      const delPolicy = $('del-policy').value;
      const adaptive  = state.adaptiveEnabled;

      // Pick a demo state that loosely matches the selected policies
      let key;
      if (addPolicy === 'fixed') key = 'retain';
      else if (addPolicy === 'add_all') key = 'delete';
      else if (adaptive && delPolicy === 'history') key = 'reject';
      else if (addPolicy === 'strict') key = 'add';
      else key = 'retain';

      resolve(window.DEMO_STATES[key]);
    }, delay);
  });
}

/* ── Render result from a state object ─────────── */
function renderState(s) {
  if (s._isError) {
    toast(s.errorMessage, 'error');
    return;
  }

  state.lastResult = s;

  // ── Result card
  const statusEl = $('result-status');
  statusEl.textContent = s.status === 'pass' ? 'PASS' : 'FAIL';
  statusEl.className = 'status-badge ' + (s.status === 'pass' ? 'pass' : 'fail');

  $('r-prediction').textContent  = s.prediction;
  $('r-ground-truth').textContent = s.groundTruth;
  $('r-score').textContent        = s.score.toFixed(3);
  $('r-error').textContent        = s.error.toFixed(3);
  $('r-trajectory').textContent   = s.trajectory;

  hide($('result-empty'));
  show($('result-data'));

  // ── Retrieved experiences
  const listEl = $('retrieved-list');
  listEl.innerHTML = '';

  $('retrieved-count').textContent = `${s.retrievedExperiences.length} records`;

  s.retrievedExperiences.forEach(exp => {
    const item = document.createElement('div');
    item.className = `mem-record ${exp.filtered ? 'rejected' : 'accepted'}`;
    item.setAttribute('role', 'listitem');
    item.innerHTML = `
      <span class="mem-rank mono">#${exp.rank}</span>
      <span class="mem-text" title="${escHtml(exp.text)}">${escHtml(exp.text)}</span>
      <span class="mem-score">sim=${exp.score.toFixed(2)}</span>
      <span class="mem-filter-tag">${exp.filtered ? 'Rejected' : 'Accepted'}</span>
    `;
    listEl.appendChild(item);
  });

  // ── Decision card
  hide($('decision-empty'));
  show($('decision-data'));

  const addBadge = $('gate-add-badge');
  const label = s.addDecision.label;
  addBadge.textContent = label;
  addBadge.className = 'gate-badge ' + ({
    ADD: 'add', RETAIN: 'retain', NONE: 'none',
  }[label] || 'none');
  $('gate-add-detail').textContent = s.addDecision.reason;

  const delBadge = $('gate-del-badge');
  const dLabel = s.delDecision.label;
  delBadge.textContent = dLabel;
  delBadge.className = 'gate-badge ' + ({
    DELETE: 'delete', NONE: 'none',
  }[dLabel] || 'none');
  $('gate-del-detail').textContent = s.delDecision.reason;

  // Utility bar
  const pct = Math.round(s.utilityScore * 100);
  const fillEl = $('utility-fill');
  fillEl.style.width = '0%';
  fillEl.parentElement.setAttribute('aria-valuenow', s.utilityScore);

  const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  if (prefersReducedMotion) {
    fillEl.style.width = `${pct}%`;
  } else {
    requestAnimationFrame(() => {
      fillEl.style.width = `${pct}%`;
    });
  }
  $('utility-value').textContent    = s.utilityScore.toFixed(3);
  $('u-retrieval-count').textContent = s.retrievalCount;
  $('u-mean-utility').textContent    = s.meanUtility.toFixed(3);
  $('u-entry-step').textContent      = s.entryStep;

  // Adaptive rejection banner
  if (s.adaptiveRejection && state.adaptiveEnabled) {
    $('rejection-detail').textContent = s.rejectionDetail || 'Record rejected before entering LLM prompt context.';
    show($('rejection-section'));
  } else {
    hide($('rejection-section'));
  }
}

/* ── Run button ─────────────────────────────────── */
$('run-btn').addEventListener('click', async () => {
  const query = $('query-input').value.trim();
  if (!query) {
    toast('Enter a query vector to execute', 'error');
    $('query-input').focus();
    return;
  }

  setLoading(true);
  try {
    const result = await simulateEpisode();
    renderState(result);
    toast(result._isError ? 'Error' : 'Episode complete', result._isError ? 'error' : 'success');
  } catch (err) {
    toast('Unexpected error — check console', 'error');
    console.error('[AML]', err);
  } finally {
    setLoading(false);
  }
});

/* ── Preset buttons ─────────────────────────────── */
$$('.preset-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    const preset = btn.dataset.preset;
    const s = window.DEMO_STATES[preset];
    if (!s) return;

    // Seed query field with matching vector for that state
    const vectors = {
      add:    '[0.42, -1.17, 0.88, 0.33, -0.61, 1.04]',
      retain: '[0.21, -0.88, 1.12, 0.44, -0.33, 0.77]',
      delete: '[0.60, -0.45, 0.72, -1.22, 0.38, -0.91]',
      reject: '[0.31, -1.05, 0.78, 0.22, -0.55, 0.99]',
      error:  '[0.00, 0.00, 0.00, 0.00, 0.00, 0.00]',
    };
    $('query-input').value = vectors[preset] || '';

    if (s._isError) {
      toast(s.errorMessage, 'error');
    } else {
      renderState(s);
      toast(`Loaded demo: ${preset.toUpperCase()} state`, 'success');
    }
  });
});

/* ── Memory panel ───────────────────────────────── */
let currentFilter = 'all';

function renderMemory() {
  renderTimeline();
  renderMemoryTable(currentFilter);
}

function renderTimeline() {
  const canvas = $('timeline-canvas');
  if (!canvas) return;

  // Make canvas responsive
  const wrap = canvas.parentElement;
  canvas.style.width  = '100%';
  canvas.style.height = '80px';

  Charts.drawTimeline('timeline-canvas', window.TIMELINE_MOCK);
}

function renderMemoryTable(filter) {
  const tbody = $('memory-table-body');
  if (!tbody) return;
  tbody.innerHTML = '';

  let records = window.MEMORY_BANK_MOCK;
  if (filter === 'high-utility') records = records.filter(r => r.meanUtility > 0.7);
  if (filter === 'low-utility')  records = records.filter(r => r.meanUtility < 0.3);
  if (filter === 'at-risk')      records = records.filter(r => r.status === 'at-risk');

  if (!records.length) {
    tbody.innerHTML = '<tr><td colspan="6" class="empty-state" style="text-align:center;padding:2rem;color:var(--text-muted);">No records match this filter.</td></tr>';
    return;
  }

  records.forEach(r => {
    const utilityClass = r.meanUtility > 0.7 ? 'high' : r.meanUtility < 0.3 ? 'low' : 'medium';
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td class="mono" style="color:var(--text-muted);font-size:0.75rem">${escHtml(r.id)}</td>
      <td class="mono">${r.entryStep}</td>
      <td class="mono">${r.retrievalCount}</td>
      <td><span class="utility-chip ${utilityClass}">${r.meanUtility.toFixed(2)}</span></td>
      <td><span class="status-chip ${r.status}">${escHtml(r.status)}</span></td>
      <td class="mono" style="color:var(--text-secondary);max-width:160px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title="${escHtml(r.queryTruncated)}">${escHtml(r.queryTruncated)}</td>
    `;
    tbody.appendChild(tr);
  });

  // Update bank-size KPI
  const bankSizeEl = $('bank-size');
  if (bankSizeEl) bankSizeEl.textContent = window.MEMORY_BANK_MOCK.length;
}

$('bank-filter').addEventListener('change', (e) => {
  currentFilter = e.target.value;
  renderMemoryTable(currentFilter);
});

/* ── Metrics panel ──────────────────────────────── */
let metricsRendered = false;

function renderMetrics() {
  if (metricsRendered) return;
  metricsRendered = true;

  const d = window.CHART_DATA;

  const srSeries = [
    { label: 'Fixed (No-Memory)',              data: d.sr.fixed,      color: '#9a9390' },
    { label: 'Add-All',                        data: d.sr.addAll,     color: '#c0392b' },
    { label: 'Strict + History Del.',          data: d.sr.strict,     color: '#1e5fa8' },
    { label: 'Strict + History + Read Reject', data: d.sr.strictRead, color: '#2e7d5a' },
  ];
  Charts.drawLineChart('sr-chart', srSeries, { steps: d.steps, minY: 0.3, maxY: 1.0, xLabel: 'Stream Step' });
  Charts.renderLegend('sr-legend', srSeries);

  const memSeries = [
    { label: 'Fixed',                          data: d.mem.fixed,      color: '#9a9390' },
    { label: 'Add-All (runaway growth)',        data: d.mem.addAll,     color: '#c0392b', fill: true },
    { label: 'Strict + History Del.',          data: d.mem.strict,     color: '#1e5fa8' },
    { label: 'Strict + Read Reject',           data: d.mem.strictRead, color: '#2e7d5a' },
  ];
  Charts.drawLineChart('mem-chart', memSeries, { steps: d.steps, minY: 0, maxY: 650, yFixed: 0, xLabel: 'Stream Step' });
  Charts.renderLegend('mem-legend', memSeries);

  const epSeries = [
    { label: 'History Deletion only',                      data: d.ep.hist,     color: '#1e5fa8', fill: true },
    { label: 'History Deletion + Adaptive Read Rejection', data: d.ep.histRead, color: '#2e7d5a', dashed: true },
  ];
  Charts.drawLineChart('ep-chart', epSeries, { steps: d.steps, minY: 0, maxY: 0.5, xLabel: 'Stream Step', yFixed: 2 });
}

/* ── Resize: re-draw charts ─────────────────────── */
let resizeTimer;
window.addEventListener('resize', () => {
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(() => {
    if (state.activePanel === 'metrics') {
      metricsRendered = false;
      renderMetrics();
    }
    if (state.activePanel === 'memory') {
      renderTimeline();
    }
  }, 200);
});

/* ── Theme change re-render ─────────────────────── */
$('theme-toggle').addEventListener('click', () => {
  setTimeout(() => {
    if (state.activePanel === 'metrics') { metricsRendered = false; renderMetrics(); }
    if (state.activePanel === 'memory')  renderTimeline();
  }, 50);
});

/* ── XSS-safe HTML escape ───────────────────────── */
function escHtml(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

/* ── Keyboard: Escape closes loading ────────────── */
document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape') {
    const overlay = $('loading-overlay');
    if (!overlay.hidden) setLoading(false);
  }
});

/* ── Init: load preset on first render ──────────── */
(function init() {
  // Seed the Run panel with the ADD demo state by default (for immediate visual richness)
  renderState(window.DEMO_STATES.add);
  $('query-input').value = '[0.42, -1.17, 0.88, 0.33, -0.61, 1.04]';
  toast('Demo loaded — explore ADD, RETAIN, DELETE, REJECT states', '');
})();

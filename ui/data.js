/**
 * data.js — Mock / demo state data for the Agent Memory Lab UI.
 * Structured to mirror the real backend interfaces in src/memory/schema.py
 * and src/evaluation/runner.py so integration is a drop-in replacement.
 *
 * Backend integration point:
 *   Replace DEMO_STATES and MEMORY_BANK_MOCK with real API calls to:
 *     POST /api/run_episode      → returns StepResult-shaped JSON
 *     GET  /api/memory_bank      → returns ExperienceRecord[] JSON
 *     GET  /api/metrics          → returns ExperimentResult JSON
 */

'use strict';

/* ────────────────────────────────────────────────
   DECISION STATES — one per memory operation
   Maps to: AdditionPolicyType + DeletionPolicyType
   ──────────────────────────────────────────────── */
window.DEMO_STATES = {

  add: {
    status: 'pass',
    prediction: '3.842',
    groundTruth: '3.801',
    score: 0.96,
    error: 0.041,
    trajectory: 'Step 1: Retrieved 3 similar experiences.\nStep 2: Applied linear combination w=[0.52, -0.31, 0.88, 0.14, -0.60, 1.02]\nStep 3: Predicted y = w^T * x + ε = 3.842\nStep 4: Evaluator: PASS (|3.842 - 3.801| = 0.041 < threshold 1.0)',
    retrievedExperiences: [
      { rank: 1, id: 'exp-0041', score: 0.97, text: '[0.41, -1.14, 0.85, 0.31, -0.60, 1.03]', utility: 0.94, filtered: false },
      { rank: 2, id: 'exp-0029', score: 0.91, text: '[0.38, -1.22, 0.90, 0.29, -0.65, 1.01]', utility: 0.88, filtered: false },
      { rank: 3, id: 'exp-0072', score: 0.84, text: '[0.45, -1.10, 0.82, 0.36, -0.57, 1.06]', utility: 0.79, filtered: false },
    ],
    addDecision:  { label: 'ADD',    reason: 'Strict oracle: ground-truth match (|err| < 1.0). π=1 → experience written to memory bank.' },
    delDecision:  { label: 'NONE',   reason: 'History-based policy: retrieval count < n_min (5). No eviction triggered.' },
    utilityScore: 0.96,
    retrievalCount: 1,
    meanUtility: 0.96,
    entryStep: 247,
    adaptiveRejection: false,
  },

  retain: {
    status: 'pass',
    prediction: '2.154',
    groundTruth: '2.198',
    score: 0.88,
    error: 0.044,
    trajectory: 'Step 1: Retrieved 3 experiences.\nStep 2: Applied weighted similarity to compute prediction.\nStep 3: Predicted y = 2.154\nStep 4: Evaluator: PASS (|2.154 - 2.198| = 0.044 < threshold 1.0)',
    retrievedExperiences: [
      { rank: 1, id: 'exp-0018', score: 0.93, text: '[0.21, -0.88, 1.12, 0.44, -0.33, 0.77]', utility: 0.85, filtered: false },
      { rank: 2, id: 'exp-0055', score: 0.87, text: '[0.18, -0.92, 1.08, 0.48, -0.29, 0.82]', utility: 0.90, filtered: false },
      { rank: 3, id: 'exp-0091', score: 0.78, text: '[0.25, -0.84, 1.15, 0.41, -0.37, 0.74]', utility: 0.72, filtered: false },
    ],
    addDecision:  { label: 'RETAIN', reason: 'Coarse LLM judge: trajectory is a minor duplicate. π=0 → not added. Existing record retained.' },
    delDecision:  { label: 'NONE',   reason: 'Periodic policy: retrieved 4× in last T=50 steps (≥ α=2). Record is active — no eviction.' },
    utilityScore: 0.88,
    retrievalCount: 8,
    meanUtility: 0.85,
    entryStep: 103,
    adaptiveRejection: false,
  },

  delete: {
    status: 'fail',
    prediction: '-0.712',
    groundTruth: '1.934',
    score: 0.0,
    error: 2.646,
    trajectory: 'Step 1: Retrieved 3 experiences (1 is stale / misaligned cluster).\nStep 2: Retrieved exp-0007 from Cluster-A (now in Cluster-C stream). High input-similarity but wrong output regime.\nStep 3: Predicted y = -0.712  ← misaligned replay\nStep 4: Evaluator: FAIL (|−0.712 − 1.934| = 2.646 > threshold 1.0)',
    retrievedExperiences: [
      { rank: 1, id: 'exp-0007', score: 0.89, text: '[0.60, -0.45, 0.72, -1.22, 0.38, -0.91]', utility: 0.21, filtered: false },
      { rank: 2, id: 'exp-0031', score: 0.80, text: '[0.55, -0.48, 0.68, -1.18, 0.41, -0.87]', utility: 0.34, filtered: false },
      { rank: 3, id: 'exp-0044', score: 0.71, text: '[0.63, -0.42, 0.75, -1.25, 0.35, -0.94]', utility: 0.18, filtered: false },
    ],
    addDecision:  { label: 'NONE',   reason: 'Strict oracle: task failed (score=0.0). π=0 → experience not written to memory bank.' },
    delDecision:  { label: 'DELETE', reason: 'History-based policy φ_hist: exp-0007 has retrieval_count=12 ≥ n=5 and mean_utility=0.21 ≤ β=0.30. Eviction triggered.' },
    utilityScore: 0.0,
    retrievalCount: 12,
    meanUtility: 0.21,
    entryStep: 11,
    adaptiveRejection: false,
  },

  reject: {
    status: 'pass',
    prediction: '1.671',
    groundTruth: '1.704',
    score: 0.91,
    error: 0.033,
    trajectory: 'Step 1: Retrieved 4 candidates (oversampled ×4 for Read filter backoff).\nStep 2: AdaptiveReadFilter: exp-0009 mean_utility=0.19 < threshold 0.42 → REJECTED from prompt.\nStep 3: Backed off to exp-0061 (rank 4, utility=0.88).\nStep 4: Predicted y = 1.671 using clean context.\nStep 5: Evaluator: PASS (|1.671 − 1.704| = 0.033 < threshold 1.0)',
    retrievedExperiences: [
      { rank: 1, id: 'exp-0009', score: 0.92, text: '[0.31, -1.05, 0.78, 0.22, -0.55, 0.99]', utility: 0.19, filtered: true  },
      { rank: 2, id: 'exp-0024', score: 0.88, text: '[0.33, -1.01, 0.81, 0.20, -0.58, 0.97]', utility: 0.88, filtered: false },
      { rank: 3, id: 'exp-0047', score: 0.81, text: '[0.28, -1.08, 0.76, 0.25, -0.52, 1.01]', utility: 0.82, filtered: false },
      { rank: 4, id: 'exp-0061', score: 0.75, text: '[0.36, -0.98, 0.83, 0.18, -0.61, 0.95]', utility: 0.88, filtered: false },
    ],
    addDecision:  { label: 'ADD',    reason: 'Strict oracle: PASS. π=1 → experience added to memory bank.' },
    delDecision:  { label: 'NONE',   reason: 'exp-0009 has utility 0.19 — below history threshold but retrieval_count < n. Read Rejection prevented pollution before eviction.' },
    utilityScore: 0.91,
    retrievalCount: 6,
    meanUtility: 0.55,
    entryStep: 44,
    adaptiveRejection: true,
    rejectionDetail: 'exp-0009 (rank 1, score=0.92) has mean utility 0.19 below adaptive threshold 0.42. Rejected from prompt — backed off to rank 4 (exp-0061, utility=0.88).',
  },

  error: {
    _isError: true,
    errorMessage: 'API error: Connection to backend refused. Running in offline demo mode.',
  },
};

/* ────────────────────────────────────────────────
   MEMORY BANK MOCK
   Maps to: ExperienceRecord[] from src/memory/schema.py
   ──────────────────────────────────────────────── */
window.MEMORY_BANK_MOCK = (() => {
  const now = 247;
  const records = [];
  const statuses = ['retained', 'retained', 'retained', 'at-risk', 'new'];
  const queries = [
    '[0.42, -1.17, 0.88]', '[0.21, -0.88, 1.12]', '[0.60, -0.45, 0.72]',
    '[0.31, -1.05, 0.78]', '[-0.14, 0.93, -0.67]', '[0.78, 0.12, -0.55]',
    '[0.55, -0.33, 1.01]', '[-0.22, 1.14, 0.43]', '[0.89, -0.71, 0.29]',
    '[0.04, -0.56, 1.33]',
  ];
  for (let i = 0; i < 20; i++) {
    const utility = Math.round((0.15 + Math.random() * 0.85) * 100) / 100;
    const retrievalCount = Math.floor(Math.random() * 18);
    const entryStep = Math.floor(Math.random() * now);
    const status = utility < 0.3 ? 'at-risk' : (entryStep > now - 20 ? 'new' : 'retained');
    records.push({
      id: `exp-${String(i).padStart(4, '0')}`,
      entryStep,
      retrievalCount,
      meanUtility: utility,
      status,
      queryTruncated: queries[i % queries.length],
    });
  }
  return records.sort((a, b) => b.meanUtility - a.meanUtility);
})();

/* ────────────────────────────────────────────────
   TIMELINE MOCK
   30 steps of ADD/RETAIN/DELETE/REJECT events
   ──────────────────────────────────────────────── */
window.TIMELINE_MOCK = (() => {
  const decisions = ['add', 'retain', 'retain', 'retain', 'delete', 'reject', 'retain', 'add'];
  return Array.from({ length: 30 }, (_, i) => ({
    step: 218 + i,
    decision: decisions[(i * 3 + i) % decisions.length],
  }));
})();

/* ────────────────────────────────────────────────
   CHART DATA — Series for Metrics panel
   Maps to: ExperimentResult from src/evaluation/runner.py
   ──────────────────────────────────────────────── */
window.CHART_DATA = (() => {
  const T = 50; // Display T=500 steps, sampled at interval of 10 → 50 points
  const steps = Array.from({ length: T }, (_, i) => (i + 1) * 10);

  function smoothSeries(base, noise) {
    const out = [];
    for (let i = 0; i < T; i++) {
      let v = base + (Math.random() - 0.5) * noise;
      out.push(Math.round(v * 1000) / 1000);
    }
    return out;
  }

  // Task Success Rate (Plateaed around 53%)
  const srFixed      = smoothSeries(0.53, 0.01);
  const srAddAll     = smoothSeries(0.525, 0.01);
  const srStrict     = smoothSeries(0.525, 0.01);
  const srStrictRead = smoothSeries(0.525, 0.01);

  // Memory Size
  const memFixed      = Array(T).fill(20);
  const memAddAll     = Array.from({ length: T }, (_, i) => Math.round(20 + i * 2)); // 20 -> 120
  const memStrict     = Array.from({ length: T }, (_, i) => Math.min(20 + Math.round(i * 0.2), 30.5)); // 20 -> 30
  const memStrictRead = Array.from({ length: T }, (_, i) => Math.min(20 + Math.round(i * 0.6), 50.5)); // 20 -> 50

  // Error Propagation Gap Δ_EP (Flat at ~47.5%)
  const epHist = smoothSeries(0.475, 0.01);
  const epHistRead = smoothSeries(0.475, 0.01);

  return {
    steps,
    sr: { fixed: srFixed, addAll: srAddAll, strict: srStrict, strictRead: srStrictRead },
    mem: { fixed: memFixed, addAll: memAddAll, strict: memStrict, strictRead: memStrictRead },
    ep: { hist: epHist, histRead: epHistRead },
  };
})();

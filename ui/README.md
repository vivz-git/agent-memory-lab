# Agent Memory Lab — Demo UI

> Research demo interface for **How Memory Management Impacts LLM Agents: An Empirical Study of Experience-Following Behavior**

## Overview

A zero-dependency, single-page browser UI that demonstrates the agent memory lifecycle:

```
Query → Retrieval → Adaptive Read Filter → LLM Execution
     → Evaluator → Addition Gate π(q,e) → Memory Bank
                 → Deletion Gate φ(q,e,t) → Eviction
```

## Visual Identity

| Token         | Value |
|---------------|-------|
| Typefaces     | Fira Sans (UI) · Fira Code (data/mono) |
| Surface       | Paper-white `#f8f7f5` / warm-dark `#100f0d` |
| Accent        | Copper-amber `#b08050` |
| ADD colour    | Forest green `#2e7d5a` |
| RETAIN colour | Academic blue `#1e5fa8` |
| DELETE colour | Research red `#c0392b` |
| REJECT colour | Violet `#7b4f9e` |

Deliberately distinct from CorrectRAG (dark slate, green accent, IBM Plex Sans).

## Panels

### Run
- Task vector input, environment selector, Addition/Deletion policy selectors
- Adaptive Read Rejection toggle (Extension §13)
- Agent result: prediction, ground truth, score, L2 error, trajectory
- Retrieved experiences with ACCEPTED / REJECTED per-record tags
- Memory Decision: gate badges, utility score bar, meta stats
- Adaptive rejection banner when System-1 filter fires

### Memory
- Memory bank table (20 mock records, filterable)
- Timeline canvas: 30-step event bar chart (ADD/RETAIN/DELETE/REJECT)
- Status chips: retained / at-risk / new

### Metrics
- KPI strip: SR, r_EF, Memory Size, Δ_EP
- Success Rate over stream (4 baselines)
- Memory Size growth (Add-All runaway vs bounded)
- Error Propagation Gap: History Deletion vs + Read Rejection
- Baseline comparison table (5 strategies)

## Demo States

| Preset  | Scenario |
|---------|----------|
| ADD     | Strict oracle pass → experience written to bank |
| RETAIN  | Duplicate detected by Coarse judge → not re-added |
| DELETE  | Task failure + history-based eviction triggered |
| REJECT  | Adaptive Read Rejection fires, backs off to rank 4 |
| Error   | Backend connection error |

## Usage

```bash
# Just open the file in any browser — no build step required
start ui/index.html          # Windows
open  ui/index.html          # macOS
```

Or serve with Python for accurate MIME types:

```bash
python -m http.server 8000 --directory ui/
# then visit http://localhost:8000
```

## Backend Integration

Replace three mock functions in `ui/app.js`:

| Function            | Endpoint                         | Returns |
|---------------------|----------------------------------|---------|
| `simulateEpisode()` | `POST /api/run_episode`          | `StepResult` JSON |
| `loadMemoryBank()`  | `GET  /api/memory_bank`          | `ExperienceRecord[]` JSON |
| `loadMetrics()`     | `GET  /api/metrics`              | `ExperimentResult` JSON |

Mock data schema in `ui/data.js` mirrors `src/memory/schema.py` exactly.

## Files

```
ui/
  index.html   Main shell — 3 panels (Run, Memory, Metrics)
  style.css    Design system (tokens, components, responsive layout)
  data.js      Mock state data matching backend schemas
  charts.js    Vanilla Canvas 2D chart renderer (no deps)
  app.js       Application logic, event handling, rendering
  README.md    This file
```

## Accessibility

- Semantic HTML (landmark roles, headings, `role="tab"`)
- `aria-live` for dynamic result regions
- Keyboard navigation: Enter/Space for tabs and buttons
- Visible focus ring (`outline: 2px solid var(--border-focus)`)
- `@media (prefers-reduced-motion: reduce)` applied
- Safe XSS escaping for all dynamic content
- Responsive: 1440px → tablet → mobile (no horizontal overflow)

## What This Demonstrates

A recruiter opening this UI immediately sees:

1. **The agent remembers experiences** — retrieved records shown per episode
2. **It evaluates them** — utility score Φ, retrieval count, mean utility
3. **It selectively keeps (ADD/RETAIN), removes (DELETE), or rejects (REJECT) them**
4. **The research extension** — Adaptive Read Rejection dampens Δ_EP before eviction

## Constraints Respected

- ✅ No modifications to `src/memory/`, `src/agent/`, `src/evaluation/`, `src/environments/`, `src/security/`, `research/`, `RESEARCH_SPEC.md`
- ✅ No new Python dependencies
- ✅ No build toolchain (npm, webpack, vite)
- ✅ Zero external JS libraries
- ✅ Committed on isolated `feat/demo-ui` worktree branch


---
name: eval-harness
description: >-
  Formal evaluation framework for AI agent memory systems and benchmark evaluation.
  Measures retrieval precision, recall@k, MRR, NDCG, agent trajectory accuracy, latency profiling, and regression baselines.
  Use when designing, executing, or benchmarking agent memory architectures and evaluation experiments.
---

# Eval Harness & Benchmark Evaluation

Evaluation-Driven Development (EDD) framework for validating and benchmarking agent memory systems.

## When to Activate

- Setting up memory benchmark experiments (e.g. replicating ACL/NeurIPS paper evaluations)
- Computing quantitative retrieval metrics: Precision@k, Recall@k, Mean Reciprocal Rank (MRR), NDCG@k
- Running regression test suites against memory baseline datasets
- Measuring agent decision accuracy and trajectory correctness across multi-turn interactions
- Profiling memory retrieval latency and token footprint

---

## 1. Core Evaluation Metrics

```python
import numpy as np

def compute_precision_at_k(retrieved_ids: list[str], ground_truth_ids: set[str], k: int) -> float:
    """Compute Precision@K."""
    if k <= 0 or not retrieved_ids:
        return 0.0
    top_k = retrieved_ids[:k]
    hits = sum(1 for doc_id in top_k if doc_id in ground_truth_ids)
    return hits / k

def compute_recall_at_k(retrieved_ids: list[str], ground_truth_ids: set[str], k: int) -> float:
    """Compute Recall@K."""
    if not ground_truth_ids:
        return 0.0
    top_k = retrieved_ids[:k]
    hits = sum(1 for doc_id in top_k if doc_id in ground_truth_ids)
    return hits / len(ground_truth_ids)

def compute_mrr(retrieved_ids: list[str], ground_truth_ids: set[str]) -> float:
    """Compute Mean Reciprocal Rank (MRR)."""
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in ground_truth_ids:
            return 1.0 / rank
    return 0.0
```

---

## 2. Benchmark Runner Architecture

Structure evaluation runs to be fully reproducible with deterministic seeds and JSON report artifacts:

```python
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path

@dataclass
class EvalResult:
    dataset_name: str
    num_samples: int
    precision_at_1: float
    precision_at_5: float
    recall_at_5: float
    mrr: float
    avg_latency_ms: float

    def save_report(self, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2)
```

---

## 3. Continuous Evaluation Protocol

1. **Baseline Registration**: Run evaluations on the untouched baseline dataset and commit the metric snapshot in `benchmarks/results/baseline.json`.
2. **Experiment Isolation**: Implement new memory algorithms in dedicated branches / modules without mutating baseline evaluation sets.
3. **Comparative Evaluation**: Run `python scripts/run_benchmark.py --compare baseline.json` to verify non-regression before merging.

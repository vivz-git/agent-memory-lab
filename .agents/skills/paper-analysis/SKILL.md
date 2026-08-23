---
name: paper-analysis
description: >-
  Systematic methodology for reading, deconstructing, and implementing academic AI papers (ACL, EMNLP, NeurIPS, ICLR).
  Extracts core hypotheses, algorithmic formulations, memory architectures, benchmark setups, and translates mathematical methods into executable Python code.
  Use when analyzing papers (such as PDF files in paper/), designing baseline replications, or synthesizing academic literature.
---

# Academic Paper Analysis & Replication Framework

Methodology for analyzing NLP/AI research papers and replicating experimental architectures in code.

## When to Activate

- Analyzing research papers (e.g. PDF papers in `paper/`, ACL/NeurIPS publications)
- Deconstructing memory architectures, retrieval mechanisms, or agent reasoning loops
- Translating equations and algorithms from research text into Python implementations
- Designing ablation studies and benchmark replication pipelines
- Synthesizing related literature for background documentation and technical reports

---

## 1. 5-Step Paper Deconstruction Workflow

```text
1. Problem & Hypothesis
   ├── What fundamental bottleneck does the paper address?
   └── What is the core hypothesis and proposed contribution?

2. Architectural Deconstruction
   ├── Input representations & embeddings
   ├── Memory read/write/forget operators
   └── Agent decision / reasoning loop

3. Mathematical Formulation Translation
   ├── Identify key equations (scoring functions, loss, update rules)
   └── Translate formulas directly to vectorized NumPy/PyTorch/Python functions

4. Empirical Benchmark Protocol
   ├── Datasets used & evaluation splits
   ├── Baseline methods compared against
   └── Primary metrics (Accuracy, F1, MRR, Latency, Token cost)

5. Replication & Code Mapping
   ├── Map components to src/memory/ and src/agent/
   └── Implement unit tests against paper toy examples
```

---

## 2. Math-to-Code Translation Pattern

Always document the paper's original equation reference in docstrings alongside the code implementation:

```python
import numpy as np

def compute_recency_weighted_similarity(
    query_vector: np.ndarray,
    memory_vectors: np.ndarray,
    timestamps: np.ndarray,
    current_time: float,
    decay_factor: float = 0.99
) -> np.ndarray:
    """
    Implements Memory Relevance Scoring from Paper Section 3.2 (Eq. 4):
    
        Score(m, q) = cos(v_m, v_q) * exp(-lambda * (t_now - t_m))
        
    Args:
        query_vector: Shape (D,) representation of current query
        memory_vectors: Shape (N, D) matrix of stored memories
        timestamps: Shape (N,) array of memory creation timestamps
        current_time: Current timestamp (t_now)
        decay_factor: Temporal decay constant lambda > 0
        
    Returns:
        Shape (N,) array of final hybrid scores
    """
    # 1. Cosine similarity
    norm_q = np.linalg.norm(query_vector) + 1e-9
    norm_m = np.linalg.norm(memory_vectors, axis=1) + 1e-9
    cos_sim = np.dot(memory_vectors, query_vector) / (norm_m * norm_q)

    # 2. Exponential temporal decay
    time_deltas = current_time - timestamps
    decay = np.exp(-decay_factor * np.maximum(0, time_deltas))

    return cos_sim * decay
```

---

## 3. Structured Paper Analysis Report Template

When analyzing papers in `paper/` or from literature, generate markdown reports saved to `research/`:

```markdown
# Paper Analysis: [Paper Title]

- **Venue & Year**: ACL 2026 / NeurIPS / ICLR
- **Authors**: ...
- **Target Problem**: Memory retention, retrieval noise, context window limits

## 1. Key Contributions
- Contribution 1
- Contribution 2

## 2. Core Architecture & Algorithms
- Memory Storage: (e.g. episodic, semantic, working memory)
- Retrieval Mechanism: (e.g. dense, sparse, graph-based)
- Update / Consolidation Strategy: (e.g. periodic sleep, LRU, importance thresholding)

## 3. Experimental Setup & Baselines
- Datasets: ...
- Baselines: ...
- Key Results: ...

## 4. Implementation Plan for Agent-Mem-Lab
- [ ] Implement core memory store in `src/memory/`
- [ ] Add evaluation harness in `src/evaluation/`
- [ ] Run benchmark reproduction against `benchmarks/`
```

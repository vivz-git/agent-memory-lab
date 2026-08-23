# Reproducibility Guide & Benchmark Specification

**Paper**: *How Memory Management Impacts LLM Agents: An Empirical Study of Experience-Following Behavior* (ACL 2026)  
**Authors**: Zidi Xiong, Yuping Lin, Wenya Xie, Pengfei He, Zirui Liu, Jiliang Tang, Himabindu Lakkaraju, Zhen Xiang  
**Official Repository**: [https://github.com/yuplin2333/agent_memory_manage.git](https://github.com/yuplin2333/agent_memory_manage.git)  

---

## 1. Executive Summary & Research Foundations

This benchmark suite reproduces and stress-tests the dynamics of **episodic memory management** in Large Language Model (LLM) agents. LLM agents accumulate self-generated trajectory demonstrations over time. Without principled memory governance, agent performance degrades due to three fundamental phenomena:

1. **The Experience-Following Property**: Agents exhibit a high Pearson correlation ($r_{EF} \approx 0.85 - 0.99$) between query input similarity $S_{\text{in}}$ and output similarity $S_{\text{out}}$. As memory banks grow, agents increasingly mimic past retrieved demonstrations rather than reasoning from base policy principles.
2. **Error Propagation**: Unfiltered additions (e.g. *Add-All*) introduce low-quality or erroneous trajectories. When retrieved in future tasks, these flawed demonstrations are replicated and amplified, causing long-term performance degradation below a static fixed baseline.
3. **Misaligned Experience Replay**: Trajectories that satisfy initial coarse evaluation filters may still introduce subtle negative biases when retrieved under slightly shifted future contexts.
4. **History-Based Utility Deletion**: Tracking downstream task utility $\Phi(q, e)$ as free self-supervised quality feedback enables dynamic eviction of corrupt memories, isolating high-utility records and bounding memory growth under strict hardware constraints.

---

## 2. Experimental Protocols

```mermaid
graph TD
    subgraph Benchmark Protocols
        PA[Protocol A: Long-Term Memory Growth]
        PB[Protocol B: Utility Deletion KDE]
        PC[Protocol C: Distribution Shift Adaptation]
        PD[Protocol D: Resource-Constrained Memory]
    end

    PA -->|Outputs| MetricsA[Accuracy Curves, Memory Bloat M(t), Pearson r_EF]
    PB -->|Outputs| MetricsB[KDE Error Separation: Deleted vs Retained]
    PC -->|Outputs| MetricsC[Cluster Shift Recovery Speed & Stale Eviction]
    PD -->|Outputs| MetricsD[Capacity Bound Adherence & Accuracy Retention]
```

### Protocol A: Long-Term Memory Growth & Experience-Following Dynamics
- **Objective**: Measure task success rate and memory footprint over a streaming horizon of $T=1000$ to $4000$ steps.
- **Conditions Compared**:
  - `Fixed`: Frozen initial verified memory bank $D_0$ ($N_0 = 100$, $\pi = 0$).
  - `Add-All`: Unconditional insertion of every execution trajectory ($\pi = 1$).
  - `Coarse`: Filtered insertion via LLM judge / error threshold $\theta$.
  - `Strict`: Filtered insertion via ground-truth oracle verification.
- **Expected Outcome**: `Add-All` degrades over time; `Strict` monotonically improves; high Pearson correlation $r_{EF} > 0.80$ across growing memories.

### Protocol B: Memory Deletion & KDE Utility Separation
- **Objective**: Quantify the mathematical separation between deleted and retained experiences under history-based utility deletion.
- **Formulation**:
  $$\phi_{\text{hist}}(i, t) = \mathbb{I}\left[fr_t(i) \ge n \;\land\; \bar{\Phi}_t(i) \le \beta\right]$$
  *(Default parameters: minimum retrievals $n=3$, utility threshold $\beta=0.5$).*
- **Analysis**: Kernel Density Estimation (KDE) over error distributions of deleted vs. retained entries.
- **Expected Outcome**: $\mathbb{E}[\text{Error} \mid \text{Deleted}] > \mathbb{E}[\text{Error} \mid \text{Retained}]$, confirming that history-based deletion selectively purges flawed memories.

### Protocol C: Task Distribution Shift Adaptation
- **Objective**: Stress-test agent adaptation when the query distribution shifts non-stationarily.
- **Procedure**: Queries are partitioned into 3 distinct Gaussian Mixture Model (GMM) clusters and sequenced $C_0 \to C_1 \to C_2$.
- **Expected Outcome**: Combined periodic + history deletion quickly evicts inactive cluster exemplars, recovering task accuracy faster than static or unmanaged banks.

### Protocol D: Resource-Constrained Bounded Memory
- **Objective**: Enforce a strict hardware capacity bound $M_{\max} \in \{50, 100, 180\}$ using lowest-utility eviction:
  $$\text{Evict } \arg\min_{i} \bar{\Phi}_t(i) \quad \text{when } |D_t| > M_{\max}$$
- **Expected Outcome**: Compact bounded memory achieves $\ge 98\%$ of unbounded strict memory performance while drastically cutting token and retrieval costs.

### Engineering Extension: System-1 Adaptive Read Rejection
- **Objective**: Prevent toxic records from polluting the context window during their initial $n$ retrieval warmup steps before history deletion triggers.
- **Mechanism**: Dynamic post-retrieval utility mask applied pre-prompting.

---

## 3. Quickstart Reproduction Guide

### Option 1: Native Python CLI Runner

```bash
# 1. Activate virtual environment
source .venv/bin/activate  # or .venv\Scripts\Activate.ps1 on Windows

# 2. Run Protocol A (Memory Growth)
python scripts/run_reproduction.py --protocol A --env reg_agent --steps 1000

# 3. Run Protocol B (Utility Deletion KDE)
python scripts/run_reproduction.py --protocol B --env reg_agent --steps 1000

# 4. Run Protocol C (Distribution Shift)
python scripts/run_reproduction.py --protocol C --env reg_agent --steps 1000

# 5. Run Protocol D (Bounded Memory Capacity)
python scripts/run_reproduction.py --protocol D --env reg_agent --steps 1000 --capacity 100

# 6. Run Full Factorial Suite (All Protocols)
python scripts/run_reproduction.py --protocol all --env reg_agent --steps 1000
```

### Option 2: Isolated Docker Container Runner

```bash
# Run test suite
docker compose run --rm test

# Run individual protocols
docker compose run --rm protocol-a
docker compose run --rm protocol-b
docker compose run --rm protocol-c
docker compose run --rm protocol-d

# Run complete benchmark suite
docker compose run --rm reproduction
```

---

## 4. CLI Argument Reference

| CLI Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--protocol` | string | `A` | Target protocol: `A`, `B`, `C`, `D`, or `all` |
| `--env` | string | `reg_agent` | Benchmark environment: `reg_agent` or `ciciot` |
| `--steps` | int | `1000` | Horizon stream length |
| `--init-mem-size`| int | `100` | Initial verified demonstration bank size ($N_0$) |
| `--capacity` | int | `100` | Maximum memory limit ($M_{\max}$) for Protocol D |
| `--seed` | int | `42` | Global random seed for reproducible sampling |
| `--output-dir` | string | `./results` | Destination directory for logs and JSON metrics |
| `--backbone` | string | `gpt-4o-mini`| Foundation model backbone identifier |
| `--extension` | flag | `False` | Enable Adaptive Read Rejection extension |
| `--dry-run` | flag | `False` | Validate parameters and pipeline without heavy execution |
| `--verbose`, `-v`| flag | `False` | Enable verbose logging |

---

## 5. Output Artifacts & Metrics Structure

All experiment outputs are serialized into `--output-dir` (`./results` by default):

```text
results/
├── reproduction_manifest.json     # Global run metadata and protocol execution status
├── protocol_a_result.json         # Protocol A success rates, memory size, Pearson r_EF
├── protocol_b_result.json         # Protocol B KDE error stats (deleted vs retained)
├── protocol_c_result.json         # Protocol C cluster-specific accuracy & pruning counts
└── protocol_d_result.json         # Protocol D bounded capacity savings & adherence
```

### Manifest Schema Example:
```json
{
  "run_config": {
    "protocol": "all",
    "env": "reg_agent",
    "steps": 1000,
    "init_mem_size": 100,
    "capacity": 100,
    "seed": 42
  },
  "protocols_executed": {
    "A": { "protocol": "A", "baselines": { ... } },
    "B": { "protocol": "B", "kde_metrics": { ... } },
    "C": { "protocol": "C", "cluster_accuracies": { ... } },
    "D": { "protocol": "D", "capacity_satisfied": true }
  },
  "timestamp": 1724455200.0,
  "status": "success"
}
```

---

## 6. Scientific Rigor & Determinism Checklist

- [x] **Pinned Seeds**: All synthetic sampling uses deterministic random generators (`seed=42`).
- [x] **Split Isolation**: Initial memory $D_0$ generated strictly from non-overlapping partition $S_{\text{init}}$.
- [x] **Zero-Leakage Streaming**: Streaming queries $q_t$ evaluated sequentially without future context leakage.
- [x] **Container Isolation**: Multi-stage Docker environment guarantees reproducible dependency versions across host machines.

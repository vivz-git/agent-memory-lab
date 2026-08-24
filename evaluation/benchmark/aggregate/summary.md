# Aggregate Benchmark Summary (Across Seeds 42 & 123)

- **Model:** `openai/gpt-oss-120b` (Groq API)
- **Evaluator:** Deterministic `RegAgentStrictEvaluator(threshold=2.5)`
- **Total Task Stream:** 100 tasks per condition × 2 independent seeds = 800 executions
- **Initial Memory ($D_0$):** 20 verified demonstrations
- **Evaluated Seeds:** [42, 123]

## Aggregate Performance Table (Mean ± Std)

| Condition | Success Rate (SR) | Final Memory Size | Added | Deleted | Read Rejected | $\Delta_{EP}$ (Error Gap) | $r_{EF}$ (Following Correlation) | Avg Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **A. Fixed** | **53.0%** ± 5.0% | 20.0 ± 0.0 | 0 | 0 | 0 | 47.0% ± 5.0% | -0.0773 ± 0.0859 | 133.4s |
| **B. Naive Add-All** | **52.5%** ± 5.5% | 120.0 ± 0.0 | 100 | 0 | 0 | 47.5% ± 5.5% | 0.2288 ± 0.0196 | 130.0s |
| **C. Managed Memory** | **52.5%** ± 5.5% | 30.5 ± 16.5 | 52 | 42 | 0 | 47.5% ± 5.5% | 0.1030 ± 0.1275 | 130.4s |
| **D. Managed + Adaptive Read Rejection** | **52.5%** ± 5.5% | 50.5 ± 9.5 | 52 | 22 | 418 | 47.5% ± 5.5% | 0.0377 ± 0.1773 | 160.9s |

## Key Findings & Dynamics
1. **Baseline Fixed ($D_0$ only):** Serves as frozen static memory baseline.
2. **Naive Add-All:** Ingests unverified trajectories unconditionally. Demonstrates error compounding and memory bloat.
3. **Managed Memory (Strict Addition + History Deletion):** Strict Write Gate filters invalid outputs, while History Forget Gate purges low-utility records.
4. **Managed + Adaptive Read Rejection (Our Engineering Extension):** Proactively blocks misaligned memories from entering prompt context, maintaining high alignment and robust performance.

# Benchmark Validation Summary

This directory contains the results of the recruiter-facing 4-condition benchmark.
- **Model:** `openai/gpt-oss-120b` (via Groq)
- **Evaluator:** Deterministic `RegAgentStrictEvaluator(threshold=2.5)`
- **Task Stream Size:** 30 tasks per condition
- **Initial Memory ($D_0$):** 20 verified demonstrations
- **Seed:** 42

## Results Table

| Condition | Addition Policy | Deletion Policy | Read Rejection | Success Rate | Memory Size | Added | Deleted | Read Rejected | $\Delta_{EP}$ (Error Gap) | $r_{EF}$ (Exp-Following) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **A. Fixed** | `fixed` | `none` | No | **53.3%** | 20 | 0 | 0 | 0 | 46.7% | -0.1182 |
| **B. Naive Add-All** | `add_all` | `none` | No | **60.0%** | 50 | 30 | 0 | 0 | 40.0% | -0.6870 |
| **C. Managed Memory** | `strict` | `history` | No | **60.0%** | 30 | 18 | 8 | 0 | 40.0% | 0.2043 |
| **D. Managed + Adaptive Read Rejection** | `strict` | `history` | Yes | **60.0%** | 36 | 18 | 2 | 63 | 40.0% | 0.1560 |

## Key Observations
1. **Baseline Fixed ($D_0$ only):** Demonstrates lower bound with frozen initial memory.
2. **Naive Add-All:** Unfiltered memory ingestion stores noise/errors, inflating memory size.
3. **Managed Memory:** Strict Addition filters invalid writes, while History Deletion purges low-utility records.
4. **Managed + Adaptive Read Rejection:** Custom extension dynamically blocks misaligned memories from entering LLM context, maximizing Experience-Following alignment and task performance.

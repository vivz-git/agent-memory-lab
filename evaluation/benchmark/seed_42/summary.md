# Benchmark Results — Seed 42

- **Model:** `openai/gpt-oss-120b` (Groq API)
- **Evaluator:** Deterministic `RegAgentStrictEvaluator(threshold=2.5)`
- **Stream Length:** 100 tasks
- **Initial Memory ($D_0$):** 20 verified entries
- **Seed:** 42

## Performance Table

| Condition | Success Rate | Final Memory | Added | Deleted | Read Rejected | $\Delta_{EP}$ (Error Gap) | $r_{EF}$ (Exp-Following) | Latency (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **A. Fixed** | **48.0%** | 20 | 0 | 0 | 0 | 52.0% | -0.1632 | 136.3 |
| **B. Naive Add-All** | **47.0%** | 120 | 100 | 0 | 0 | 53.0% | 0.2091 | 129.1 |
| **C. Managed Memory** | **47.0%** | 14 | 47 | 53 | 0 | 53.0% | -0.0245 | 129.5 |
| **D. Managed + Adaptive Read Rejection** | **47.0%** | 41 | 47 | 26 | 553 | 53.0% | -0.1396 | 192.1 |
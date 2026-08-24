# Benchmark Results — Seed 123

- **Model:** `openai/gpt-oss-120b` (Groq API)
- **Evaluator:** Deterministic `RegAgentStrictEvaluator(threshold=2.5)`
- **Stream Length:** 100 tasks
- **Initial Memory ($D_0$):** 20 verified entries
- **Seed:** 123

## Performance Table

| Condition | Success Rate | Final Memory | Added | Deleted | Read Rejected | $\Delta_{EP}$ (Error Gap) | $r_{EF}$ (Exp-Following) | Latency (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **A. Fixed** | **58.0%** | 20 | 0 | 0 | 0 | 42.0% | 0.0086 | 130.4 |
| **B. Naive Add-All** | **58.0%** | 120 | 100 | 0 | 0 | 42.0% | 0.2484 | 130.9 |
| **C. Managed Memory** | **58.0%** | 47 | 58 | 31 | 0 | 42.0% | 0.2305 | 131.2 |
| **D. Managed + Adaptive Read Rejection** | **58.0%** | 60 | 58 | 18 | 283 | 42.0% | 0.2149 | 129.8 |
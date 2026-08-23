# RESEARCH SPECIFICATION: Memory Management & Experience-Following Dynamics

**Project Title**: How Memory Management Impacts LLM Agents: An Empirical Study of Experience-Following Behavior
**Artifact Target**: `research/RESEARCH_SPEC.md`
**Status**: DRAFT (Second Research Gate)

---

## 1. PROJECT DEFINITION

- **Project Title**: How Memory Management Impacts LLM Agents: An Empirical Study of Experience-Following Behavior
- **Research Problem**: LLM agents with episodic memory accumulate self-generated trajectories over time. Unfiltered additions introduce errors that are amplified by the agent's tendency to mimic past experiences, while redundant or misaligned records degrade long-term execution and bloat the context window.
- **Exact Research Question**: *How do the evolving dynamics of the memory bank, driven by continuous memory addition and deletion operations, influence long-term agent execution?*
- **Why this differs from CorrectRAG**: Standard RAG focuses on the precision of retrieving static, external human-verified knowledge documents. This paper focuses on the **internal dynamics of a self-evolving episodic memory**, where the agent's own noisy output trajectories are constantly appended to the retrieval pool. It isolates the distinct impacts of "Addition" (writing) and "Deletion" (forgetting) based on downstream utility.
- **Practical Hiring Signal**: Demonstrates deep competency in LLM agent cognitive architectures, rigorous empirical benchmark engineering, mathematical evaluation of self-improving systems, and the ability to reproduce a complex ACL paper locally with tight compute constraints.

---

## 2. PAPER FIDELITY

### What we reproduce faithfully:
- The fundamental definitions of the **Experience-Following Property**, **Error Propagation**, and **Misaligned Experience Replay**.
- The mathematical formulations for Input Similarity, Output Similarity, Pearson correlation ($r_{EF}$), and History-based Utility Deletion.
- The exact execution loop: Query -> Top-K Retrieval -> In-Context LLM Execution -> Evaluator -> Addition -> Deletion.
- The 4 core Addition strategies (Fixed, Add-All, Coarse, Strict) and 4 core Deletion strategies (None, Periodic, History, Combined).

### What we simplify:
- We will replace the fine-tuned LLM evaluator (C3) with few-shot prompting on a modern lightweight model (e.g., GPT-4o-mini) to avoid the overhead of fine-tuning a custom judge.

### What we intentionally do not reproduce:
- **AgentDriver (nuScenes) and EHRAgent (MIMIC-III)**: We explicitly exclude these two environments. NuScenes requires 100GB+ of raw sensor data and complex UniAD metric tracking. MIMIC-III requires setting up a clinical PostgreSQL database.
- **Why**: They bloat the project with infrastructure overhead and data governance issues, violating our mandate for a fast, lightweight, and scientifically meaningful local reproduction.

---

## 3. CORE MECHANISM

- **Experience-Following Property**: Agents heavily condition their generation on retrieved demonstrations. Measured by the Pearson correlation ($r$) between Input Similarity $S_{\text{in}}(q_t, q_k)$ and Output Similarity $S_{\text{out}}(e_t, e_k)$.
- **Memory Retrieval**: Given query $q_t$, fetch top-$K$ pairs $\xi_K$ maximizing $S_{\text{in}}$.
- **Memory Addition**: Binary gate $\pi(q_t, e_t) \in \{0, 1\}$.
  - *Fixed*: $\pi=0$.
  - *Add-All*: $\pi=1$.
  - *Coarse*: $\pi=1$ if LLM judge passes.
  - *Strict*: $\pi=1$ if ground-truth matches.
- **Memory Deletion**: Binary gate $\phi(q_i, e_i, t) \in \{0, 1\}$.
  - *Periodic* ($\phi_{\text{per}}$): Evict if retrieved $\le \alpha$ times in the last $T$ steps.
  - *History-based* ($\phi_{\text{hist}}$): Evict if total retrievals $\ge n$ and historical mean utility $\bar{\Phi} \le \beta$.
- **Error Propagation**: Low-quality additions introduce flawed demonstrations. When retrieved, the agent replicates and amplifies errors, causing long-term degradation.
- **Misaligned Experience Replay**: Demonstrations that superficially pass evaluators but provide suboptimal/toxic guidance in specific future contexts.
- **Utility/History Signal**: The use of downstream task success/failure as a free, self-supervised quality label $\Phi(q, e)$ to dynamically grade historical records.

---

## 4. SYSTEM ARCHITECTURE

- **Task Environment**: An abstract `Environment` interface providing states, executing actions, and yielding ground-truth rewards.
- **Agent**: An LLM-backed policy that accepts a query and retrieved context, outputting an execution trajectory.
- **Experience Representation**: Data structure containing `id`, `query_vector`, `trajectory`, `retrieval_count`, `utility_history`, `mean_utility`.
- **Memory Store**: A vectorized episodic bank `D_t` supporting insertion, deletion, and $K$-NN similarity search.
- **Retrieval**: Cosine similarity $S_{\text{in}}(q_t, q_i)$ matcher over query embeddings or feature vectors.
- **Memory Admission**: The `AdditionPolicy` module (evaluates $\pi$).
- **Memory Utility Evaluation**: The `UtilityTracker` module (appends task score $\Phi$ to the record's history).
- **Deletion/Eviction**: The `DeletionPolicy` module (evaluates $\phi$ asynchronously or post-task).
- **Execution Loop**: `run_episode(query) -> retrieve -> execute -> evaluate -> add -> update_utility -> delete`.
- **Evaluator**: `TrajectoryEvaluator` (Strict Oracle or Coarse LLM Judge).
- **Experiment Runner**: CLI pipeline to manage task streams, seeds, memory bounds, and metric logging.

---

## 5. REPRODUCTION SCOPE

**Selected Environments**:
1. **RegAgent (Primary)**
   - *Why*: A 6D synthetic linear regression task ($x \sim \mathcal{N}, y = w^T x + \epsilon$). It is perfectly controllable, mathematically pure, requires 0 external data, and runs in seconds. It isolated the exact mathematical phenomena of error compounding and utility deletion without LLM noise.
2. **CIC-IoT Agent (Secondary)**
   - *Why*: A tabular network traffic classification task (8 classes). It provides a real-world, semantic LLM reasoning analog without the prohibitive overhead of AgentDriver or EHRAgent.

These two environments form the fastest scientifically meaningful subset that preserves the research question and generates comparative evidence (synthetic vs. semantic).

---

## 6. BASELINES

A. **No-Memory Baseline (Fixed Baseline)**
   - Initial verified memory $D_0$ ($N=100$) is provided.
   - $\pi = 0$ (No new experiences are added).
   - Serves as the lower-bound for self-improving agents.

B. **Naive Memory-Growth Baseline (Add-All)**
   - $\pi = 1$ (Every trajectory is added).
   - $\phi = 0$ (No deletion).
   - Demonstrates Error Propagation and runaway context bloat.

C. **Selective Memory-Addition**
   - Addition filtered by Coarse LLM Judge or Strict Oracle.
   - No deletion.
   - Demonstrates the necessity of addition gating, but still vulnerable to misaligned replay.

D. **Selective Addition + Deletion**
   - Addition filtered by Coarse/Strict.
   - Deletion applied (Periodic, History, or Combined).
   - Demonstrates optimal bounded memory and utility purification.

*Identical parameters across baselines*: Initial memory $D_0$, stream queries, LLM temperature $T=0.0$, top-$K$ parameter.

---

## 7. EVALUATION

**Minimum High-Value Experiment Suite**:
1. **Long-Term Memory Growth & Evolution**
   - Run stream of $T=1000$ to $4000$ queries.
   - *Metrics*: Task Success Rate (SR), Memory Size $M(t)$, Pearson $r_{EF}$ for Experience-Following.
   - *Goal*: Show Add-All degrades vs Fixed, and Strict Addition improves.
2. **Memory Deletion & Utility Benefit**
   - Run Strict + History Deletion vs Strict + No Deletion.
   - *Metrics*: KDE Error Distribution of Deleted vs. Retained records.
   - *Goal*: Prove deleted records mathematically contain higher average error than retained ones.
3. **Task Distribution Shift**
   - Sequence queries using GMM clustering (Cluster A -> B -> C) to simulate domain shift.
   - *Goal*: Show History/Periodic deletion stabilizes performance by evicting stale cluster records.
4. **Resource-Constrained Memory**
   - Impose hard capacity limit $M_{\max} = 100$. Evict lowest-utility records when overflowing.
   - *Goal*: Show bounded memory + utility deletion matches unbounded strict memory.

**Rigor / Statistical Treatment**:
- 5 random seeds for synthetic generation.
- Strict isolation of initial memory generation vs test stream queries.

---

## 8. SECURITY / SAFETY

- **Sandbox Boundary**: All executions are tightly scoped. RegAgent executes closed-form math. CIC-IoT executes restricted LLM parsing. No arbitrary code execution (`exec`/`eval`) is permitted.
- **Allowed Tools**: No external OS/network tools.
- **Malformed-Memory Handling**: Strict Pydantic validation on memory records. If LLM outputs unparseable text, assign 0 utility and discard.
- **Prompt Injection**: User inputs (simulated cyber traffic) are sanitized and typed before insertion into the prompt.
- **Secret Handling**: OpenAI API keys managed strictly via `.env` and `pydantic-settings`, never logged.

---

## 9. ENGINEERING CONSTRAINTS

- **Language**: Python 3.10+
- **Frameworks**: Pydantic v2 (data validation), LiteLLM / OpenAI SDK (model abstraction), NumPy/SciPy (metrics), Matplotlib/Seaborn (plots).
- **Architecture**: Modular, SOLID principles, heavy use of dependency injection for evaluators and memory banks.
- **Infrastructure**: `requirements.txt` / `Dockerfile` provided for instant local booting.
- **No Unnecessary Frameworks**: No LangChain, AutoGen, or heavy agent frameworks. Pure Python logic.

---

## 10. FILE / MODULE PLAN

```text
src/
  agent/
    core.py              # Base agent loop
    prompts.py           # RegAgent & CIC-IoT prompt templates
  memory/
    bank.py              # Episodic memory store (vectorized retrieval)
    addition.py          # Fixed, Add-all, Coarse, Strict policies
    deletion.py          # Periodic, History, Combined policies
  environments/
    reg_agent_env.py     # 6D Synthetic environment
    ciciot_env.py        # Tabular IoT environment
  evaluation/
    metrics.py           # Pearson r_EF, L2 errors, SR
    evaluator.py         # Coarse LLM Judge and Strict Oracles
  utils/
    logger.py            # Artifact & metrics serializer
    shift.py             # GMM clustering logic

tests/
  test_memory.py
  test_environments.py

scripts/
  run_experiments.py     # CLI entry point

research/
  RESEARCH_SPEC.md
  ...
```

---

## 11. MILESTONES

- **M1**: Research Spec lock (Current).
- **M2**: Environments & Memory Primitives (Vector store, addition/deletion interfaces, RegAgent synthetic generator).
- **M3**: Agent Loop & Evaluators (In-context prompting, Strict/Coarse judges).
- **M4**: Memory Management Integration (Hooking up utility tracking and eviction logic).
- **M5**: Benchmark Runner (Stream generation, metric collection).
- **M6**: Experiments Execution (Running the 4 core protocols and generating plots).
- **M7**: QA & Validation (Check Pearson $r$ correlations and KDE graphs against paper).
- **M8**: Documentation (README with paper comparison and reproduction results).

---

## 12. DEFINITION OF DONE

The project is fully complete when:
- RegAgent and CIC-IoT Agent environments are functional.
- Fixed, Add-All, Coarse, Strict addition implemented.
- Periodic and History-based deletion implemented.
- 4 core experiments (Growth, Deletion KDE, Shift, Bounded) execute successfully.
- Results and plots are serialized and confirm the paper's core claims.
- `pytest` suite passes for core memory logic.
- README accurately documents the reproduction architecture and results.
- Git history contains clean atomic commits mapping to milestones.

---

## 13. OUR ENGINEERING EXTENSION

**Focused Extension**: *Adaptive Retrieval Filtering via Utility Thresholding (System-1 "Read" Rejection)*

- **Concept**: The paper focuses exclusively on filtering at the "Write" (Addition) and "Forget" (Deletion) stages. However, history-based deletion only kicks in after a record has been retrieved $n$ times. If a toxic record is in memory, it will still pollute the context window $n$ times before eviction.
- **Implementation**: We implement a "Read" filter. When the retriever fetches top-$K$ records, we apply an adaptive mask. If a retrieved record $k$ has a historical mean utility $\bar{\Phi}_t(k)$ significantly below the agent's moving average utility, we dynamically reject it from entering the LLM prompt, backing off to the $(K+1)$-th record.
- **Feasibility**: Extremely low overhead. It's a simple boolean mask applied post-retrieval and pre-prompting.
- **Measurability**: We will plot the Error Propagation gap ($\Delta_{EP}$) for History-Deletion vs. History-Deletion + Read-Rejection. We expect Read-Rejection to dampen the initial performance dip seen before the $n$ threshold is reached.

# Academic Paper Forensic Analysis Report
**Title**: How Memory Management Impacts LLM Agents: An Empirical Study of Experience-Following Behavior  
**Authors**: Zidi Xiong, Yuping Lin, Wenya Xie, Pengfei He, Zirui Liu, Jiliang Tang, Himabindu Lakkaraju, Zhen Xiang  
**Affiliations**: Harvard University, University of Georgia, Michigan State University, University of Minnesota-Twin Cities  
**Publication**: Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (ACL 2026, Volume 1: Long Papers), pages 623–645  
**Artifact Codebase**: `https://github.com/yuplin2333/agent_memory_manage.git`  
**Evaluation Scope**: Strict forensic extraction and evidence-based synthesis of `paper/2026.acl-long.27.pdf`.

---

## 1. Executive Summary & Core Research Questions

### 1.1 Executive Summary
Episodic memory systems in Large Language Model (LLM) agents store past task queries (inputs) and execution trajectories (outputs) to serve as in-context demonstrations for future queries. While static in-context learning relies on fixed, curated demonstrations, agentic episodic memory is dynamic and inherently noisy: memory banks evolve continuously through additions and deletions, accumulating self-generated trajectories.

This paper conducts the first systematic, quantitative study of how foundational memory management operations—**Memory Addition** and **Memory Deletion**—impact the long-term behavioral dynamics and task performance of LLM agents. Across four distinct agent domains (synthetic regression, healthcare EHR analysis, autonomous driving, and IoT security), the authors discover and formalize the **Experience-Following Property**: agents exhibit a high propensity to replicate retrieved executions when query similarity is high ($r \approx 0.82 \text{--} 0.97$). While this property powers self-improvement when memory records are accurate, it triggers two severe failure modes: **Error Propagation** (amplification of past mistakes) and **Misaligned Experience Replay** (retrieval of suboptimal demonstrations). The study shows that trajectory evaluators acting as addition filters and history-based deletion gates (utilizing downstream task performance as free supervision) can curb error compounding, reduce memory footprint by $>75\%$, and outperform error-free ground-truth baselines.

### 1.2 Core Research Questions & Hypotheses
1. **Primary Research Question**: *How do the evolving dynamics of the memory bank, driven by continuous memory addition and deletion operations, influence long-term agent execution?* (§1, Page 623; §3, Page 625)
2. **Hypothesis 1 (Experience-Following)**: LLM agents condition their generation heavily on retrieved episodic demonstrations, such that the similarity between a current query and a retrieved query directly dictates the similarity between the generated execution and the retrieved execution (§1, Page 623; §3.3, Page 627).
3. **Hypothesis 2 (Error Compounding)**: Unfiltered or coarsely filtered additions introduce noisy trajectories that are mimicked and amplified over time, causing progressive long-term self-degradation (§3.4, Page 627–628).
4. **Hypothesis 3 (Downstream Utility as Free Quality Labels)**: A memory record’s intrinsic quality is manifested in the downstream performance of tasks that retrieve it. Tracking cumulative downstream utility allows the agent to identify and delete misaligned records without external labels (§4.1, Page 628; §4.3, Page 629–630).
5. **Hypothesis 4 (Bounded Memory Sufficiency)**: Unbounded memory accumulation is unnecessary; strategically pruning low-utility and unretrieved records preserves or enhances performance under strict memory capacity constraints (§5.2, Page 630–631).

---

## 2. Methodology & Theoretical Formulations

```
+-----------------------------------------------------------------------------------+
|                              Agent Execution Lifecycle                             |
+-----------------------------------------------------------------------------------+
                                          |
                                   Task Query q_t
                                          |
                                          v
                           +------------------------------+
                           |     1. Memory Retrieval      |
                           |   xi_K = argtop-K Sim_in     |
                           +------------------------------+
                                          |
                               Retrieved Pairs xi_K
                                          |
                                          v
                           +------------------------------+
                           |  2. In-Context Reasoning &   |
                           |     Execution Generation     |
                           |       e_t ~ LLM(q_t, xi_K)   |
                           +------------------------------+
                                          |
                                  Trajectory e_t
                                          |
                                          v
                           +------------------------------+
                           |    3. Environment Action &   |
                           |      Evaluator Scoring       |
                           |      score = Phi(q_t, e_t)   |
                           +------------------------------+
                                          |
                  +-----------------------+-----------------------+
                  |                                               |
                  v                                               v
   +------------------------------+               +------------------------------+
   |      4. Memory Addition      |               |      5. Memory Deletion      |
   |      pi(q_t, e_t) in {0, 1}  |               |    phi(q_i, e_i, t) in {0, 1}|
   |  Store if evaluator passes   |               | Periodic / History / Combined|
   +------------------------------+               +------------------------------+
                  |                                               |
                  +-----------------------> <---------------------+
                                          |
                                          v
                           +------------------------------+
                           |   Updated Memory Bank D_t+1  |
                           +------------------------------+
```

### 2.1 Agent Memory Bank & Memory Reading (Retrieval)
Let the episodic memory bank at time step $t$ contain $N$ query-execution trajectory pairs:
$$\mathcal{D}_t = \{(q_1, e_1), (q_2, e_2), \dots, (q_N, e_N)\}$$
Given an incoming query $q_t$, the retrieval module retrieves the top-$K$ most relevant records $\xi_K \subset \mathcal{D}_t$:
$$\xi_K = \operatorname{arg\,top-}K_{(q_i, e_i) \in \mathcal{D}_t} \operatorname{Sim}_{\text{in}}(q_t, q_i)$$
The retrieved set $\xi_K$ is serialized into the agent prompt as in-context exemplars to generate trajectory $e_t \sim \text{LLM}(q_t, \xi_K)$ (§3, Page 625).

### 2.2 Memory Addition Formulations
Memory addition evaluates $(q_t, e_t)$ via an addition decision rule $\pi(q_t, e_t) \in \{0, 1\}$ (§3.1, Page 625–626):
$$\mathcal{D}_{t+1}^{\text{add}} = \begin{cases} \mathcal{D}_t \cup \{(q_t, e_t)\}, & \text{if } \pi(q_t, e_t) = 1 \\ \mathcal{D}_t, & \text{if } \pi(q_t, e_t) = 0 \end{cases}$$

1. **Fixed Memory Baseline**:
   $$\pi_{\text{fixed}}(q, e) = 0 \quad (\forall q, e)$$
   Relies purely on the initial curated seed memory $\mathcal{D}_0$.
2. **Add-All Approach**:
   $$\pi_{\text{all}}(q, e) = 1 \quad (\forall q, e)$$
   Indiscriminately appends every executed trajectory.
3. **Selective Addition (Coarse / Automatic Evaluator)**:
   $$\pi_{\text{automatic}}(q, e) = \mathbf{1}[\text{Evaluator}_{\text{auto}}(q, e) \ge \tau_{\text{add}}]$$
   - **RegAgent**: Error threshold $|y - \hat{y}| \le \tau_{\text{add}}$, with $\tau_{\text{add}} \in \{1.6 \text{ (C1)}, 1.4 \text{ (C2)}, 1.2 \text{ (C3)}\}$.
   - **Real Agents**: 
     - **C1**: Zero-shot prompt evaluation via GPT-4o-mini.
     - **C2**: Zero-shot prompt evaluation via GPT-4.1-mini.
     - **C3**: Fine-tuned GPT-4.1-mini trained on 300 domain-specific judge trajectories.
4. **Selective Addition (Strict / Oracle Evaluator)**:
   $$\pi_{\text{strict}}(q, e) = \mathbf{1}[\text{GroundTruthMatch}(q, e)]$$
   - **RegAgent**: Absolute error $|y - \hat{y}| \le 1.0$.
   - **EHRAgent**: Exact answer match against clinical database query ground truth.
   - **AgentDriver**: UniAD 3-second average waypoint $L_2$ distance $< 2.5\text{m}$.
   - **CIC-IoT Agent**: Exact string match of attack label against ground truth.

### 2.3 Memory Deletion Formulations
Memory deletion evaluates existing records $(q_i, e_i) \in \mathcal{D}$ using a deletion decision rule $\phi(q_i, e_i) \in \{0, 1\}$ (§4.1, Page 628):

1. **Periodical-Based Deletion ($\phi_{\text{per}}$)**:
   Tracks retrieval counts within a rolling execution window $[t', t]$:
   $$\phi_{\text{per}}(q_i, e_i, t, t') = \mathbf{1}[\text{fr}_t(q_i, e_i) - \text{fr}_{t'}(q_i, e_i) \le \alpha]$$
   where $\text{fr}_t(q_i, e_i)$ denotes total retrievals up to step $t$, and $\alpha$ is the minimum activity threshold.
   - **Theoretical Memory Bound**: The total memory size $M$ remains strictly bounded:
     $$M \le \alpha (t - t') K$$
     where $K$ is the number of retrieved items per step.
2. **History-Based (Utility-Based) Deletion ($\phi_{\text{hist}}$)**:
   Measures the average downstream execution utility across all instances where $(q_i, e_i)$ was retrieved:
   $$\phi_{\text{hist}}(q_i, e_i, t) = \begin{cases} \mathbf{1}\left[ \frac{1}{\text{fr}_t(q_i, e_i)} \sum_{m=1}^{\text{fr}_t(q_i, e_i)} \Phi(q_m, e_m) \le \beta \right], & \text{if } \text{fr}_t(q_i, e_i) > n \\ 0, & \text{otherwise} \end{cases}$$
   where $\Phi(q_m, e_m)$ is the utility score of downstream task $m$, $n$ is the minimum retrieval threshold preventing estimation bias, and $\beta$ is the utility threshold.
3. **Combined Deletion ($\phi_{\text{comb}}$)**:
   $$\phi_{\text{comb}}(q_i, e_i, t, t') = \phi_{\text{per}}(q_i, e_i, t, t') \lor \phi_{\text{hist}}(q_i, e_i, t)$$

### 2.4 Mathematical Formulation of Experience-Following Property
For any test execution at step $t$, let $\text{Sim}_{\text{in}}(q_t, q_{\text{top1}})$ and $\text{Sim}_{\text{out}}(e_t, e_{\text{top1}})$ denote the input and output similarities to the closest retrieved demonstration (§3.3, Page 627; §A.1, Page 634–635):

- **RegAgent**:
  $$\text{Sim}_{\text{in}}(x_1, x_2) = \frac{x_1^\top x_2}{\|x_1\|_2 \|x_2\|_2}, \quad \text{Sim}_{\text{out}}(y_1, y_2) = \exp(-\gamma |y_1 - y_2|^2), \quad \gamma = 1.0$$
- **EHRAgent**:
  $$\text{Sim}_{\text{in}}(q_1, q_2) = \cos(\mathbf{E}_{\text{text}}(q_1), \mathbf{E}_{\text{text}}(q_2)), \quad \text{Sim}_{\text{out}}(e_1, e_2) = \text{pycode\_similar}(e_1, e_2)$$
- **AgentDriver**:
  $$\text{Sim}_{\text{in}} = \exp(-L_2(\mathbf{s}_1, \mathbf{s}_2)), \quad \text{Sim}_{\text{out}}(\mathbf{v}_1, \mathbf{v}_2) = \exp(-\gamma \|\mathbf{v}_1 - \mathbf{v}_2\|_2^2), \quad \gamma = 1.0$$
- **CIC-IoT Agent**:
  $$S_{\text{cont}}(f_i) = \frac{|x_1(f_i) - x_2(f_i)|}{\max(|x_1(f_i)|, |x_2(f_i)|)}, \quad S_{\text{disc}}(f_i) = \mathbf{1}[x_1(f_i) \ne x_2(f_i)]$$
  $$\text{Sim}_{\text{in}} = 1 - \frac{1}{|F|}\sum_{i=1}^{|F|} S(f_i), \quad \text{Sim}_{\text{out}} = \cos(\mathbf{E}_{\text{text}}(e_1), \mathbf{E}_{\text{text}}(e_2))$$

The overall **Experience-Following Property** is quantified by the Pearson correlation coefficient $r$:
$$r = \frac{\sum_{t=1}^T (\text{Sim}_{\text{in}}^{(t)} - \bar{S}_{\text{in}})(\text{Sim}_{\text{out}}^{(t)} - \bar{S}_{\text{out}})}{\sqrt{\sum_{t=1}^T (\text{Sim}_{\text{in}}^{(t)} - \bar{S}_{\text{in}})^2} \sqrt{\sum_{t=1}^T (\text{Sim}_{\text{out}}^{(t)} - \bar{S}_{\text{out}})^2}}$$

---

## 3. Experimental Setup & Benchmarks

### 3.1 Agent Architecture Matrix (Table 3, Page 635)

| Agent Name | Task Domain | Input Format | Output Trajectory Format | Retrieval Feature & Metric | Retrieved Demos ($K$) | Initial Mem ($|\mathcal{D}_0|$) | Test Stream Size | Primary Metric |
|---|---|---|---|---|---|---|---|---|
| **RegAgent** | 6D Linear Regression with Noise | 6D vector $x \sim \mathcal{N}(\mu, 1), \mu \in \{-0.5, 0, 0.5\}$ | Scalar guess $\hat{y} = w^\top x + \epsilon, \epsilon \in [-1, 1]$ | 6D vector cosine similarity | $K=6$ | 100 | 4,000 | Success Rate (SR, $|\hat{y} - y| \le 1.0$) |
| **EHRAgent** | EHR DB Question Answering | Natural language clinical query | Python multi-step DB tool execution / SQL | `text-embedding-3-large` cosine sim | $K=4$ | 100 | 2,392 | Accuracy (ACC, exact answer match) |
| **AgentDriver** | Autonomous Vehicle Trajectory Planning | Ego-state, perception, goal, history vector | 6-step waypoint trajectory ($x$ lateral, $y$ longitudinal) | Weighted state/goal/history $L_2$ distance | $K=1$ | 180 | 2,000 | Success Rate (SR, 3s avg $L_2 < 2.5\text{m}$) |
| **CIC-IoT Agent** | Network Traffic Attack Detection | 33 packet flow features (8 classes) | Chain-of-Thought reasoning + attack label | Feature-wise relative change similarity | $K=3$ | 100 (synthetic) | 1,000 | Accuracy (ACC, string match) |

### 3.2 Evaluated LLM Backbones
- **Primary Backbone**: `GPT-4o-mini` across all main experiments (§3.1, Page 625).
- **Advanced Proprietary Backbones**: `GPT-4o`, `DeepSeek-V3` (§B.2, Figures 10–12, Page 641–642).
- **Open-Source Backbones**: `Qwen3-32B`, `Qwen3-14B` (§B.2, Table 4, Page 642).

### 3.3 Deletion Hyperparameters Summary (Appendix A.1, Page 634–635)
- **RegAgent**:
  - Periodic: Period = 500 steps, $\alpha = 0$.
  - History: Min retrievals $n = 5$, utility threshold $\beta = 0.5$ (absolute error $\le 1.0$).
- **EHRAgent**:
  - Periodic: Period = 200 steps, $\alpha = 0$.
  - History: Min retrievals $n = 5$, $\beta = 0.3$ (for 4o-mini / 4.1-mini) and $\beta = 0.7$ (for Strict / 4.1-mini FT).
- **AgentDriver**:
  - Periodic: Period = 500 steps, $\alpha = 0$.
  - History: Min retrievals $n = 3$, mean UniAD 3-second L2 distance threshold $> 5.0\text{m}$ (for strict) or accumulated success rate $< 0.5$ (for coarse).
- **CIC-IoT Agent**:
  - Periodic: Period = 500 steps, $\alpha = 1$.
  - History: Min retrievals $n = 3$, quality threshold $\beta = 0.7$.

---

## 4. Comprehensive Empirical Results (Exact Numbers Extracted)

### 4.1 Table 1: Memory Addition Strategies Performance (Page 626)
*Evaluation of fixed memory, add-all, three coarse automatic evaluators (C1, C2, C3), and strict evaluator across 4 agents.*

| Judge / Strategy | RegAgent SR (%) ↑ | RegAgent Mem Size ↓ | EHRAgent ACC (%) ↑ | EHRAgent Mem Size ↓ | AgentDriver SR (%) ↑ | AgentDriver Mem Size ↓ | CIC-IoT Agent ACC (%) ↑ | CIC-IoT Agent Mem Size ↓ |
|---|---|---|---|---|---|---|---|---|
| **Fixed (Baseline)** | 67.53 | 100 | 16.75 | 100 | 40.11 | 180 | 71.50 | 50* |
| **Add All** | 55.48 | 4100 | 13.05 | 2411 | 32.32 | 2125 | 59.90 | 1050 |
| **Coarse C1** | 63.18 | 3511 | 26.19 | 1447 | 36.92 | 1161 | 74.00 | 1030 |
| **Coarse C2** | 65.78 | 3347 | 32.21 | 1467 | 40.01 | 1119 | 68.80 | 936 |
| **Coarse C3** | 67.35 | 3139 | 34.66 | 1094 | 47.37 | 1285 | 79.50 | 952 |
| **Strict** | **70.95** | 2938 | **38.50** | 1012 | **51.00** | 1178 | **85.40** | 904 |

*\*Note: In CIC-IoT, the fixed baseline active entries evaluated was 50.*

### 4.2 Table 2: Memory Deletion Strategies Performance (Page 629)
*Interaction between Addition Evaluators (Coarse C1 vs Strict) and Deletion Strategies (No Deletion, Periodical, History-based, Combined).*

| Addition Judge | Deletion Strategy | RegAgent SR (%) ↑ | RegAgent Mem Size ↓ | EHRAgent ACC (%) ↑ | EHRAgent Mem Size ↓ | AgentDriver SR (%) ↑ | AgentDriver Mem Size ↓ | CIC-IoT Agent ACC (%) ↑ | CIC-IoT Agent Mem Size ↓ |
|---|---|---|---|---|---|---|---|---|---|
| **Coarse (C1)** | **No del** | 63.18 | 3511 | 25.91 | 1447 | 36.92 | 1161 | 74.00 | 1030 |
| | **Period** | 60.88 | 1012 | 26.65 | 338 | 36.38 | 426 | 78.10 | 355 |
| | **History** | 62.10 | 3205 | 33.55 | 1004 | 34.00 | 1019 | 73.70 | 952 |
| | **Combined** | 59.32 | 951 | 31.47 | 279 | 35.62 | 372 | 68.80 | 352 |
| **Strict** | **No del** | 70.95 | 2938 | 38.67 | 1012 | 51.00 | 1178 | 85.40 | 904 |
| | **Period** | 67.65 | 949 | 38.59 | 302 | 50.94 | 467 | 80.80 | 310 |
| | **History** | 69.80 | 2286 | 42.06 | 784 | **51.81** | 846 | **89.60** | 788 |
| | **Combined** | 66.58 | 890 | **42.34** | 248 | 49.97 | 323 | 85.50 | 188 |

### 4.3 Table 4: Open-Source Qwen Backbones on RegAgent (Page 642)
*Reports SR (%) with Pearson correlation $r$ for addition; SR (%) with (retained / deleted mean quality error) for deletion.*

| Strategy | Qwen3-32B SR (%) [Pearson $r$ / Quality] | Qwen3-14B SR (%) [Pearson $r$ / Quality] |
|---|---|---|
| **Add All** | 56.9 ($r = 0.74$) | 55.4 ($r = 0.82$) |
| **Coarse** | 67.9 ($r = 0.69$) | 65.7 ($r = 0.76$) |
| **Strict** | 72.9 ($r = 0.72$) | 72.9 ($r = 0.89$) |
| **Coarse + Hist** | 66.5 (retained 0.61 / deleted 1.01) | 64.4 (retained 0.63 / deleted 0.97) |
| **Strict + Hist** | 68.4 (retained 0.40 / deleted 0.54) | 73.6 (retained 0.41 / deleted 0.51) |

### 4.4 Table 5: Robustness across Retrieval Counts $K$ (Page 642)

| Agent | Setting 1 | Setting 2 | Setting 3 |
|---|---|---|---|
| **RegAgent** | **6 demos** | **3 demos** | **12 demos** |
| - Add All | 55.5% ($r = 0.95$) | 57.2% ($r = 0.92$) | 49.6% ($r = 0.90$) |
| - Strict | 71.0% ($r = 0.92$) | 75.0% ($r = 0.90$) | 64.1% ($r = 0.95$) |
| **CIC-IoT Agent** | **3 demos** | **1 demo** | **5 demos** |
| - Add All | 71.5% ($r = 0.87$) | 73.2% ($r = 0.91$) | 78.2% ($r = 0.93$) |
| - Strict | 85.4% ($r = 0.82$) | 80.4% ($r = 0.84$) | 84.1% ($r = 0.85$) |

### 4.5 Table 6: Robustness across Deletion Hyperparameters (Page 642)

| Agent / Parameter Sweep | Setting 1 | Setting 2 | Setting 3 |
|---|---|---|---|
| **RegAgent History ($n$)** | $n=5$: **69.8%** (0.44 / 0.53) | $n=10$: **67.4%** (0.45 / 0.57) | $n=15$: **70.1%** (0.44 / 0.56) |
| **RegAgent History ($\beta$)** | $\beta=0.5$: **69.8%** (0.44 / 0.53) | $\beta=0.4$: **67.3%** (0.43 / 0.55) | $\beta=0.6$: **62.0%** (0.41 / 0.48) |
| **RegAgent Periodic (period)** | period = 500: **67.7%** | period = 100: **53.2%** | period = 300: **64.0%** |
| **CIC-IoT History ($n$)** | $n=3$: **89.6%** | $n=5$: **87.6%** | $n=7$: **88.7%** |
| **CIC-IoT History ($\beta$)** | $\beta=0.7$: **89.6%** | $\beta=0.6$: **90.2%** | $\beta=0.8$: **89.6%** |
| **CIC-IoT Periodic (period)** | period = 500: **80.8%** | period = 100: **79.8%** | period = 300: **79.1%** |

### 4.6 Table 7: Size-Matched Quality Evaluation on RegAgent (Page 643)
*Subsampling 1,000 frequently retrieved memories from final banks, evaluated on 1,000 fresh test cases.*

| Memory Bank Strategy | Success Rate (SR) ↑ (%) |
|---|---|
| **Strict Addition + History-Based Deletion** | **74.4%** |
| **Strict Addition Only (No Deletion)** | 72.8% |

### 4.7 Table 8: Ground-Truth Correctness Rate of Retained vs. Deleted Records (Page 644)

| Agent | Memory Category | GPT-4o-mini Judge | GPT-4.1-mini Judge | GPT-4.1-mini FT Judge |
|---|---|---|---|---|
| **EHRAgent** | **Retained Records** | **44.1%** | **49.1%** | **54.8%** |
| | **Deleted Records** | 36.3% | 32.1% | 48.2% |
| **CIC-IoT Agent** | **Retained Records** | **78.9%** | **72.2%** | **86.6%** |
| | **Deleted Records** | 56.7% | 55.1% | 61.0% |

---

## 5. Visual Trends & Figure Forensics

- **Figure 1 (Page 623)**: Memory management workflow diagram showing retrieval of $(\mathbf{Q}_N, \mathbf{E}_N)$, execution $\mathbf{E}_t$, addition decision $\pi(\mathbf{Q}_t, \mathbf{E}_t)$, and deletion decision $\phi(\mathbf{Q}_k, \mathbf{E}_k)$.
- **Figure 2 (Page 626)**: Performance over time (running task index) for EHRAgent (2400 tasks) and AgentDriver (2000 tasks). Shows strict evaluator and C3 (fine-tuned) achieving continuous self-improvement, while Add-All and C1/C2 decline or stagnate.
- **Figure 3 (Page 627)**: Output similarity vs. Input similarity scatter/trend for RegAgent and AgentDriver. Add-all ($r=0.95$), Strict ($r=0.92$), C1 ($r=0.94$), C2 ($r=0.97$), C3 ($r=0.96$) show high linear correlation; Fixed memory ($r=0.52$) clusters in low similarity.
- **Figure 4 (Page 627)**: Error propagation comparison vs. Error-Free (EF) ground-truth variant on RegAgent and AgentDriver. Shows immediate performance gap between agent execution memory and error-free memory; Add-All and Coarse worsen this gap, whereas Strict addition approaches and even surpasses ground-truth after ~2000 tasks in AgentDriver.
- **Figure 5 (Page 629)**: Performance bar charts comparing Add-Only vs. Hist-Del across 4o-mini, 4.1-mini, 4.1-mini FT, and Strict on EHRAgent and AgentDriver.
- **Figure 6 (Page 630)**: Kernel Density Estimation (KDE) curves of absolute error for deleted vs. retained memories in RegAgent. Left: Coarse 1.6 (Deleted Mean = 1.0978, Retained Mean = 0.8129). Right: Strict (Deleted Mean = 0.5296, Retained Mean = 0.4434).
- **Figure 7 (Page 630)**: Performance under Task Distribution Shift (GMM clustered 3 task groups). Vertical line indicates shift timestamp. Shows combined deletion maintains robust stability across shifts, outperforming history-only deletion on EHRAgent.
- **Figure 8 (Page 641)**: Accuracy trends over long-term task stream on EHRAgent and CIC-IoT across all addition strategies.
- **Figure 9 (Page 641)**: Cumulative output similarity vs. input similarity for EHRAgent (Strict $r=0.88$, Add-all $r=0.61$, Fixed $r=-0.73$) and CIC-IoT (Strict $r=0.82$, Add-all $r=0.87$, Fixed $r=0.19$).
- **Figure 10 (Page 641)**: Long-term performance of GPT-4o backbone on AgentDriver (Fixed, Strict, Strict+Hist, Strict+Comb).
- **Figure 11 (Page 642)**: Long-term performance of DeepSeek-V3 backbone on AgentDriver (Fixed, Strict, Strict+Hist, Strict+Comb).
- **Figure 12 (Page 642)**: Output vs. Input similarity across backbones on AgentDriver: GPT-4o-mini ($r=0.85$), GPT-4o ($r=0.71$), DeepSeek-V3 ($r=0.60$).
- **Figure 13 (Page 643)**: Comparison between history-based deletion and Error-Free (EF) variants over 2000/4000 tasks. Around task 1000, Strict+Comb in AgentDriver surpasses Error-Free baseline.
- **Figure 14 (Page 643)**: Performance comparison of history deletion across evaluators on RegAgent and CIC-IoT.
- **Figure 15 (Page 643)**: KDE curves for RegAgent under Coarse 1.2 (Deleted mean 0.6842 vs Retained mean 0.4932) and Coarse 1.4 (Deleted mean 0.8703 vs Retained mean 0.5507).
- **Figure 16 (Page 644)**: KDE curves for AgentDriver L2 error of deleted vs retained records across evaluators: Strict (Del 6.2174 vs Ret 2.7180), 4o-mini (Del 5.9805 vs Ret 6.7042), 4.1-mini (Del 6.4500 vs Ret 4.6277), 4.1-mini FT (Del 7.0483 vs Ret 3.3856).
- **Figure 17 (Page 645)**: Memory resource constraints (fixed capacity: 100 records for EHRAgent, 180 for AgentDriver) comparing unlimited vs limited memory. Combined deletion under fixed capacity retains high performance.
- **Figure 18 (Page 645)**: AgentDriver performance vs bounded memory capacity limit (180, 270, 360, 450 records). Performance converges rapidly with strict evaluator, proving unbounded memory growth is unnecessary.

---

## 6. Critical Limitations, Failure Modes & Edge Cases

1. **Scope Restriction to Basic Operations**:
   - The study intentionally restricts analysis to atomic addition and deletion operations, omitting complex transformations like knowledge-graph structuring (AriGraph), recursive summarization (AWM), or verbal reflection (Reflexion). While isolating fundamentals is scientifically rigorous, interactions with complex memory architectures require further research (§6, Page 631).
2. **Evaluator Quality Inversion (The Coarse Evaluator Trap)**:
   - Untuned zero-shot LLM evaluators can cause **quality inversion**, where retained memories possess higher average error than deleted memories (e.g., AgentDriver under GPT-4o-mini: retained mean $L_2 = 6.7042\text{m}$ vs deleted mean $L_2 = 5.9805\text{m}$, §B.7, Figure 16, Page 644).
3. **The Imitation vs. Reasoning Dilemma**:
   - High experience-following ($r \to 1.0$) suppresses the LLM's intrinsic step-by-step reasoning, making the agent brittle when retrieved exemplars contain subtle intermediate logic errors (§3.3, Page 627).
4. **Volume vs. Quality Divergence Across Domains**:
   - In synthetic continuous vector spaces (RegAgent), high memory density provides geometrical coverage advantages (strict no-deletion 70.95% vs strict+hist 69.80%). However, in structured real-world tasks (AgentDriver, EHRAgent, CIC-IoT), memory quality and utility drastically outweigh sheer volume (§4.2, Page 629).
5. **Lack of Theoretical Convergence Bounds**:
   - The findings are strictly empirical. Proving formal PAC-learning convergence guarantees or regret bounds under non-stationary task streams remains an open theoretical challenge (§6, Page 631).

---

## 7. Page & Section Citation Index

| Topic / Finding | Section | Page Number(s) | Table / Figure Reference |
|---|---|---|---|
| **Evolving Memory Dynamics Motivation** | Section 1 | 623–624 | Figure 1 |
| **Episodic vs Semantic/Procedural Memory Taxonomy** | Section 2.1 | 624 | Paragraph 2 |
| **Agent Execution Cycle & Retrieval Definition** | Section 3 | 625 | Paragraph 1 |
| **Addition Strategies (Fixed, Add-all, Coarse, Strict)** | Section 3.1 | 625–626 | Section 3.1 |
| **Addition Empirical Results Across 4 Agents** | Section 3.2 | 626 | Table 1, Figure 2 |
| **Experience-Following Property Discovery & $r$ Values** | Section 3.3 | 627 | Figure 3 |
| **Error Propagation & Error-Free (EF) Comparisons** | Section 3.4 | 627–628 | Figure 4 |
| **Deletion Formulations ($\phi_{\text{per}}, \phi_{\text{hist}}, \phi_{\text{comb}}$)** | Section 4.1 | 628 | Equations in §4.1 |
| **Deletion Empirical Results Across 4 Agents** | Section 4.2 | 629 | Table 2, Figure 5 |
| **Misaligned Experience Replay & KDE Density Curves** | Section 4.3 | 629–630 | Figure 6 |
| **Task Distribution Shift via GMM Clustering** | Section 5.1 | 630 | Figure 7, Appendix A.5 (p. 640–641) |
| **Bounded Memory Resource Constraints** | Section 5.2 | 630–631 | Figures 17 & 18 (p. 645) |
| **Paper Limitations & Theoretical Gaps** | Limitations | 631 | Section 6 Limitations |
| **Detailed Agent Setups & Hyperparameters** | Appendix A.1 | 634–635 | Table 3 |
| **Task Prompts (RegAgent, CIC-IoT, Evaluators)** | Appendix A.2–A.4| 636–640 | System Prompts in §A.2–A.4 |
| **Multi-Backbone Evaluations (GPT-4o, DeepSeek, Qwen)** | Appendix B.2 | 641–642 | Table 4, Figures 10–12 |
| **Hyperparameter Robustness Sweeps ($K, n, \beta, \text{period}$)**| Appendix B.3 | 642 | Table 5, Table 6 |
| **Error-Free Deletion & Size-Matched Controlled Tests** | Appendix B.4, B.6| 642–643 | Figure 13, Table 7 |
| **KDE Quality Curves & Ground-Truth Correctness Rates**| Appendix B.7 | 643–644 | Table 8, Figures 15 & 16 |

---
*Forensic analysis compiled by the Paper Forensics Agent. All facts, formulas, and numbers are strictly grounded in ACL 2026 Paper 27.*

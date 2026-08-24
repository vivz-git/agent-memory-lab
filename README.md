# Agent Memory Lab

Agent Memory Lab is a research-inspired LLM agent memory system that controls what an agent remembers, forgets, and retrieves.

## 1. Problem
Most autonomous agents use a "Naive Add-All" memory policy—blindly storing every interaction and experience into their vector database. This causes rapid bloat, degrades latency, increases token costs, and fills the context window with toxic or irrelevant information over long-running tasks.

## 2. What Was Built
I engineered a **Managed Memory Architecture** that sits between the LLM and its vector store. By applying strict write-discipline and actively pruning stale records, the system maintains a lean, highly relevant memory footprint without sacrificing the agent's task success rate.

## 3. Architecture
The system features three core components:
- **Strict Addition Gate**: Evaluates new experiences immediately after a task, injecting only highly novel and successful experiences into the database.
- **History-Based Deletion**: Continuously tracks the retrieval utility of stored records, evicting experiences that become stale or misaligned over time.
- **Adaptive Read Rejection (ARR)**: A pre-assembly firewall that filters out low-utility records *before* they are sent to the LLM's prompt.

## 4. Demo
*(Placeholder for Demo Screenshot / GIF)*

## 5. Tech Stack
- **Language**: Python 3.10+
- **LLM Integration**: Groq API (`openai/gpt-oss-120b`)
- **Testing**: PyTest (130/130 passing)
- **Vector Search**: Custom Numpy-based similarity kernels (Cosine, RBF)
- **UI**: Vanilla HTML/JS/CSS (Recruiter-friendly interactive dashboard)

## 6. Benchmark Results
Benchmarked across 800 live Groq executions in a synthetic continuous-learning environment (RegAgent).

| Condition | Success Rate | Final Memory Size |
| :--- | :--- | :--- |
| **A. Fixed (No-Memory)** | 53.0% | 20.0 |
| **B. Naive Add-All** | 52.5% | 120.0 |
| **C. Managed Memory** | 52.5% | 30.5 |
| **D. Managed + ARR** | 52.5% | 50.5 |

**Key Findings:**
- **~75% reduction in average memory size (120 &rarr; 30.5)** using Managed Memory compared to Naive Add-All.
- **Comparable task success:** Success Rate remained comparable across all conditions (centering around ~53%), demonstrating that the agent can operate effectively on a significantly leaner memory footprint.

*(Note: The experiment also tracked $r_{EF}$, the Experience-Following correlation, confirming the core thesis that agents bias their outputs toward retrieved context. Naive Add-All showed the highest correlation at +0.2288).*

## 7. Adaptive Read Rejection
The Adaptive Read Rejection firewall successfully intercepted **418 read rejections on average across two benchmark seeds**. It acted as a strict safety and efficiency filter, preventing misaligned contexts from polluting the prompt. 

*Note: Adaptive Read Rejection did not improve accuracy in this benchmark; its primary benefit here is prompt hygiene and cost-efficiency.*

## 8. Limitations
- **Zero-Shot Math Plateau**: The agent's Success Rate was hard-capped at ~53% due to the inherent mathematical limitations of the chosen LLM on the 6D regression dataset, not the memory architecture.
- **No General Proof**: This project does not prove a general result across all LLMs or real-world domains. The findings are specific to this synthetic environment and the Groq 120b model.

## 9. How to Run
```bash
# Clone the repository
git clone https://github.com/vivz-git/agent-memory-lab
cd agent-memory-lab

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your GROQ_API_KEY

# Run the primary recruiter-facing benchmark validation
python scripts/run_benchmark.py
```

## 10. Research Paper Inspiration
This project was inspired by the ACL 2026 paper: *"How Memory Management Impacts LLM Agents: An Empirical Study of Experience-Following Behavior"*. I implemented the core agent loop, memory policies, and metrics from scratch, extending the research with the novel Adaptive Read Rejection mechanism.

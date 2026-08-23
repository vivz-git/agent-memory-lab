# Developer Setup & Engineering Guide

**Project**: How Memory Management Impacts LLM Agents: An Empirical Study of Experience-Following Behavior (ACL 2026 Reproduction)  
**Target Audience**: Developers, Subagent Engineers, and Infrastructure Contributors  

---

## 1. System Requirements & Prerequisites

- **Operating System**: Linux (Ubuntu 22.04+ recommended), macOS, or Windows 10/11 (PowerShell / WSL2)
- **Python Version**: `>= 3.10, < 3.12` (Python 3.10.x pinned in production containers)
- **Container Runtime**: Docker Engine 20.10+ & Docker Compose v2+
- **OpenAI API Key**: Required for live LLM agent runs and coarse LLM evaluators (optional for synthetic `reg_agent` dry runs)

---

## 2. Quick Local Environment Setup

### 2.1 Clone and Navigate to Worktree
```bash
git clone https://github.com/yuplin2333/agent_memory_manage.git
cd agent_memory_manage
```

### 2.2 Create and Activate Virtual Environment
```bash
# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate

# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 2.3 Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

### 2.4 Run Automated Environment Verification
```bash
python scripts/setup_dev.py
```
This utility automatically:
- Checks Python version compatibility ($\ge 3.10$).
- Verifies critical dependency imports (`numpy`, `scipy`, `pandas`, `sklearn`, `pydantic`, `openai`, `matplotlib`, `seaborn`, `pytest`).
- Generates `.env` from `.env.example` if absent.
- Creates runtime artifact directories (`results/`, `docs/`, `tests/`).

---

## 3. Environment Variables Configuration

Copy `.env.example` to `.env` and fill in your API credentials:

```bash
cp .env.example .env
```

### Configuration Variables Reference:
| Variable | Default Value | Description |
| :--- | :--- | :--- |
| `OPENAI_API_KEY` | *(None)* | OpenAI API key for agent backbone and evaluators |
| `OPENAI_API_BASE` | `https://api.openai.com/v1` | Custom OpenAI compatible base URL |
| `MODEL_BACKBONE` | `gpt-4o-mini` | LLM model backbone for agent decision making |
| `MODEL_TEMPERATURE`| `0.0` | Sampling temperature ($0.0$ for deterministic greedy runs) |
| `EVALUATOR_MODEL` | `gpt-4o-mini` | LLM judge for coarse trajectory evaluation |
| `LOG_LEVEL` | `INFO` | Logging verbosity (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |
| `RESULTS_DIR` | `./results` | Destination path for serialized metrics and plots |
| `RANDOM_SEED` | `42` | Global random seed for reproducible sampling |
| `BENCHMARK_ENV` | `reg_agent` | Benchmark environment (`reg_agent` or `ciciot`) |
| `NUM_STEPS` | `1000` | Horizon length for stream execution |
| `INITIAL_MEMORY_SIZE`| `100` | Verified baseline demonstration count ($N_0$) |
| `TOP_K` | `6` | Number of demonstration exemplars retrieved per step |
| `MEMORY_CAPACITY` | `100` | Max capacity upper bound ($M_{\max}$) for Protocol D |

---

## 4. Code Quality, Formatting & Testing

### 4.1 Linting and Formatting with Ruff
```bash
# Run linter
ruff check .

# Automatically apply safe fixes
ruff check --fix .

# Check code formatting
ruff format --check .

# Format code
ruff format .
```

### 4.2 Running the Test Suite
```bash
# Run all tests with verbose output
pytest -v

# Run infrastructure verification tests only
pytest tests/test_infra.py -v

# Run with test coverage report
pytest --cov=scripts --cov=src --cov-report=term-missing
```

---

## 5. Docker Development & Testing

### 5.1 Build the Multi-Stage Container
```bash
docker build -t agent-memory-benchmark:latest .
```

### 5.2 Run Containerized Test Suite
```bash
docker compose run --rm test
```

### 5.3 Run Containerized Benchmark Protocols
```bash
# Run Protocol A inside Docker
docker compose run --rm protocol-a

# Run Full Reproduction Suite inside Docker
docker compose run --rm reproduction
```

---

## 6. Architecture & Ownership Boundaries

To maintain clean separation of concerns and avoid merge conflicts across subagents:

```text
agent_memory_manage/
├── Dockerfile                  # [Infrastructure] Multi-stage container runtime
├── docker-compose.yml          # [Infrastructure] Service orchestration
├── requirements.txt            # [Infrastructure] Pinned dependency specifications
├── pyproject.toml              # [Infrastructure] PEP 517/621 build & tool configs
├── .env.example                # [Infrastructure] Environment variable template
├── .github/workflows/ci.yml    # [Infrastructure] GitHub Actions CI pipeline
├── scripts/                    # [Infrastructure] Execution & setup CLI entrypoints
│   ├── run_reproduction.py
│   └── setup_dev.py
├── docs/                       # [Infrastructure & Science] Documentation
│   ├── DEVELOPMENT.md
│   └── REPRODUCIBILITY.md
├── tests/
│   ├── test_infra.py           # [Infrastructure] Infrastructure & CLI tests
│   ├── test_memory.py          # [Core Team] Memory core unit tests
│   └── test_environments.py    # [Core Team] Environment unit tests
└── src/                        # [Core Engineering Team]
    ├── agent/                  # Base policy & prompt templates
    ├── memory/                 # Episodic bank, addition & deletion policies
    ├── environments/           # RegAgent & CIC-IoT task environments
    ├── evaluation/             # Metrics & strict/coarse evaluators
    └── utils/                  # Logging, clustering & shift generators
```

---

## 7. Continuous Integration (CI) Pipeline

Every pull request and push to `main` triggers automated CI via GitHub Actions (`.github/workflows/ci.yml`):
1. **Matrix Validation**: Tested against Python 3.10 and 3.11.
2. **Lint & Code Style**: Enforces Ruff rules and formatting.
3. **Unit & Integration Tests**: Executes `pytest` with coverage tracking.
4. **CLI Contract Verification**: Executes all reproduction protocol flags in `--dry-run` mode.
5. **Docker Build Check**: Verifies that the multi-stage Docker build succeeds without warnings.

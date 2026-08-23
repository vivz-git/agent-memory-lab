---
name: docker-patterns
description: >-
  Docker and Docker Compose patterns for containerized AI agent workflows, reproducible research environments,
  multi-stage builds, security hardening, non-root user execution, and multi-service orchestration (Vector DBs, Redis, Evaluators).
  Use when creating Dockerfiles, docker-compose.yml files, containerizing experiments, or debugging container issues.
---

# Docker & Containerization Patterns

Production and research containerization standards for AI agent memory systems.

## When to Activate

- Creating or optimizing `Dockerfile` or `docker-compose.yml`
- Setting up reproducible research and benchmarking environments
- Configuring vector databases (Chroma, Qdrant, Milvus) and caching layers (Redis) in containers
- Hardening container security (non-root execution, vulnerability minimization)
- Debugging container networking, volume mounts, or layer caching

---

## 1. Multi-Stage Dockerfile for Python AI Workflows

Always use multi-stage builds to separate build dependencies (compilers, wheel build tools) from the lightweight runtime image.

```dockerfile
# ---------------------------------------------------------
# Stage 1: Build & Dependency Resolution
# ---------------------------------------------------------
FROM python:3.10-slim AS builder

WORKDIR /build

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --user --no-warn-script-location -r requirements.txt

# ---------------------------------------------------------
# Stage 2: Runtime Image (Minimal & Secure)
# ---------------------------------------------------------
FROM python:3.10-slim AS runtime

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/home/appuser/.local/bin:$PATH"

# Create non-root user
RUN groupadd -g 1001 appgroup && \
    useradd -u 1001 -g appgroup -m -s /bin/bash appuser

# Copy installed Python dependencies from builder
COPY --from=builder --chown=appuser:appgroup /root/.local /home/appuser/.local

# Copy application source code
COPY --chown=appuser:appgroup . /app

# Switch to non-root user
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

ENTRYPOINT ["python", "-m", "src.agent.main"]
```

---

## 2. Docker Compose for Multi-Service Research Stack

Use `docker-compose.yml` to orchestrate agent runtime, vector search engine, and evaluation harnesses.

```yaml
version: "3.9"

services:
  agent-app:
    build:
      context: .
      target: runtime
    container_name: agent_memory_lab
    restart: unless-stopped
    environment:
      - ENVIRONMENT=production
      - VECTOR_DB_HOST=vector-db
      - VECTOR_DB_PORT=6333
      - REDIS_URL=redis://cache:6379/0
    volumes:
      - ./data:/app/data
      - ./benchmarks:/app/benchmarks
    depends_on:
      vector-db:
        condition: service_healthy
      cache:
        condition: service_healthy
    networks:
      - agent-net

  vector-db:
    image: qdrant/qdrant:v1.9.0
    container_name: agent_vector_store
    restart: unless-stopped
    ports:
      - "6333:6333"
    volumes:
      - qdrant_storage:/qdrant/storage
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:6333/healthz"]
      interval: 10s
      timeout: 5s
      retries: 3
    networks:
      - agent-net

  cache:
    image: redis:7.2-alpine
    container_name: agent_cache
    restart: unless-stopped
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 3
    networks:
      - agent-net

volumes:
  qdrant_storage:
  redis_data:

networks:
  agent-net:
    driver: bridge
```

---

## 3. Docker Security Checklist

- [x] **Never run containers as `root`** — define non-root `appuser`.
- [x] **Pin base image versions** — use explicit tags (e.g., `python:3.10-slim`, not `python:latest`).
- [x] **No Secrets in Images** — do not embed API keys or `.env` files into Docker layers.
- [x] **Use `.dockerignore`** — ignore `.git`, `__pycache__`, `.env`, test logs, and virtual environments.
- [x] **Read-only Filesystems** where possible to prevent runtime tampering.

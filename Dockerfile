# ==============================================================================
# Multi-stage Dockerfile for Agent Memory Management Empirical Benchmark
# ==============================================================================

# ------------------------------------------------------------------------------
# Stage 1: Build stage (compile dependencies into virtualenv)
# ------------------------------------------------------------------------------
FROM python:3.10-slim AS builder

WORKDIR /build

# Install minimal build tools for C extensions if required
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies with layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt


# ------------------------------------------------------------------------------
# Stage 2: Final runtime image
# ------------------------------------------------------------------------------
FROM python:3.10-slim AS runner

# Set runtime environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    RESULTS_DIR="/app/results"

# Create non-root system group and user
RUN groupadd -g 10001 appgroup && \
    useradd -u 10001 -g appgroup -s /bin/bash -m appuser

# Set working directory
WORKDIR /app

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv

# Copy repository source code and assign non-root ownership
COPY --chown=appuser:appgroup . /app

# Ensure results directory exists with proper permissions
RUN mkdir -p /app/results && chown -R appuser:appgroup /app/results

# Switch to non-root user
USER appuser

# Expose entrypoint for CLI reproduction runner
ENTRYPOINT ["python", "scripts/run_reproduction.py"]
CMD ["--help"]

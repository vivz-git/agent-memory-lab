---
name: python-testing
description: >-
  Comprehensive Python testing strategies using pytest, TDD (Test-Driven Development), async test fixtures,
  mocking strategies for LLMs and Vector DBs (respx, unittest.mock), parametrization, and coverage enforcement.
  Use when writing, running, or designing tests for Python applications and agent components.
---

# Python Testing & TDD Patterns

Test-driven development, unit testing, integration testing, and mocking patterns for AI agent memory systems.

## When to Activate

- Writing new features, memory backends, or agent tools (TDD: Red-Green-Refactor)
- Designing test suites, fixtures, and mock environments
- Testing asynchronous agent pipelines and memory retrieval
- Mocking external LLM APIs (Anthropic, OpenAI, local models) and Vector DBs
- Enforcing test coverage and regression prevention

---

## 1. Test-Driven Development (TDD) Workflow

1. **RED**: Write a failing unit test asserting the exact expected behavior and contract.
2. **GREEN**: Write minimal production code to satisfy the test.
3. **REFACTOR**: Clean up code structure, type annotations, and performance while keeping tests passing.

---

## 2. Directory Structure

```text
tests/
├── conftest.py              # Shared fixtures (event loop, mock clients, test data)
├── unit/
│   ├── test_memory_core.py  # Fast unit tests (no network, purely in-memory)
│   ├── test_schemas.py      # Pydantic validation tests
│   └── test_retrieval.py    # Algorithmic ranking & scoring tests
├── integration/
│   ├── test_vector_store.py # Tests against local/containerized DB
│   └── test_agent_loop.py   # Multi-step agent integration tests
└── eval/
    └── test_benchmarks.py   # Regression and accuracy threshold tests
```

---

## 3. Pytest Async Fixtures & Mocking

Always mock external LLM and vector service calls in unit tests to ensure fast, deterministic test runs.

```python
import pytest
from unittest.mock import AsyncMock, patch
from src.memory.core import MemoryManager, MemoryRecord

@pytest.fixture
def mock_embedding_client():
    client = AsyncMock()
    # Mock a deterministic 1536-dim vector
    client.embed_query = AsyncMock(return_value=[0.1] * 1536)
    client.embed_documents = AsyncMock(return_value=[[0.1] * 1536])
    return client

@pytest.fixture
def sample_records() -> list[MemoryRecord]:
    return [
        MemoryRecord(id="rec-1", content="User prefers dark mode", metadata={"type": "preference"}),
        MemoryRecord(id="rec-2", content="Project deadline is Friday", metadata={"type": "task"}),
    ]

@pytest.mark.asyncio
async def test_memory_retrieval_returns_relevant_records(mock_embedding_client, sample_records):
    # Arrange
    manager = MemoryManager(embedding_client=mock_embedding_client)
    for record in sample_records:
        await manager.store(record)

    # Act
    results = await manager.retrieve("user settings", top_k=1)

    # Assert
    assert len(results) == 1
    assert results[0].id == "rec-1"
    assert "dark mode" in results[0].content
    mock_embedding_client.embed_query.assert_awaited_once_with("user settings")
```

---

## 4. Parametrized Edge Case Testing

Test boundary conditions, empty inputs, token overflows, and invalid inputs:

```python
import pytest
from pydantic import ValidationError
from src.memory.schemas import MemoryConfig

@pytest.mark.parametrize("top_k", [1, 5, 10, 100])
def test_valid_top_k(top_k: int):
    config = MemoryConfig(top_k=top_k)
    assert config.top_k == top_k

@pytest.mark.parametrize("invalid_top_k", [0, -1, -50])
def test_invalid_top_k_raises_validation_error(invalid_top_k: int):
    with pytest.raises(ValidationError):
        MemoryConfig(top_k=invalid_top_k)
```

---

## 5. Execution Commands

```bash
# Run all tests with coverage report
pytest --cov=src --cov-report=term-missing tests/

# Run fast unit tests only
pytest tests/unit/

# Run integration tests
pytest tests/integration/

# Fail fast on first error
pytest -x
```

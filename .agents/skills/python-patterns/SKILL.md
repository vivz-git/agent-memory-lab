---
name: python-patterns
description: >-
  Provides production-grade Python idioms, PEP 8 standards, modern type hinting (Python 3.10+),
  Pydantic v2 patterns, async concurrency, and clean architecture specifically suited for AI agent memory systems.
  Use when writing, refactoring, or reviewing Python code in the repository.
---

# Python Development Patterns

Production-grade Python patterns and architecture guidelines tailored for AI agent systems and memory research engineering.

## When to Activate

- Designing or implementing new Python modules, classes, and pipelines
- Structuring memory systems, agent abstractions, and evaluation tools
- Adding type annotations, Pydantic models, or async workflows
- Refactoring existing code for performance, modularity, and maintainability
- Reviewing code for Pythonic idioms and PEP 8 compliance

---

## 1. Type Hints & Modern Typing (Python 3.10+)

Use strict type annotations across all public and internal interfaces.

### Guidelines
- Use built-in generics (`list[str]`, `dict[str, Any]`, `tuple[int, ...]`) instead of `typing.List`, `typing.Dict`.
- Use the `|` operator for unions (`str | None`) instead of `Optional[str]` or `Union[str, None]`.
- Use `typing.Protocol` for structural subtyping (duck typing with strict interfaces).
- Use `typing.Self` for fluent builders and method chaining.

```python
from typing import Protocol, TypeVar, Self
from dataclasses import dataclass
from abc import abstractmethod

T = TypeVar("T")

class MemoryStore(Protocol):
    """Structural interface for pluggable memory stores."""
    async def add(self, key: str, value: dict[str, str]) -> bool: ...
    async def retrieve(self, query: str, top_k: int = 5) -> list[dict[str, str]]: ...
    async def clear(self) -> None: ...

@dataclass(frozen=True, slots=True)
class MemoryRecord:
    id: str
    content: str
    metadata: dict[str, str]

    def with_metadata(self, key: str, value: str) -> Self:
        updated = {**self.metadata, key: value}
        return type(self)(id=self.id, content=self.content, metadata=updated)
```

---

## 2. Data Validation with Pydantic v2

All configuration, agent state schemas, and external API payloads must use Pydantic v2 `BaseModel`.

```python
from pydantic import BaseModel, Field, ConfigDict, field_validator

class MemoryConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    backend: str = Field(default="vector", description="Storage backend type")
    dimension: int = Field(default=1536, gt=0, description="Embedding vector dimension")
    similarity_threshold: float = Field(default=0.75, ge=0.0, le=1.0)
    top_k: int = Field(default=5, gt=0)

    @field_validator("backend")
    @classmethod
    def validate_backend(cls, v: str) -> str:
        valid_backends = {"vector", "graph", "hybrid", "kv"}
        if v.lower() not in valid_backends:
            raise ValueError(f"Invalid backend: {v}. Must be one of {valid_backends}")
        return v.lower()
```

---

## 3. Asynchronous Concurrency & Resource Safety

Agent systems execute IO-bound workloads (LLM calls, database queries, vector searches).

### Rules
1. Never block the event loop with synchronous sleep or IO; use `asyncio.sleep()`, `httpx.AsyncClient()`, or async database drivers.
2. Always manage concurrency with `asyncio.Semaphore` or `asyncio.TaskGroup`.
3. Ensure resource cleanup using `asynccontextmanager`.

```python
import asyncio
from contextlib import asynccontextmanager
from typing import AsyncIterator

class AsyncAgentClient:
    def __init__(self, concurrency_limit: int = 10):
        self._semaphore = asyncio.Semaphore(concurrency_limit)

    @asynccontextmanager
    async def session(self) -> AsyncIterator["AsyncAgentClient"]:
        try:
            yield self
        finally:
            await self.cleanup()

    async def query_memory_batch(self, queries: list[str]) -> list[list[str]]:
        async with asyncio.TaskGroup() as tg:
            tasks = [
                tg.create_task(self._safe_query(q))
                for q in queries
            ]
        return [t.result() for t in tasks]

    async def _safe_query(self, query: str) -> list[str]:
        async with self._semaphore:
            # Simulate async retrieval
            await asyncio.sleep(0.01)
            return [f"result for {query}"]

    async def cleanup(self) -> None:
        pass
```

---

## 4. Error Handling & Custom Domain Exceptions

- Create an explicit exception hierarchy inheriting from a base domain exception.
- Avoid catching generic `Exception` without re-raising or logging with traceback.
- Never use bare `except:`.

```python
class MemoryLabError(Exception):
    """Base exception for all Agent Memory Lab errors."""

class RetrievalError(MemoryLabError):
    """Raised when memory retrieval fails."""

class SchemaValidationError(MemoryLabError):
    """Raised when memory record schema fails validation."""
```

---

## 5. Clean Architecture for Memory Systems

Keep domain models, storage interfaces, and LLM integrations decoupled:
- **Domain Layer (`src/memory/core.py`)**: Entities, dataclasses, protocols, business logic.
- **Infrastructure Layer (`src/memory/backends/`)**: Vector DB adapters, SQLite/Redis adapters.
- **Application Layer (`src/agent/`)**: Memory coordination, agent decision loop.
- **Evaluation Layer (`src/evaluation/`)**: Benchmark runners and metrics.

"""Pydantic v2 schemas and enums for episodic experience-memory subsystem."""

from __future__ import annotations

from enum import Enum
from typing import Any
import uuid

from pydantic import BaseModel, ConfigDict, Field


class SimilarityMetricType(str, Enum):
    """Supported similarity metrics for memory retrieval."""

    COSINE = "cosine"
    RBF = "rbf"
    RELATIVE_FEATURE = "relative_feature"
    EUCLIDEAN = "euclidean"


class AdditionPolicyType(str, Enum):
    """Memory admission / addition policy types."""

    FIXED = "fixed"
    ADD_ALL = "add_all"
    COARSE = "coarse"
    STRICT = "strict"


class DeletionPolicyType(str, Enum):
    """Memory eviction / deletion policy types."""

    NONE = "none"
    PERIODIC = "periodic"
    HISTORY = "history"
    COMBINED = "combined"
    CONSTRAINED_CAPACITY = "constrained_capacity"


class ExperienceRecord(BaseModel):
    """Represents a single episodic experience record stored in the memory bank."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    query_key: list[float] | str
    query_text: str | None = None
    trajectory_text: str
    raw_output: Any = None
    retrieval_count: int = 0
    utility_history: list[float] = Field(default_factory=list)
    retrieval_steps: list[int] = Field(default_factory=list)
    mean_utility: float = 0.0
    entry_step: int = 0
    last_retrieved_step: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)

    def record_retrieval(self, step: int, utility: float | None = None) -> None:
        """Record that this memory entry was retrieved at execution step `step`.

        Args:
            step: The execution step/timestamp when retrieval occurred.
            utility: Optional downstream utility score earned on the task that retrieved it.
        """
        self.retrieval_count += 1
        self.retrieval_steps.append(int(step))
        self.last_retrieved_step = int(step)
        if utility is not None:
            self.update_utility(utility, step=step)

    def update_utility(self, utility_score: float, step: int | None = None) -> None:
        """Append a downstream task utility score and update mean utility.

        Args:
            utility_score: Downstream utility score (e.g. task reward, SR, accuracy).
            step: Optional step when utility was recorded.
        """
        self.utility_history.append(float(utility_score))
        if self.utility_history:
            self.mean_utility = float(sum(self.utility_history) / len(self.utility_history))
        if step is not None:
            self.last_retrieved_step = int(step)

    def record_utility(self, utility_score: float, step: int | None = None) -> None:
        """Convenience alias for update_utility."""
        self.update_utility(utility_score, step=step)

    def retrievals_in_window(self, start_step: int, end_step: int) -> int:
        """Count the number of retrievals that occurred within the window [start_step, end_step].

        Args:
            start_step: Window start step (inclusive).
            end_step: Window end step (inclusive).

        Returns:
            Number of retrieval events in the window.
        """
        return sum(1 for s in self.retrieval_steps if start_step <= s <= end_step)

    def to_demonstration_text(self) -> str:
        """Format experience as a demonstration string for in-context prompting."""
        if self.query_text:
            return f"Input: {self.query_text}\nTrajectory: {self.trajectory_text}"
        return f"Input: {self.query_key}\nTrajectory: {self.trajectory_text}"


class MemoryQuery(BaseModel):
    """Structured query object for memory retrieval."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    key: list[float] | str | Any
    query_text: str | None = None
    top_k: int = 1
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievalResult(BaseModel):
    """Retrieval outcome encapsulating the retrieved record and its similarity score."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    record: ExperienceRecord
    score: float
    rank: int = 0

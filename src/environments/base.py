"""Base environment interfaces, query structures, and task results."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
import numpy as np
from pydantic import BaseModel, Field, ConfigDict


class TaskQuery(BaseModel):
    """Represents an input task query presented to an agent."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    query_id: str
    query_vector: List[float] = Field(
        default_factory=list,
        description="Vector representation of the query for similarity retrieval.",
    )
    features: Dict[str, Any] = Field(
        default_factory=dict,
        description="Structured key-value feature dictionary.",
    )
    raw_input: Any = Field(
        default=None,
        description="Raw domain-specific input data (e.g. 6D array, dict, text).",
    )
    ground_truth: Any = Field(
        default=None,
        description="True target or label for the query.",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional contextual metadata (e.g. cluster, split, mu).",
    )

    def get_numpy_vector(self) -> np.ndarray:
        """Return the query vector as a 1D numpy array."""
        return np.asarray(self.query_vector, dtype=np.float64)


class TaskResult(BaseModel):
    """Represents the evaluation outcome of an agent's execution on a query."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    query_id: str
    prediction: Any = Field(
        default=None,
        description="Parsed prediction from the agent.",
    )
    ground_truth: Any = Field(
        default=None,
        description="Ground truth target value or label.",
    )
    raw_output: str = Field(
        default="",
        description="Full raw text trajectory output from the LLM.",
    )
    is_success: bool = Field(
        default=False,
        description="Whether the prediction met the environment success criteria.",
    )
    score: float = Field(
        default=0.0,
        description="Downstream utility score Phi(q, e), typically in [0.0, 1.0].",
    )
    error: float = Field(
        default=0.0,
        description="Scalar error metric (e.g. absolute difference or 0/1 loss).",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Domain-specific evaluation details.",
    )


class BaseEnvironment(ABC):
    """Abstract base class for all agent environments."""

    @abstractmethod
    def sample_initial_memory(
        self, n_samples: int = 100, seed: Optional[int] = None
    ) -> List[TaskQuery]:
        """Generate verified seed demonstration queries D_0."""
        pass

    @abstractmethod
    def sample_stream(
        self, n_samples: int = 1000, seed: Optional[int] = None
    ) -> List[TaskQuery]:
        """Generate a sequential stream of test queries."""
        pass

    @abstractmethod
    def evaluate(
        self, query: TaskQuery, prediction: Any, raw_output: str = ""
    ) -> TaskResult:
        """Evaluate agent prediction against query ground truth."""
        pass

    @abstractmethod
    def compute_input_similarity(
        self,
        vec_a: Union[TaskQuery, np.ndarray, List[float]],
        vec_b: Union[TaskQuery, np.ndarray, List[float]],
    ) -> float:
        """Compute input similarity S_in between two queries or feature vectors."""
        pass

    @abstractmethod
    def compute_output_similarity(self, out_a: Any, out_b: Any) -> float:
        """Compute output similarity S_out between two execution outputs."""
        pass

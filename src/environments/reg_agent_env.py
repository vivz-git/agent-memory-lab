"""6D Synthetic Gaussian Regression Environment (RegAgent)."""

from __future__ import annotations

import re
from typing import Any, List, Optional, Sequence, Union
import numpy as np

from src.environments.base import BaseEnvironment, TaskQuery, TaskResult

DEFAULT_W = np.array([0.8, -0.5, 1.2, -1.0, 0.4, -0.7], dtype=np.float64)
VALID_MUS = (-0.5, 0.0, 0.5)


class RegAgentEnvironment(BaseEnvironment):
    """6D Synthetic Linear Regression Environment with bounded noise.

    Attributes:
        w: Implicit 6D ground-truth weight vector.
        gamma: RBF scaling factor for output similarity computation (default=1.0).
        noise_bound: Half-width for uniform noise interval [-noise_bound, noise_bound].
        success_threshold: Maximum absolute error |y_hat - y| for task success (default=1.0).
    """

    def __init__(
        self,
        w: Optional[Sequence[float]] = None,
        gamma: float = 1.0,
        noise_bound: float = 1.0,
        success_threshold: float = 1.0,
        dim: int = 6,
    ) -> None:
        if w is not None:
            self.w = np.asarray(w, dtype=np.float64)
            if self.w.shape[0] != dim:
                raise ValueError(f"Weight vector must be {dim}-dimensional, got {self.w.shape[0]}")
        else:
            self.w = DEFAULT_W.copy()

        self.dim = dim
        self.gamma = gamma
        self.noise_bound = noise_bound
        self.success_threshold = success_threshold

    def generate_single_query(
        self,
        query_id: str,
        mu: Optional[float] = None,
        rng: Optional[np.random.RandomState] = None,
        include_noise: bool = True,
    ) -> TaskQuery:
        """Generate a single 6D regression query."""
        if rng is None:
            rng = np.random.RandomState()

        if mu is None:
            mu = float(rng.choice(VALID_MUS))

        # Sample x ~ N(mu * 1_6, I_6)
        x = rng.normal(loc=mu, scale=1.0, size=self.dim)

        # Compute y = w^T x + epsilon
        noise = float(rng.uniform(-self.noise_bound, self.noise_bound)) if include_noise else 0.0
        y = float(np.dot(self.w, x) + noise)

        return TaskQuery(
            query_id=query_id,
            query_vector=x.tolist(),
            raw_input=x.tolist(),
            ground_truth=y,
            features={f"x_{i}": float(val) for i, val in enumerate(x)},
            metadata={
                "mu": mu,
                "noise": noise,
                "clean_y": float(np.dot(self.w, x)),
            },
        )

    def sample_initial_memory(
        self, n_samples: int = 100, seed: Optional[int] = 42
    ) -> List[TaskQuery]:
        """Generate verified initial demonstration memory bank D_0 (N=100)."""
        rng = np.random.RandomState(seed)
        queries: List[TaskQuery] = []
        for i in range(n_samples):
            qid = f"init_reg_{i:04d}"
            queries.append(self.generate_single_query(qid, rng=rng, include_noise=True))
        return queries

    def sample_stream(
        self,
        n_samples: int = 1000,
        seed: Optional[int] = 128,
        cluster_shift: bool = False,
    ) -> List[TaskQuery]:
        """Generate test stream queries.

        Args:
            n_samples: Total number of stream steps (e.g. 1000 to 4000).
            seed: RNG seed for reproducible streaming.
            cluster_shift: If True, sequences stream sequentially across mu=-0.5 -> 0.0 -> 0.5.
        """
        rng = np.random.RandomState(seed)
        queries: List[TaskQuery] = []

        if cluster_shift:
            mus = list(VALID_MUS)
            n_per_cluster = n_samples // len(mus)
            remainder = n_samples % len(mus)
            counts = [n_per_cluster] * len(mus)
            for j in range(remainder):
                counts[j] += 1

            step_idx = 0
            for mu_val, count in zip(mus, counts):
                for _ in range(count):
                    qid = f"stream_reg_{step_idx:05d}"
                    queries.append(
                        self.generate_single_query(
                            qid, mu=mu_val, rng=rng, include_noise=True
                        )
                    )
                    step_idx += 1
        else:
            for i in range(n_samples):
                qid = f"stream_reg_{i:05d}"
                queries.append(self.generate_single_query(qid, rng=rng, include_noise=True))

        return queries

    def evaluate(
        self, query: TaskQuery, prediction: Any, raw_output: str = ""
    ) -> TaskResult:
        """Evaluate scalar prediction against ground truth target y."""
        gt = float(query.ground_truth)
        pred_val: Optional[float] = None

        if isinstance(prediction, (int, float, np.floating, np.integer)):
            pred_val = float(prediction)
        elif isinstance(prediction, str):
            match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", prediction)
            if match:
                try:
                    pred_val = float(match.group(0))
                except ValueError:
                    pred_val = None

        if pred_val is None:
            return TaskResult(
                query_id=query.query_id,
                prediction=None,
                ground_truth=gt,
                raw_output=raw_output,
                is_success=False,
                score=0.0,
                error=float("inf"),
                metadata={"error_reason": "unparseable_prediction"},
            )

        abs_error = abs(pred_val - gt)
        is_success = bool(abs_error <= self.success_threshold)
        score = 1.0 if is_success else 0.0

        return TaskResult(
            query_id=query.query_id,
            prediction=pred_val,
            ground_truth=gt,
            raw_output=raw_output,
            is_success=is_success,
            score=score,
            error=float(abs_error),
            metadata={"threshold": self.success_threshold},
        )

    def compute_input_similarity(
        self,
        vec_a: Union[TaskQuery, np.ndarray, List[float]],
        vec_b: Union[TaskQuery, np.ndarray, List[float]],
    ) -> float:
        """Compute cosine similarity between two 6D query vectors."""
        a = vec_a.get_numpy_vector() if isinstance(vec_a, TaskQuery) else np.asarray(vec_a, dtype=np.float64)
        b = vec_b.get_numpy_vector() if isinstance(vec_b, TaskQuery) else np.asarray(vec_b, dtype=np.float64)

        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        if norm_a < 1e-12 or norm_b < 1e-12:
            return 0.0

        cos_sim = float(np.dot(a, b) / (norm_a * norm_b))
        return float(np.clip(cos_sim, -1.0, 1.0))

    def compute_output_similarity(self, out_a: Any, out_b: Any) -> float:
        """Compute RBF kernel output similarity exp(-gamma * |y1 - y2|^2)."""
        val_a = self._extract_scalar(out_a)
        val_b = self._extract_scalar(out_b)

        if val_a is None or val_b is None:
            return 0.0

        diff_sq = (val_a - val_b) ** 2
        sim = float(np.exp(-self.gamma * diff_sq))
        return float(np.clip(sim, 0.0, 1.0))

    @staticmethod
    def _extract_scalar(val: Any) -> Optional[float]:
        if isinstance(val, (int, float, np.floating, np.integer)):
            return float(val)
        if isinstance(val, str):
            match = re.search(r"boxed\{([^}]+)\}", val)
            target = match.group(1) if match else val
            num_match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", target)
            if num_match:
                try:
                    return float(num_match.group(0))
                except ValueError:
                    return None
        return None

"""Data Leakage & Contamination Prevention Safeguards.

Implements strict validation to guarantee split isolation between seed memory D_0
and test task streams S_test as specified in research/RESEARCH_SPEC.md & research/evaluation_plan.md:
- SHA-256 hash collision detection
- Exact query overlap identification
- Minimum embedding distance assertion: ||e(q_i) - e(q_j)||_2 > epsilon_min
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from typing import Any, Optional, Sequence, Union
import numpy as np


@dataclass
class LeakageViolation:
    """Details of an identified contamination or leakage instance."""
    violation_type: str  # 'hash_collision', 'exact_overlap', 'distance_violation'
    init_index: int
    test_index: int
    init_repr: str
    test_repr: str
    metric_value: Optional[float] = None
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "violation_type": self.violation_type,
            "init_index": self.init_index,
            "test_index": self.test_index,
            "init_repr": self.init_repr,
            "test_repr": self.test_repr,
            "metric_value": self.metric_value,
            "description": self.description,
        }


@dataclass
class LeakageReport:
    """Comprehensive summary of split isolation and contamination analysis."""
    is_clean: bool
    total_init_samples: int
    total_test_samples: int
    hash_collisions: int = 0
    exact_overlaps: int = 0
    distance_violations: int = 0
    violations: list[LeakageViolation] = field(default_factory=list)
    details: list[str] = field(default_factory=list)

    def summary(self) -> str:
        status = "PASSED (NO LEAKAGE)" if self.is_clean else "FAILED (LEAKAGE DETECTED)"
        lines = [
            f"=== LEAKAGE AUDIT REPORT: {status} ===",
            f"Initial Seed Partition (D_0) Size: {self.total_init_samples}",
            f"Test Stream Partition (S_test) Size: {self.total_test_samples}",
            f"SHA-256 Hash Collisions: {self.hash_collisions}",
            f"Exact Query Overlaps: {self.exact_overlaps}",
            f"Embedding Distance Violations: {self.distance_violations}",
            f"Total Violations: {len(self.violations)}",
        ]
        if self.details:
            lines.append("Violation Details (Top 10):")
            for det in self.details[:10]:
                lines.append(f"  - {det}")
            if len(self.details) > 10:
                lines.append(f"  ... and {len(self.details) - 10} more.")
        return "\n".join(lines)


class LeakageChecker:
    """Auditor for temporal split isolation between seed memory and test queries."""

    def __init__(
        self,
        min_embedding_distance: float = 1e-4,
        distance_metric: str = "euclidean",
    ):
        self.min_embedding_distance = min_embedding_distance
        self.distance_metric = distance_metric.lower()

    @staticmethod
    def compute_sha256(item: Any) -> str:
        """Compute deterministic SHA-256 hash of any input object."""
        if isinstance(item, (np.ndarray, list, tuple)):
            # If numeric vector, round to 6 decimal places for stable hashing
            try:
                arr = np.asarray(item, dtype=np.float64)
                content = np.round(arr, decimals=6).tobytes()
                return hashlib.sha256(content).hexdigest()
            except (ValueError, TypeError):
                pass

        if isinstance(item, dict):
            serialized = json.dumps(item, sort_keys=True, default=str)
        else:
            serialized = str(item).strip()

        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def check_hash_collisions(
        self,
        init_queries: Sequence[Any],
        test_queries: Sequence[Any],
    ) -> list[LeakageViolation]:
        """Detect SHA-256 hash collisions between initial seed and test queries."""
        violations: list[LeakageViolation] = []
        init_hashes = {self.compute_sha256(q): (idx, q) for idx, q in enumerate(init_queries)}

        for test_idx, test_q in enumerate(test_queries):
            test_hash = self.compute_sha256(test_q)
            if test_hash in init_hashes:
                init_idx, init_q = init_hashes[test_hash]
                violations.append(
                    LeakageViolation(
                        violation_type="hash_collision",
                        init_index=init_idx,
                        test_index=test_idx,
                        init_repr=str(init_q)[:100],
                        test_repr=str(test_q)[:100],
                        description=f"Exact SHA-256 collision ({test_hash[:12]}...)",
                    )
                )

        return violations

    def check_exact_overlaps(
        self,
        init_queries: Sequence[Any],
        test_queries: Sequence[Any],
    ) -> list[LeakageViolation]:
        """Detect exact value/string matches between initial seed and test queries."""
        violations: list[LeakageViolation] = []

        for test_idx, test_q in enumerate(test_queries):
            for init_idx, init_q in enumerate(init_queries):
                is_equal = False
                if isinstance(init_q, np.ndarray) and isinstance(test_q, np.ndarray):
                    if init_q.shape == test_q.shape and np.allclose(init_q, test_q, atol=1e-8):
                        is_equal = True
                elif init_q == test_q or str(init_q).strip() == str(test_q).strip():
                    is_equal = True

                if is_equal:
                    violations.append(
                        LeakageViolation(
                            violation_type="exact_overlap",
                            init_index=init_idx,
                            test_index=test_idx,
                            init_repr=str(init_q)[:100],
                            test_repr=str(test_q)[:100],
                            description="Verbatim query match between init and test set",
                        )
                    )

        return violations

    def check_embedding_distances(
        self,
        init_embeddings: np.ndarray,
        test_embeddings: np.ndarray,
        min_distance: Optional[float] = None,
    ) -> list[LeakageViolation]:
        """Verify that all test embeddings are strictly separated from seed embeddings."""
        violations: list[LeakageViolation] = []
        eps = min_distance if min_distance is not None else self.min_embedding_distance

        init_arr = np.asarray(init_embeddings, dtype=np.float64)
        test_arr = np.asarray(test_embeddings, dtype=np.float64)

        if init_arr.ndim == 1:
            init_arr = init_arr.reshape(1, -1)
        if test_arr.ndim == 1:
            test_arr = test_arr.reshape(1, -1)

        if self.distance_metric == "cosine":
            # Compute cosine distance = 1 - cosine_similarity
            init_norm = np.linalg.norm(init_arr, axis=1, keepdims=True)
            test_norm = np.linalg.norm(test_arr, axis=1, keepdims=True)
            init_norm = np.where(init_norm == 0, 1e-12, init_norm)
            test_norm = np.where(test_norm == 0, 1e-12, test_norm)

            init_u = init_arr / init_norm
            test_u = test_arr / test_norm
            # Sim matrix: (N_test, N_init)
            sim_matrix = np.matmul(test_u, init_u.T)
            dist_matrix = 1.0 - sim_matrix
        else:
            # Euclidean distance: ||test_i - init_j||_2
            # (N_test, 1, D) - (1, N_init, D) -> (N_test, N_init)
            diff = test_arr[:, np.newaxis, :] - init_arr[np.newaxis, :, :]
            dist_matrix = np.linalg.norm(diff, axis=-1)

        # Identify pairs with dist < eps
        test_indices, init_indices = np.where(dist_matrix < eps)
        for t_idx, i_idx in zip(test_indices, init_indices):
            dist_val = float(dist_matrix[t_idx, i_idx])
            violations.append(
                LeakageViolation(
                    violation_type="distance_violation",
                    init_index=int(i_idx),
                    test_index=int(t_idx),
                    init_repr=f"init_emb[{i_idx}]",
                    test_repr=f"test_emb[{t_idx}]",
                    metric_value=dist_val,
                    description=f"{self.distance_metric.capitalize()} distance {dist_val:.6f} < threshold {eps:.6f}",
                )
            )

        return violations

    def verify_split_isolation(
        self,
        init_queries: Sequence[Any],
        test_queries: Sequence[Any],
        init_embeddings: Optional[np.ndarray] = None,
        test_embeddings: Optional[np.ndarray] = None,
        min_distance: Optional[float] = None,
        raise_on_leakage: bool = False,
    ) -> LeakageReport:
        """Run complete end-to-end split isolation verification suite."""
        all_violations: list[LeakageViolation] = []
        details: list[str] = []

        # 1. Check SHA-256 hash collisions
        hash_violations = self.check_hash_collisions(init_queries, test_queries)
        all_violations.extend(hash_violations)
        for v in hash_violations:
            details.append(f"[Hash Collision] Init #{v.init_index} == Test #{v.test_index}: {v.description}")

        # 2. Check exact query overlaps
        overlap_violations = self.check_exact_overlaps(init_queries, test_queries)
        # Filter out duplicates already captured by hash check
        seen_pairs = {(v.init_index, v.test_index) for v in hash_violations}
        unique_overlaps = [v for v in overlap_violations if (v.init_index, v.test_index) not in seen_pairs]
        all_violations.extend(unique_overlaps)
        for v in unique_overlaps:
            details.append(f"[Exact Overlap] Init #{v.init_index} == Test #{v.test_index}")

        # 3. Check embedding distances if representations provided
        dist_violations: list[LeakageViolation] = []
        if init_embeddings is not None and test_embeddings is not None:
            dist_violations = self.check_embedding_distances(
                init_embeddings, test_embeddings, min_distance=min_distance
            )
            all_violations.extend(dist_violations)
            for v in dist_violations:
                details.append(f"[Distance Violation] Init #{v.init_index} vs Test #{v.test_index}: {v.description}")

        is_clean = len(all_violations) == 0
        report = LeakageReport(
            is_clean=is_clean,
            total_init_samples=len(init_queries),
            total_test_samples=len(test_queries),
            hash_collisions=len(hash_violations),
            exact_overlaps=len(overlap_violations),
            distance_violations=len(dist_violations),
            violations=all_violations,
            details=details,
        )

        if raise_on_leakage and not is_clean:
            raise AssertionError(f"Data leakage assertion failed:\n{report.summary()}")

        return report

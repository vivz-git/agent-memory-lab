"""Vectorized episodic memory bank implementation supporting multiple similarity metrics."""

from __future__ import annotations

from difflib import SequenceMatcher
from typing import Any
import numpy as np

from src.memory.schema import (
    ExperienceRecord,
    MemoryQuery,
    RetrievalResult,
    SimilarityMetricType,
)


class BaseMemoryBank:
    """Vectorized episodic memory bank supporting insertion, retrieval, deletion, and utility tracking.

    Supports Cosine Similarity, RBF Kernel distance, Relative Feature Difference (tabular),
    and Euclidean similarity metrics.
    """

    def __init__(
        self,
        metric: SimilarityMetricType | str = SimilarityMetricType.COSINE,
        rbf_gamma: float = 1.0,
        discrete_feature_indices: list[int] | set[int] | None = None,
        feature_min_max: dict[int, tuple[float, float]] | None = None,
    ) -> None:
        """Initialize the BaseMemoryBank.

        Args:
            metric: Similarity metric to use for retrieval.
            rbf_gamma: Gamma hyperparameter for RBF kernel similarity: exp(-gamma * ||v1 - v2||^2).
            discrete_feature_indices: Feature indices to treat as categorical/discrete in relative feature difference.
            feature_min_max: Optional feature normalization ranges {feature_idx: (min_val, max_val)}.
        """
        if isinstance(metric, str):
            metric = SimilarityMetricType(metric.lower())
        self.metric = metric
        self.rbf_gamma = float(rbf_gamma)
        self.discrete_feature_indices = set(discrete_feature_indices) if discrete_feature_indices else set()
        self.feature_min_max = feature_min_max or {}

        self._records: dict[str, ExperienceRecord] = {}
        self._key_matrix: np.ndarray | None = None
        self._id_order: list[str] = []
        self._dirty_index: bool = False
        self._is_vector_keys: bool = True

    def add(self, record: ExperienceRecord) -> None:
        """Add a single experience record to the memory bank.

        Args:
            record: The ExperienceRecord instance to store.
        """
        self._records[record.id] = record
        self._dirty_index = True

    def add_many(self, records: list[ExperienceRecord]) -> None:
        """Add multiple experience records to the memory bank.

        Args:
            records: List of ExperienceRecord instances to store.
        """
        for record in records:
            self._records[record.id] = record
        self._dirty_index = True

    def get(self, record_id: str) -> ExperienceRecord | None:
        """Retrieve a record by its unique ID.

        Args:
            record_id: Unique record identifier.

        Returns:
            The ExperienceRecord if found, otherwise None.
        """
        return self._records.get(record_id)

    def delete(self, record_id: str) -> bool:
        """Delete a record by its unique ID.

        Args:
            record_id: Unique record identifier.

        Returns:
            True if the record was found and deleted, False otherwise.
        """
        if record_id in self._records:
            del self._records[record_id]
            self._dirty_index = True
            return True
        return False

    def delete_many(self, record_ids: list[str]) -> int:
        """Delete multiple records by their unique IDs.

        Args:
            record_ids: Collection of record IDs to delete.

        Returns:
            Count of successfully deleted records.
        """
        count = 0
        for rid in record_ids:
            if rid in self._records:
                del self._records[rid]
                count += 1
        if count > 0:
            self._dirty_index = True
        return count

    def size(self) -> int:
        """Return the current number of records in the memory bank."""
        return len(self._records)

    def __len__(self) -> int:
        """Return the current number of records in the memory bank."""
        return self.size()

    def all_records(self) -> list[ExperienceRecord]:
        """Return a list of all experience records stored in the memory bank."""
        return list(self._records.values())

    def clear(self) -> None:
        """Clear all records from the memory bank."""
        self._records.clear()
        self._key_matrix = None
        self._id_order.clear()
        self._dirty_index = False

    def update_utility(self, record_id: str, utility_score: float, step: int | None = None) -> bool:
        """Update the downstream utility score of a specific memory record.

        Args:
            record_id: ID of the record to update.
            utility_score: Downstream utility score earned.
            step: Optional execution step.

        Returns:
            True if record was found and updated, False otherwise.
        """
        record = self._records.get(record_id)
        if record is not None:
            record.update_utility(utility_score, step=step)
            return True
        return False

    def _rebuild_index(self) -> None:
        """Rebuild cached numpy key matrix for vectorized retrieval."""
        self._id_order = list(self._records.keys())
        if not self._id_order:
            self._key_matrix = None
            self._dirty_index = False
            return

        sample_key = self._records[self._id_order[0]].query_key
        if isinstance(sample_key, (list, tuple, np.ndarray)):
            try:
                matrix = np.array(
                    [self._records[rid].query_key for rid in self._id_order],
                    dtype=np.float64,
                )
                self._key_matrix = matrix
                self._is_vector_keys = True
            except (ValueError, TypeError):
                self._key_matrix = None
                self._is_vector_keys = False
        else:
            self._key_matrix = None
            self._is_vector_keys = False

        self._dirty_index = False

    def compute_similarity(
        self,
        query_key: list[float] | np.ndarray | str,
        candidate_key: list[float] | np.ndarray | str,
    ) -> float:
        """Compute similarity between a query key and a single candidate key.

        Args:
            query_key: Query vector or string.
            candidate_key: Candidate memory key vector or string.

        Returns:
            Similarity score (higher indicates more similar).
        """
        if isinstance(query_key, str) or isinstance(candidate_key, str):
            q_str = str(query_key)
            c_str = str(candidate_key)
            if q_str == c_str:
                return 1.0
            return float(SequenceMatcher(None, q_str, c_str).ratio())

        q_vec = np.asarray(query_key, dtype=np.float64)
        c_vec = np.asarray(candidate_key, dtype=np.float64)

        if self.metric == SimilarityMetricType.COSINE:
            norm_q = np.linalg.norm(q_vec)
            norm_c = np.linalg.norm(c_vec)
            if norm_q == 0.0 and norm_c == 0.0:
                return 1.0
            if norm_q == 0.0 or norm_c == 0.0:
                return 0.0
            return float(np.dot(q_vec, c_vec) / (norm_q * norm_c))

        elif self.metric == SimilarityMetricType.RBF:
            sq_dist = np.sum((q_vec - c_vec) ** 2)
            return float(np.exp(-self.rbf_gamma * sq_dist))

        elif self.metric == SimilarityMetricType.RELATIVE_FEATURE:
            return self._compute_relative_feature_similarity_single(q_vec, c_vec)

        elif self.metric == SimilarityMetricType.EUCLIDEAN:
            dist = np.linalg.norm(q_vec - c_vec)
            return float(1.0 / (1.0 + dist))

        # Default fallback: cosine
        norm_q = np.linalg.norm(q_vec)
        norm_c = np.linalg.norm(c_vec)
        denom = norm_q * norm_c
        return float(np.dot(q_vec, c_vec) / denom) if denom > 1e-12 else 0.0

    def _compute_relative_feature_similarity_single(
        self,
        x1: np.ndarray,
        x2: np.ndarray,
    ) -> float:
        """Compute relative feature difference similarity for a single pair of vectors."""
        dim = len(x1)
        if dim == 0:
            return 1.0

        diff_sum = 0.0
        for i in range(dim):
            v1 = float(x1[i])
            v2 = float(x2[i])
            if i in self.discrete_feature_indices:
                # Discrete: 0 if equal else 1
                diff_sum += 0.0 if v1 == v2 else 1.0
            else:
                # Continuous: |x1 - x2| / max(|x1|, |x2|)
                abs_diff = abs(v1 - v2)
                max_val = max(abs(v1), abs(v2))
                if max_val < 1e-8:
                    diff_sum += 0.0
                else:
                    diff_sum += min(1.0, abs_diff / max_val)

        avg_diff = diff_sum / dim
        return float(max(0.0, 1.0 - avg_diff))

    def _compute_vectorized_similarities(self, q_vec: np.ndarray) -> np.ndarray:
        """Compute similarity scores against all indexed records using vectorized NumPy."""
        if self._key_matrix is None or len(self._key_matrix) == 0:
            return np.array([], dtype=np.float64)

        if self.metric == SimilarityMetricType.COSINE:
            norm_q = np.linalg.norm(q_vec)
            norm_matrix = np.linalg.norm(self._key_matrix, axis=1)

            dot_prods = np.dot(self._key_matrix, q_vec)
            denoms = norm_matrix * norm_q

            # Handle zero norms safely
            zero_denom_mask = denoms < 1e-12
            scores = np.zeros(len(self._key_matrix), dtype=np.float64)
            np.divide(dot_prods, denoms, out=scores, where=~zero_denom_mask)

            if norm_q == 0.0:
                scores[norm_matrix == 0.0] = 1.0

            return scores

        elif self.metric == SimilarityMetricType.RBF:
            diffs = self._key_matrix - q_vec
            sq_dists = np.sum(diffs ** 2, axis=1)
            return np.exp(-self.rbf_gamma * sq_dists)

        elif self.metric == SimilarityMetricType.RELATIVE_FEATURE:
            N, D = self._key_matrix.shape
            diffs = np.zeros((N, D), dtype=np.float64)

            for j in range(D):
                col = self._key_matrix[:, j]
                q_val = q_vec[j]
                if j in self.discrete_feature_indices:
                    diffs[:, j] = (col != q_val).astype(np.float64)
                else:
                    abs_diff = np.abs(col - q_val)
                    max_val = np.maximum(np.abs(col), np.abs(q_val))
                    zero_mask = max_val < 1e-8
                    col_diff = np.zeros(N, dtype=np.float64)
                    np.divide(abs_diff, max_val, out=col_diff, where=~zero_mask)
                    diffs[:, j] = np.clip(col_diff, 0.0, 1.0)

            avg_diffs = np.mean(diffs, axis=1)
            return np.maximum(0.0, 1.0 - avg_diffs)

        elif self.metric == SimilarityMetricType.EUCLIDEAN:
            dists = np.linalg.norm(self._key_matrix - q_vec, axis=1)
            return 1.0 / (1.0 + dists)

        # Fallback cosine
        dot_prods = np.dot(self._key_matrix, q_vec)
        norm_q = np.linalg.norm(q_vec)
        norm_matrix = np.linalg.norm(self._key_matrix, axis=1)
        denoms = norm_matrix * norm_q
        scores = np.zeros(len(self._key_matrix), dtype=np.float64)
        np.divide(dot_prods, denoms, out=scores, where=denoms > 1e-12)
        return scores

    def retrieve(
        self,
        query: list[float] | np.ndarray | str | MemoryQuery,
        top_k: int = 1,
        filter_ids: set[str] | None = None,
        min_score: float | None = None,
    ) -> list[RetrievalResult]:
        """Retrieve the top-K most similar experience records from the memory bank.

        Args:
            query: Query vector, text key, or MemoryQuery object.
            top_k: Number of nearest candidate records to retrieve.
            filter_ids: Optional set of record IDs to exclude from retrieval.
            min_score: Optional minimum similarity score threshold.

        Returns:
            List of RetrievalResult objects ordered descending by similarity score.
        """
        if top_k <= 0 or self.size() == 0:
            return []

        if isinstance(query, MemoryQuery):
            query_key = query.key
            top_k = query.top_k or top_k
        else:
            query_key = query

        if self._dirty_index or self._key_matrix is None:
            self._rebuild_index()

        filter_set = filter_ids or set()

        # Check if vectorized path can be used
        if self._is_vector_keys and not isinstance(query_key, str):
            q_vec = np.asarray(query_key, dtype=np.float64)
            scores = self._compute_vectorized_similarities(q_vec)

            # Pair with IDs and filter
            candidates: list[tuple[str, float]] = []
            for idx, rid in enumerate(self._id_order):
                if rid in filter_set:
                    continue
                score = float(scores[idx])
                if min_score is not None and score < min_score:
                    continue
                candidates.append((rid, score))

        else:
            # Iterative non-vectorized path (for string keys or mixed structures)
            candidates = []
            for rid, record in self._records.items():
                if rid in filter_set:
                    continue
                score = self.compute_similarity(query_key, record.query_key)
                if min_score is not None and score < min_score:
                    continue
                candidates.append((rid, score))

        # Sort descending by score, tie-break by record ID for determinism
        candidates.sort(key=lambda x: (-x[1], x[0]))

        selected = candidates[:top_k]
        results: list[RetrievalResult] = []
        for rank, (rid, score) in enumerate(selected, start=1):
            results.append(
                RetrievalResult(
                    record=self._records[rid],
                    score=float(score),
                    rank=rank,
                )
            )

        return results

    def retrieve_with_scores(
        self,
        query: list[float] | np.ndarray | str | MemoryQuery,
        top_k: int = 1,
    ) -> list[tuple[ExperienceRecord, float]]:
        """Convenience method returning (ExperienceRecord, score) tuples."""
        results = self.retrieve(query, top_k=top_k)
        return [(r.record, r.score) for r in results]

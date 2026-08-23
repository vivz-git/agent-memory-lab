"""Engineering Extension: Adaptive Retrieval Filtering via System-1 Read Rejection."""

from __future__ import annotations

from typing import Any
import numpy as np

from src.memory.bank import BaseMemoryBank
from src.memory.schema import ExperienceRecord, RetrievalResult


class AdaptiveReadFilter:
    """System-1 'Read' Rejection filter for episodic memory retrieval.

    Concept:
        Standard history-based deletion only kicks in after a record has been retrieved
        at least n times. If a toxic/misaligned record is admitted to memory, it will pollute
        the agent's prompt context n times before eviction.

        AdaptiveReadFilter intercepts top-K retrieval candidates and applies an adaptive
        read mask: if a candidate record has historical mean utility below an adaptive
        threshold (derived from fixed baseline or agent moving average), it is rejected
        from entering the prompt, dynamically backing off to the next nearest (K+1)-th candidate.
    """

    def __init__(
        self,
        min_retrievals: int = 1,
        utility_threshold: float = 0.5,
        use_moving_average: bool = True,
        moving_avg_window: int = 50,
        margin: float = 0.1,
        oversample_factor: int = 4,
        fallback_to_top_k: bool = True,
        higher_is_better: bool = True,
    ) -> None:
        """Initialize AdaptiveReadFilter.

        Args:
            min_retrievals: Minimum retrieval history count before read rejection applies (default: 1).
            utility_threshold: Baseline utility threshold gamma below which records are filtered.
            use_moving_average: Whether to adapt threshold dynamically based on recent agent utility.
            moving_avg_window: Window size for agent moving average utility.
            margin: Safety margin subtracted from moving average (threshold = max(baseline, avg - margin)).
            oversample_factor: Multiplier for retrieving candidate pool from memory bank to allow backoff.
            fallback_to_top_k: If all candidates are filtered, whether to fallback to top-K raw candidates.
            higher_is_better: Whether higher utility values represent superior quality.
        """
        self.min_retrievals = int(min_retrievals)
        self.utility_threshold = float(utility_threshold)
        self.use_moving_average = use_moving_average
        self.moving_avg_window = int(moving_avg_window)
        self.margin = float(margin)
        self.oversample_factor = int(oversample_factor)
        self.fallback_to_top_k = fallback_to_top_k
        self.higher_is_better = higher_is_better

        self._agent_utility_history: list[float] = []
        self._total_queries: int = 0
        self._total_evaluated_candidates: int = 0
        self._total_rejected_candidates: int = 0
        self._total_accepted_candidates: int = 0

    def update_agent_utility(self, utility_score: float) -> None:
        """Record downstream task utility to update agent moving average.

        Args:
            utility_score: Downstream task execution score.
        """
        self._agent_utility_history.append(float(utility_score))
        if len(self._agent_utility_history) > self.moving_avg_window * 2:
            self._agent_utility_history = self._agent_utility_history[-self.moving_avg_window :]

    def get_current_threshold(self) -> float:
        """Compute the current effective rejection threshold."""
        if not self.use_moving_average or not self._agent_utility_history:
            return self.utility_threshold

        recent = self._agent_utility_history[-self.moving_avg_window :]
        moving_avg = float(np.mean(recent))

        if self.higher_is_better:
            # Dynamic threshold: max of static baseline and (moving_avg - margin)
            return max(self.utility_threshold, moving_avg - self.margin)
        else:
            # For loss/error: min of static baseline and (moving_avg + margin)
            return min(self.utility_threshold, moving_avg + self.margin)

    def should_reject(self, record: ExperienceRecord) -> bool:
        """Check if an individual candidate record should be rejected from the prompt context.

        Args:
            record: The ExperienceRecord candidate.

        Returns:
            True if record is rejected, False if admitted.
        """
        # If record lacks sufficient downstream evaluations, do not reject
        if record.retrieval_count < self.min_retrievals:
            return False

        effective_threshold = self.get_current_threshold()
        if self.higher_is_better:
            return record.mean_utility < effective_threshold
        else:
            return record.mean_utility > effective_threshold

    def filter_candidates(
        self,
        candidates: list[RetrievalResult],
        top_k: int = 1,
    ) -> list[RetrievalResult]:
        """Filter a list of retrieved candidates, backing off until top_k accepted are found.

        Args:
            candidates: Ordered candidate list from memory bank retrieval.
            top_k: Desired number of accepted demonstrations.

        Returns:
            List of accepted RetrievalResult objects up to top_k.
        """
        self._total_queries += 1
        accepted: list[RetrievalResult] = []

        for cand in candidates:
            self._total_evaluated_candidates += 1
            if self.should_reject(cand.record):
                self._total_rejected_candidates += 1
                continue

            self._total_accepted_candidates += 1
            accepted.append(cand)
            if len(accepted) == top_k:
                break

        # Fallback handling if filter was too aggressive
        if len(accepted) == 0 and self.fallback_to_top_k and candidates:
            accepted = candidates[:top_k]

        # Re-assign ranks for the accepted subset
        reranked: list[RetrievalResult] = []
        for rank, res in enumerate(accepted, start=1):
            reranked.append(
                RetrievalResult(
                    record=res.record,
                    score=res.score,
                    rank=rank,
                )
            )

        return reranked

    def retrieve_filtered(
        self,
        bank: BaseMemoryBank,
        query: list[float] | np.ndarray | str | Any,
        top_k: int = 1,
        filter_ids: set[str] | None = None,
    ) -> list[RetrievalResult]:
        """Convenience method: retrieve oversampled pool from bank and apply adaptive filtering.

        Args:
            bank: The memory bank instance.
            query: Query key / vector / MemoryQuery.
            top_k: Target number of accepted exemplars.
            filter_ids: Optional set of record IDs to exclude.

        Returns:
            List of accepted RetrievalResult objects of size <= top_k.
        """
        if top_k <= 0 or bank.size() == 0:
            return []

        oversample_k = min(bank.size(), max(top_k * self.oversample_factor, top_k + 10))
        raw_candidates = bank.retrieve(
            query=query,
            top_k=oversample_k,
            filter_ids=filter_ids,
        )

        return self.filter_candidates(raw_candidates, top_k=top_k)

    def get_stats(self) -> dict[str, Any]:
        """Return diagnostic metrics on filter activations."""
        total_eval = self._total_evaluated_candidates
        rejection_rate = (
            float(self._total_rejected_candidates / total_eval)
            if total_eval > 0
            else 0.0
        )
        return {
            "total_queries": self._total_queries,
            "total_evaluated_candidates": self._total_evaluated_candidates,
            "total_rejected_candidates": self._total_rejected_candidates,
            "total_accepted_candidates": self._total_accepted_candidates,
            "rejection_rate": rejection_rate,
            "current_threshold": self.get_current_threshold(),
        }

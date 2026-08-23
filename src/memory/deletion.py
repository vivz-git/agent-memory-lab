"""Memory deletion policies governing episodic memory forgetting and eviction."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from src.memory.bank import BaseMemoryBank
from src.memory.schema import DeletionPolicyType, ExperienceRecord


class BaseDeletionPolicy(ABC):
    """Abstract base class for memory deletion / eviction policies."""

    policy_type: DeletionPolicyType

    @abstractmethod
    def get_eviction_candidates(
        self,
        bank: BaseMemoryBank,
        current_step: int,
        **kwargs: Any,
    ) -> list[str]:
        """Identify candidate record IDs to delete from the memory bank.

        Args:
            bank: The memory bank instance to inspect.
            current_step: The current execution step / timestamp.
            **kwargs: Additional contextual information.

        Returns:
            List of record IDs marked for eviction.
        """
        pass

    def apply(
        self,
        bank: BaseMemoryBank,
        current_step: int,
        **kwargs: Any,
    ) -> list[str]:
        """Execute deletion on the bank and return the list of evicted record IDs.

        Args:
            bank: The memory bank instance.
            current_step: The current execution step.
            **kwargs: Additional contextual information.

        Returns:
            List of successfully deleted record IDs.
        """
        candidates = self.get_eviction_candidates(bank, current_step, **kwargs)
        if candidates:
            bank.delete_many(candidates)
        return candidates


class PeriodicDeletionPolicy(BaseDeletionPolicy):
    """Periodic activity-based deletion policy: phi_per(i, t, t-T).

    Evicts memory records whose retrieval frequency within the rolling window [t-T, t]
    is less than or equal to alpha.
    """

    policy_type = DeletionPolicyType.PERIODIC

    def __init__(
        self,
        period: int = 500,
        alpha: int = 0,
        min_age: int = 0,
        force: bool = False,
    ) -> None:
        """Initialize PeriodicDeletionPolicy.

        Args:
            period: Window length T and evaluation step period.
            alpha: Maximum activity threshold (retrievals <= alpha triggers eviction).
            min_age: Grace period: records entered after (current_step - min_age) are protected.
            force: If True, evaluates eviction regardless of step cadence (current_step % period == 0).
        """
        self.period = int(period)
        self.alpha = int(alpha)
        self.min_age = int(min_age)
        self.force = force

    def is_due(self, current_step: int) -> bool:
        """Check if periodic deletion should run at the current step."""
        if self.force:
            return True
        return current_step > 0 and (current_step % self.period == 0)

    def get_eviction_candidates(
        self,
        bank: BaseMemoryBank,
        current_step: int,
        **kwargs: Any,
    ) -> list[str]:
        """Collect records whose retrieval count in [t-period, t] <= alpha."""
        if not self.is_due(current_step):
            return []

        start_step = max(0, current_step - self.period)
        candidates: list[str] = []

        for record in bank.all_records():
            # Respect grace period for newly added records
            if self.min_age > 0 and record.entry_step > (current_step - self.min_age):
                continue

            window_retrievals = record.retrievals_in_window(start_step, current_step)
            if window_retrievals <= self.alpha:
                candidates.append(record.id)

        return candidates


class HistoryBasedDeletionPolicy(BaseDeletionPolicy):
    """History-based (utility-based) deletion policy: phi_hist(i, t).

    Evicts memory records that have been retrieved at least n times (preventing sample bias)
    and whose historical mean downstream utility is less than or equal to beta.
    """

    policy_type = DeletionPolicyType.HISTORY

    def __init__(
        self,
        min_retrievals: int = 5,
        utility_threshold: float = 0.5,
        higher_is_better: bool = True,
    ) -> None:
        """Initialize HistoryBasedDeletionPolicy.

        Args:
            min_retrievals: Minimum retrieval count n before record is eligible for history deletion.
            utility_threshold: Threshold beta below which (or above which if not higher_is_better)
                               the record is evicted.
            higher_is_better: If True, evicts when mean_utility <= utility_threshold (rewards/SR);
                              If False, evicts when mean_utility >= utility_threshold (loss/error).
        """
        self.min_retrievals = int(min_retrievals)
        self.utility_threshold = float(utility_threshold)
        self.higher_is_better = higher_is_better

    def should_evict_record(self, record: ExperienceRecord) -> bool:
        """Check if an individual record meets the history deletion criteria."""
        if record.retrieval_count < self.min_retrievals:
            return False

        if self.higher_is_better:
            return record.mean_utility <= self.utility_threshold
        else:
            return record.mean_utility >= self.utility_threshold

    def get_eviction_candidates(
        self,
        bank: BaseMemoryBank,
        current_step: int,
        **kwargs: Any,
    ) -> list[str]:
        """Collect all records in bank satisfying history deletion criteria."""
        candidates: list[str] = []
        for record in bank.all_records():
            if self.should_evict_record(record):
                candidates.append(record.id)
        return candidates


class CombinedDeletionPolicy(BaseDeletionPolicy):
    """Combined deletion policy: phi_comb = phi_per OR phi_hist.

    Unions candidate evictions from periodic activity pruning and history utility pruning.
    """

    policy_type = DeletionPolicyType.COMBINED

    def __init__(
        self,
        periodic_policy: PeriodicDeletionPolicy | None = None,
        history_policy: HistoryBasedDeletionPolicy | None = None,
        period: int = 500,
        alpha: int = 0,
        min_retrievals: int = 5,
        utility_threshold: float = 0.5,
        higher_is_better: bool = True,
    ) -> None:
        """Initialize CombinedDeletionPolicy."""
        self.periodic_policy = periodic_policy or PeriodicDeletionPolicy(
            period=period,
            alpha=alpha,
        )
        self.history_policy = history_policy or HistoryBasedDeletionPolicy(
            min_retrievals=min_retrievals,
            utility_threshold=utility_threshold,
            higher_is_better=higher_is_better,
        )

    def get_eviction_candidates(
        self,
        bank: BaseMemoryBank,
        current_step: int,
        **kwargs: Any,
    ) -> list[str]:
        """Return union of candidates from periodic and history deletion."""
        candidates_set: set[str] = set()

        if self.periodic_policy.is_due(current_step):
            candidates_set.update(
                self.periodic_policy.get_eviction_candidates(bank, current_step, **kwargs)
            )

        candidates_set.update(
            self.history_policy.get_eviction_candidates(bank, current_step, **kwargs)
        )

        return sorted(list(candidates_set))


class ConstrainedCapacityDeletionPolicy(BaseDeletionPolicy):
    """Resource-constrained memory deletion policy enforcing hard capacity bound M_max.

    When |D_t| > M_max:
    1. First apply periodic / history deletion if due.
    2. If size still exceeds M_max, evict lowest mean utility records until |D_t| <= M_max.
    """

    policy_type = DeletionPolicyType.CONSTRAINED_CAPACITY

    def __init__(
        self,
        max_capacity: int = 100,
        periodic_policy: PeriodicDeletionPolicy | None = None,
        history_policy: HistoryBasedDeletionPolicy | None = None,
        higher_is_better: bool = True,
    ) -> None:
        """Initialize ConstrainedCapacityDeletionPolicy.

        Args:
            max_capacity: Hard memory size limit M_max.
            periodic_policy: Optional periodic policy applied first.
            history_policy: Optional history policy applied first.
            higher_is_better: Whether higher utility is better when ranking candidates.
        """
        self.max_capacity = int(max_capacity)
        self.periodic_policy = periodic_policy
        self.history_policy = history_policy
        self.higher_is_better = higher_is_better

    def get_eviction_candidates(
        self,
        bank: BaseMemoryBank,
        current_step: int,
        **kwargs: Any,
    ) -> list[str]:
        """Find candidate IDs to evict to satisfy capacity limit M_max."""
        candidates: set[str] = set()

        # Step 1: Apply periodic deletion if configured and due
        if self.periodic_policy is not None and self.periodic_policy.is_due(current_step):
            candidates.update(
                self.periodic_policy.get_eviction_candidates(bank, current_step, **kwargs)
            )

        # Step 2: Apply history deletion if configured
        if self.history_policy is not None:
            candidates.update(
                self.history_policy.get_eviction_candidates(bank, current_step, **kwargs)
            )

        # Check effective remaining size
        current_size = bank.size()
        effective_size = current_size - len(candidates)

        if effective_size > self.max_capacity:
            excess = effective_size - self.max_capacity
            remaining_records = [
                r for r in bank.all_records() if r.id not in candidates
            ]

            # Sort remaining records by utility (worst first)
            # Tie-break by retrieval count ascending, then entry_step ascending
            if self.higher_is_better:
                remaining_records.sort(
                    key=lambda r: (r.mean_utility, r.retrieval_count, r.entry_step, r.id)
                )
            else:
                remaining_records.sort(
                    key=lambda r: (-r.mean_utility, r.retrieval_count, r.entry_step, r.id)
                )

            for r in remaining_records[:excess]:
                candidates.add(r.id)

        return sorted(list(candidates))


def create_deletion_policy(
    policy_type: DeletionPolicyType | str,
    **kwargs: Any,
) -> BaseDeletionPolicy:
    """Factory to construct memory deletion policies.

    Args:
        policy_type: The policy type enum or string name.
        **kwargs: Arguments forwarded to the policy constructor.

    Returns:
        An instance of BaseDeletionPolicy.
    """
    if isinstance(policy_type, str):
        policy_type = DeletionPolicyType(policy_type.lower())

    if policy_type == DeletionPolicyType.NONE:
        # None deletion: dummy policy that never evicts
        class NoDeletionPolicy(BaseDeletionPolicy):
            policy_type = DeletionPolicyType.NONE
            def get_eviction_candidates(self, bank: BaseMemoryBank, current_step: int, **kw: Any) -> list[str]:
                return []
        return NoDeletionPolicy()
    elif policy_type == DeletionPolicyType.PERIODIC:
        return PeriodicDeletionPolicy(**kwargs)
    elif policy_type == DeletionPolicyType.HISTORY:
        return HistoryBasedDeletionPolicy(**kwargs)
    elif policy_type == DeletionPolicyType.COMBINED:
        return CombinedDeletionPolicy(**kwargs)
    elif policy_type == DeletionPolicyType.CONSTRAINED_CAPACITY:
        return ConstrainedCapacityDeletionPolicy(**kwargs)
    else:
        raise ValueError(f"Unknown deletion policy type: {policy_type}")

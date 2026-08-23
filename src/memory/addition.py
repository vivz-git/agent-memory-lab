"""Memory addition policies governing episodic memory admission."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable

from src.memory.schema import AdditionPolicyType


class BaseAdditionPolicy(ABC):
    """Abstract base class for memory addition admission policies."""

    policy_type: AdditionPolicyType

    @abstractmethod
    def should_add(
        self,
        query: Any,
        trajectory: Any,
        evaluation_result: Any = None,
        ground_truth: Any = None,
        **kwargs: Any,
    ) -> bool:
        """Determine whether to admit (add) the query-trajectory experience into memory.

        Args:
            query: The task query input.
            trajectory: The agent's generated output / execution trajectory.
            evaluation_result: Optional result from an evaluator or LLM judge.
            ground_truth: Optional ground-truth target for oracle verification.
            **kwargs: Additional metadata or context.

        Returns:
            True if the experience should be admitted into memory, False otherwise.
        """
        pass

    def __call__(
        self,
        query: Any,
        trajectory: Any,
        evaluation_result: Any = None,
        ground_truth: Any = None,
        **kwargs: Any,
    ) -> bool:
        """Call policy as a function."""
        return self.should_add(
            query=query,
            trajectory=trajectory,
            evaluation_result=evaluation_result,
            ground_truth=ground_truth,
            **kwargs,
        )


class FixedAdditionPolicy(BaseAdditionPolicy):
    """Fixed memory baseline: pi_fixed(q, e) = 0.

    Never adds any new experiences. Memory remains frozen at the initial seed bank D_0.
    """

    policy_type = AdditionPolicyType.FIXED

    def should_add(
        self,
        query: Any,
        trajectory: Any,
        evaluation_result: Any = None,
        ground_truth: Any = None,
        **kwargs: Any,
    ) -> bool:
        """Always returns False."""
        return False


class AddAllAdditionPolicy(BaseAdditionPolicy):
    """Add-all naive growth baseline: pi_all(q, e) = 1.

    Unconditionally stores every executed trajectory into memory.
    """

    policy_type = AdditionPolicyType.ADD_ALL

    def should_add(
        self,
        query: Any,
        trajectory: Any,
        evaluation_result: Any = None,
        ground_truth: Any = None,
        **kwargs: Any,
    ) -> bool:
        """Always returns True."""
        return True


class CoarseAdditionPolicy(BaseAdditionPolicy):
    """Selective memory addition via coarse / automatic evaluators (LLM Judge or heuristic).

    pi_coarse(q, e) = 1[Evaluator_auto(q, e) >= tau_coarse].
    """

    policy_type = AdditionPolicyType.COARSE

    def __init__(
        self,
        threshold: float = 1.0,
        error_based: bool = False,
        accept_strings: tuple[str, ...] = ("yes", "correct", "true", "pass", "success", "1"),
    ) -> None:
        """Initialize CoarseAdditionPolicy.

        Args:
            threshold: Quality score threshold (or max error if error_based=True).
            error_based: If True, admits experience when error <= threshold;
                         If False, admits experience when score >= threshold.
            accept_strings: Valid positive string judgment prefixes.
        """
        self.threshold = float(threshold)
        self.error_based = error_based
        self.accept_strings = tuple(s.lower() for s in accept_strings)

    def should_add(
        self,
        query: Any,
        trajectory: Any,
        evaluation_result: Any = None,
        ground_truth: Any = None,
        **kwargs: Any,
    ) -> bool:
        """Check if coarse evaluator result passes admission criteria."""
        if evaluation_result is not None:
            if isinstance(evaluation_result, bool):
                return evaluation_result

            if isinstance(evaluation_result, (int, float)):
                val = float(evaluation_result)
                if self.error_based:
                    return val <= self.threshold
                return val >= self.threshold

            if isinstance(evaluation_result, str):
                cleaned = evaluation_result.strip().lower()
                # Check first line or exact token
                first_line = cleaned.splitlines()[0].strip() if cleaned else ""
                for token in self.accept_strings:
                    if first_line == token or first_line.startswith(token):
                        return True
                return False

        # If evaluation_result is None but ground_truth is provided with error_based
        if ground_truth is not None and self.error_based:
            try:
                pred_val = float(trajectory)
                gt_val = float(ground_truth)
                return abs(pred_val - gt_val) <= self.threshold
            except (ValueError, TypeError):
                pass

        return False


class StrictAdditionPolicy(BaseAdditionPolicy):
    """Selective memory addition via strict ground-truth oracle verification.

    pi_strict(q, e) = 1[OracleMatch(e, e*) == 1].
    """

    policy_type = AdditionPolicyType.STRICT

    def __init__(
        self,
        error_threshold: float = 1.0,
        exact_match: bool = False,
        oracle_fn: Callable[[Any, Any], bool] | None = None,
    ) -> None:
        """Initialize StrictAdditionPolicy.

        Args:
            error_threshold: Maximum permissible error under continuous metrics (e.g. |y - y_hat| <= 1.0).
            exact_match: Whether to require exact equality (pred == gt).
            oracle_fn: Optional custom callable (trajectory, ground_truth) -> bool.
        """
        self.error_threshold = float(error_threshold)
        self.exact_match = exact_match
        self.oracle_fn = oracle_fn

    def should_add(
        self,
        query: Any,
        trajectory: Any,
        evaluation_result: Any = None,
        ground_truth: Any = None,
        **kwargs: Any,
    ) -> bool:
        """Check if experience passes strict oracle verification."""
        if self.oracle_fn is not None:
            return bool(self.oracle_fn(trajectory, ground_truth))

        if evaluation_result is not None:
            if isinstance(evaluation_result, bool):
                return evaluation_result
            if isinstance(evaluation_result, (int, float)):
                return float(evaluation_result) <= self.error_threshold

        if ground_truth is not None:
            if self.exact_match:
                return str(trajectory).strip() == str(ground_truth).strip()

            try:
                pred_val = float(trajectory)
                gt_val = float(ground_truth)
                return abs(pred_val - gt_val) <= self.error_threshold
            except (ValueError, TypeError):
                return str(trajectory).strip().lower() == str(ground_truth).strip().lower()

        return False


def create_addition_policy(
    policy_type: AdditionPolicyType | str,
    **kwargs: Any,
) -> BaseAdditionPolicy:
    """Factory to construct memory addition policies.

    Args:
        policy_type: The policy type enum or string name.
        **kwargs: Arguments forwarded to the policy constructor.

    Returns:
        An instance of BaseAdditionPolicy.
    """
    if isinstance(policy_type, str):
        policy_type = AdditionPolicyType(policy_type.lower())

    if policy_type == AdditionPolicyType.FIXED:
        return FixedAdditionPolicy()
    elif policy_type == AdditionPolicyType.ADD_ALL:
        return AddAllAdditionPolicy()
    elif policy_type == AdditionPolicyType.COARSE:
        return CoarseAdditionPolicy(**kwargs)
    elif policy_type == AdditionPolicyType.STRICT:
        return StrictAdditionPolicy(**kwargs)
    else:
        raise ValueError(f"Unknown addition policy type: {policy_type}")

"""Evaluation framework package for Memory Management & Experience-Following Dynamics in LLM Agents.

Contains metrics, evaluators, leakage prevention checks, configuration schemas,
and benchmark runners.
"""

from src.evaluation.config import (
    AddAllConfig,
    BaselineConfig,
    BenchmarkConfig,
    BoundedCapacityConfig,
    CoarseAdditionConfig,
    CombinedDeletionConfig,
    FixedMemoryConfig,
    HistoryDeletionConfig,
    NoDeletionConfig,
    PeriodicDeletionConfig,
    ProtocolAConfig,
    ProtocolBConfig,
    ProtocolCConfig,
    ProtocolDConfig,
    ProtocolEConfig,
    StrictAdditionConfig,
    StrictDeletionConfig,
)
from src.evaluation.evaluator import (
    BaseEvaluator,
    CICIOTCoarseEvaluator,
    CICIOTStrictEvaluator,
    CoarseLLMJudgeEvaluator,
    EvaluationResult,
    RegAgentCoarseEvaluator,
    RegAgentStrictEvaluator,
    StrictOracleEvaluator,
    parse_ciciot_prediction,
    parse_judge_decision,
    parse_regagent_prediction,
)
from src.evaluation.leakage import (
    LeakageChecker,
    LeakageReport,
    LeakageViolation,
)
from src.evaluation.metrics import (
    compute_accuracy,
    compute_cosine_similarity,
    compute_error_propagation_gap,
    compute_error_replication_rate,
    compute_experience_following_correlation,
    compute_l2_error,
    compute_memory_retention_ratio,
    compute_rbf_similarity,
    compute_regression_success_rate,
)
from src.evaluation.runner import (
    ExperimentResult,
    ExperimentRunner,
    StepMetric,
)

__all__ = [
    # Metrics
    "compute_accuracy",
    "compute_regression_success_rate",
    "compute_experience_following_correlation",
    "compute_error_propagation_gap",
    "compute_memory_retention_ratio",
    "compute_l2_error",
    "compute_error_replication_rate",
    "compute_cosine_similarity",
    "compute_rbf_similarity",
    # Evaluators
    "BaseEvaluator",
    "EvaluationResult",
    "StrictOracleEvaluator",
    "CoarseLLMJudgeEvaluator",
    "RegAgentStrictEvaluator",
    "RegAgentCoarseEvaluator",
    "CICIOTStrictEvaluator",
    "CICIOTCoarseEvaluator",
    "parse_regagent_prediction",
    "parse_ciciot_prediction",
    "parse_judge_decision",
    # Leakage
    "LeakageChecker",
    "LeakageReport",
    "LeakageViolation",
    # Config
    "BaselineConfig",
    "FixedMemoryConfig",
    "AddAllConfig",
    "CoarseAdditionConfig",
    "StrictAdditionConfig",
    "NoDeletionConfig",
    "PeriodicDeletionConfig",
    "HistoryDeletionConfig",
    "StrictDeletionConfig",
    "CombinedDeletionConfig",
    "BoundedCapacityConfig",
    "ProtocolAConfig",
    "ProtocolBConfig",
    "ProtocolCConfig",
    "ProtocolDConfig",
    "ProtocolEConfig",
    "BenchmarkConfig",
    # Runner
    "ExperimentRunner",
    "StepMetric",
    "ExperimentResult",
]

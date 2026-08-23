"""Evaluation metrics for Memory Management & Experience-Following Dynamics.

Implements mathematical formulations defined in research/RESEARCH_SPEC.md and
research/evaluation_plan.md:
- Task Accuracy & Regression Success Rate
- Experience-Following Pearson Correlation (r_EF)
- Error Propagation Gap (Delta_EP)
- Memory Retention Ratio (rho)
- Continuous Trajectory L2 Error
- Error Replication Rate (ERR)
"""

from __future__ import annotations

import math
from typing import Any, Sequence, Union
import numpy as np


def compute_accuracy(
    predictions: Sequence[Any],
    ground_truths: Sequence[Any],
    normalize_strings: bool = True,
) -> float:
    """Compute exact match classification accuracy.

    Args:
        predictions: Sequence of predicted labels or values.
        ground_truths: Sequence of true labels or values.
        normalize_strings: If True, strip whitespace and compare strings case-insensitively.

    Returns:
        Fraction of exact matches in range [0.0, 1.0]. Returns 0.0 if empty.
    """
    if len(predictions) != len(ground_truths):
        raise ValueError(
            f"Length mismatch: predictions ({len(predictions)}) != ground_truths ({len(ground_truths)})"
        )
    if not predictions:
        return 0.0

    correct = 0
    for pred, gt in zip(predictions, ground_truths):
        if normalize_strings and isinstance(pred, str) and isinstance(gt, str):
            if pred.strip().lower() == gt.strip().lower():
                correct += 1
        else:
            if pred == gt:
                correct += 1

    return float(correct / len(predictions))


def compute_regression_success_rate(
    predictions: Sequence[float | int],
    ground_truths: Sequence[float | int],
    threshold: float = 1.0,
) -> float:
    """Compute regression success rate (SR) defined as fraction of |pred - gt| <= threshold.

    For RegAgent, the standard strict threshold is 1.0.

    Args:
        predictions: Sequence of numeric predictions.
        ground_truths: Sequence of ground truth numeric targets.
        threshold: Maximum absolute error allowed for success (default 1.0).

    Returns:
        Fraction of predictions within threshold in range [0.0, 1.0]. Returns 0.0 if empty.
    """
    if len(predictions) != len(ground_truths):
        raise ValueError(
            f"Length mismatch: predictions ({len(predictions)}) != ground_truths ({len(ground_truths)})"
        )
    if not predictions:
        return 0.0

    successful = sum(
        1 for pred, gt in zip(predictions, ground_truths)
        if abs(float(pred) - float(gt)) <= threshold
    )
    return float(successful / len(predictions))


def compute_experience_following_correlation(
    input_similarities: Sequence[float],
    output_similarities: Sequence[float],
) -> float:
    """Compute Pearson correlation coefficient r between input and output similarities.

    Formula:
        r = sum((S_in - mean(S_in)) * (S_out - mean(S_out))) /
            (sqrt(sum((S_in - mean(S_in))^2)) * sqrt(sum((S_out - mean(S_out))^2)))

    Args:
        input_similarities: Sequence of input similarity scores S_in(q_t, q_k).
        output_similarities: Sequence of output similarity scores S_out(e_t, e_k).

    Returns:
        Pearson correlation r in [-1.0, 1.0]. Returns 0.0 if sample size < 2 or variance is 0.
    """
    if len(input_similarities) != len(output_similarities):
        raise ValueError(
            f"Length mismatch: input_similarities ({len(input_similarities)}) != "
            f"output_similarities ({len(output_similarities)})"
        )
    n = len(input_similarities)
    if n < 2:
        return 0.0

    x = np.asarray(input_similarities, dtype=np.float64)
    y = np.asarray(output_similarities, dtype=np.float64)

    # Handle NaN or Inf
    valid_mask = np.isfinite(x) & np.isfinite(y)
    if np.sum(valid_mask) < 2:
        return 0.0

    x = x[valid_mask]
    y = y[valid_mask]

    x_mean = np.mean(x)
    y_mean = np.mean(y)

    x_dev = x - x_mean
    y_dev = y - y_mean

    ss_x = np.sum(x_dev ** 2)
    ss_y = np.sum(y_dev ** 2)

    # If variance of either variable is zero, correlation is undefined -> return 0.0
    if ss_x <= 1e-15 or ss_y <= 1e-15:
        return 0.0

    r = float(np.sum(x_dev * y_dev) / np.sqrt(ss_x * ss_y))

    # Clamp to [-1.0, 1.0] to guard against floating point inaccuracies
    return float(np.clip(r, -1.0, 1.0))


def compute_error_propagation_gap(
    actual_metrics: Union[float, Sequence[float]],
    error_free_metrics: Union[float, Sequence[float]],
) -> Union[float, list[float]]:
    """Compute error propagation compounding gap Delta_EP.

    Formula:
        Delta_EP(t) = Metric(Agent_EF, t) - Metric(Agent, t)

    Args:
        actual_metrics: Performance metric(s) achieved by actual agent.
        error_free_metrics: Performance metric(s) achieved by error-free (EF) oracle twin.

    Returns:
        Scalar gap if inputs are scalars, or list of step gaps if sequences.
    """
    if isinstance(actual_metrics, (int, float)) and isinstance(error_free_metrics, (int, float)):
        return float(error_free_metrics) - float(actual_metrics)

    if isinstance(actual_metrics, Sequence) and isinstance(error_free_metrics, Sequence):
        if len(actual_metrics) != len(error_free_metrics):
            raise ValueError(
                f"Length mismatch: actual_metrics ({len(actual_metrics)}) != "
                f"error_free_metrics ({len(error_free_metrics)})"
            )
        return [float(ef) - float(act) for act, ef in zip(actual_metrics, error_free_metrics)]

    raise TypeError("Inputs must both be numeric scalars or both be sequences of numbers.")


def compute_memory_retention_ratio(
    current_mem_size: int,
    total_added: int,
    initial_mem_size: int = 0,
) -> float:
    """Compute memory retention ratio rho(t).

    Formula:
        rho(t) = M(t) / (N_0 + total_added)

    Args:
        current_mem_size: Current number of records in memory bank M(t).
        total_added: Total number of records added up to time t.
        initial_mem_size: Initial seed memory size N_0.

    Returns:
        Retention ratio in range [0.0, 1.0] (or > 1.0 if misconfigured).
    """
    total_capacity_attempted = initial_mem_size + total_added
    if total_capacity_attempted <= 0:
        return 1.0 if current_mem_size == 0 else 0.0
    return float(current_mem_size / total_capacity_attempted)


def compute_l2_error(
    predictions: Sequence[Sequence[float] | float],
    ground_truths: Sequence[Sequence[float] | float],
) -> float:
    """Compute mean L2 Euclidean distance between predicted and ground truth vectors/scalars.

    Args:
        predictions: Sequence of predicted vector waypoints or scalar numbers.
        ground_truths: Sequence of target vector waypoints or scalar numbers.

    Returns:
        Mean L2 error across all samples.
    """
    if len(predictions) != len(ground_truths):
        raise ValueError("Length mismatch between predictions and ground truths.")
    if not predictions:
        return 0.0

    errors = []
    for pred, gt in zip(predictions, ground_truths):
        pred_arr = np.atleast_1d(np.asarray(pred, dtype=np.float64))
        gt_arr = np.atleast_1d(np.asarray(gt, dtype=np.float64))
        errors.append(np.linalg.norm(pred_arr - gt_arr))

    return float(np.mean(errors))


def compute_error_replication_rate(
    is_actual_erroneous: Sequence[bool],
    retrieved_has_error: Sequence[bool],
    output_similarities: Sequence[float],
    mimic_threshold: float = 0.8,
) -> float:
    """Compute Error Replication Rate (ERR).

    ERR = P(e_t is erroneous and S_out(e_t, e_k) >= tau_mimic | retrieved has error)

    Args:
        is_actual_erroneous: Boolean flag per step indicating if agent execution was erroneous.
        retrieved_has_error: Boolean flag per step indicating if retrieved exemplar contained error.
        output_similarities: Output similarity to top retrieved demonstration.
        mimic_threshold: Output similarity threshold to classify as imitation (default 0.8).

    Returns:
        Conditional probability in [0.0, 1.0]. Returns 0.0 if condition count is 0.
    """
    if not (len(is_actual_erroneous) == len(retrieved_has_error) == len(output_similarities)):
        raise ValueError("All input sequences must have identical lengths.")

    condition_count = 0
    replicated_count = 0

    for act_err, ret_err, s_out in zip(is_actual_erroneous, retrieved_has_error, output_similarities):
        if ret_err:
            condition_count += 1
            if act_err and s_out >= mimic_threshold:
                replicated_count += 1

    if condition_count == 0:
        return 0.0
    return float(replicated_count / condition_count)


def compute_cosine_similarity(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
    """Compute cosine similarity between two 1D vectors: dot(a, b) / (||a|| * ||b||)."""
    a = np.asarray(vec_a, dtype=np.float64)
    b = np.asarray(vec_b, dtype=np.float64)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a <= 1e-15 or norm_b <= 1e-15:
        return 0.0
    return float(np.clip(np.dot(a, b) / (norm_a * norm_b), -1.0, 1.0))


def compute_rbf_similarity(val_a: float | Sequence[float], val_b: float | Sequence[float], gamma: float = 1.0) -> float:
    """Compute RBF Gaussian kernel similarity: exp(-gamma * ||a - b||^2)."""
    a = np.atleast_1d(np.asarray(val_a, dtype=np.float64))
    b = np.atleast_1d(np.asarray(val_b, dtype=np.float64))
    sq_dist = np.sum((a - b) ** 2)
    return float(math.exp(-gamma * sq_dist))

"""ExperimentRunner module for executing benchmark task streams, recording step-by-step metrics,
running counterfactual error-free (EF) parallel twins, and serializing results.

Implements execution specifications from research/RESEARCH_SPEC.md and research/evaluation_plan.md.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, field
import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence, Union
import numpy as np

from src.evaluation.evaluator import (
    BaseEvaluator,
    EvaluationResult,
    StrictOracleEvaluator,
    parse_regagent_prediction,
)
from src.evaluation.metrics import (
    compute_accuracy,
    compute_error_propagation_gap,
    compute_error_replication_rate,
    compute_experience_following_correlation,
    compute_memory_retention_ratio,
    compute_regression_success_rate,
    compute_rbf_similarity,
    compute_cosine_similarity,
)


@dataclass
class StepMetric:
    """Detailed observation and metric record at a single stream step t."""
    step: int
    query: Any
    ground_truth: Any
    trajectory: Any
    passed: bool
    score: float
    error_magnitude: Optional[float] = None
    retrieved_count: int = 0
    input_similarity: float = 0.0
    output_similarity: float = 0.0
    memory_size: int = 0
    running_success_rate: float = 0.0
    cumulative_r_ef: float = 0.0
    is_erroneous: bool = False
    retrieved_has_error: bool = False
    ef_trajectory: Optional[Any] = None
    ef_passed: Optional[bool] = None
    ef_error_magnitude: Optional[float] = None
    ef_running_success_rate: Optional[float] = None
    error_propagation_gap: Optional[float] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # Serialize arrays or complex objects to primitives
        if isinstance(d["query"], np.ndarray):
            d["query"] = d["query"].tolist()
        if isinstance(d["ground_truth"], np.ndarray):
            d["ground_truth"] = d["ground_truth"].tolist()
        if isinstance(d["trajectory"], np.ndarray):
            d["trajectory"] = d["trajectory"].tolist()
        return d


@dataclass
class ExperimentResult:
    """Aggregated experimental results and time-series metric trajectories."""
    experiment_name: str
    seed: int
    total_steps: int
    final_success_rate: float
    final_memory_size: int
    initial_memory_size: int
    total_added: int
    total_deleted: int
    pearson_r_ef: float
    error_replication_rate: float
    memory_retention_ratio: float
    mean_error_propagation_gap: Optional[float] = None
    ef_final_success_rate: Optional[float] = None
    step_metrics: list[StepMetric] = field(default_factory=list)
    deleted_record_errors: list[float] = field(default_factory=list)
    retained_record_errors: list[float] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_name": self.experiment_name,
            "seed": self.seed,
            "total_steps": self.total_steps,
            "final_success_rate": self.final_success_rate,
            "final_memory_size": self.final_memory_size,
            "initial_memory_size": self.initial_memory_size,
            "total_added": self.total_added,
            "total_deleted": self.total_deleted,
            "pearson_r_ef": self.pearson_r_ef,
            "error_replication_rate": self.error_replication_rate,
            "memory_retention_ratio": self.memory_retention_ratio,
            "mean_error_propagation_gap": self.mean_error_propagation_gap,
            "ef_final_success_rate": self.ef_final_success_rate,
            "deleted_record_errors_mean": float(np.mean(self.deleted_record_errors)) if self.deleted_record_errors else None,
            "retained_record_errors_mean": float(np.mean(self.retained_record_errors)) if self.retained_record_errors else None,
            "metadata": self.metadata,
        }

    def to_json(self, filepath: Union[str, Path], include_step_metrics: bool = True) -> None:
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = self.to_dict()
        if include_step_metrics:
            data["step_metrics"] = [m.to_dict() for m in self.step_metrics]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)

    def to_csv(self, filepath: Union[str, Path]) -> None:
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not self.step_metrics:
            return

        fieldnames = list(self.step_metrics[0].to_dict().keys())
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for metric in self.step_metrics:
                writer.writerow(metric.to_dict())

    def summary(self) -> str:
        lines = [
            f"=== EXPERIMENT RESULTS: {self.experiment_name} (Seed: {self.seed}) ===",
            f"Total Task Horizon (T): {self.total_steps}",
            f"Final Success Rate (SR/ACC): {self.final_success_rate * 100:.2f}%",
            f"Experience-Following Pearson r_EF: {self.pearson_r_ef:.4f}",
            f"Initial Mem Size: {self.initial_memory_size} -> Final Mem Size: {self.final_memory_size}",
            f"Total Added: {self.total_added} | Total Deleted: {self.total_deleted}",
            f"Memory Retention Ratio: {self.memory_retention_ratio:.4f}",
            f"Error Replication Rate: {self.error_replication_rate * 100:.2f}%",
        ]
        if self.mean_error_propagation_gap is not None:
            lines.append(f"Mean Error Propagation Gap (Delta_EP): {self.mean_error_propagation_gap:.4f}")
        if self.ef_final_success_rate is not None:
            lines.append(f"Error-Free Twin Success Rate: {self.ef_final_success_rate * 100:.2f}%")
        if self.deleted_record_errors and self.retained_record_errors:
            lines.append(f"Deleted Mean Error: {np.mean(self.deleted_record_errors):.4f} | Retained Mean Error: {np.mean(self.retained_record_errors):.4f}")
        return "\n".join(lines)


# Protocols for Orchestrators / Agents / Memory Banks for dependency injection

class MemoryRecordProtocol(Protocol):
    id: Any
    query: Any
    trajectory: Any
    ground_truth: Optional[Any]
    utility_history: List[float]
    mean_utility: float


class AgentProtocol(Protocol):
    def execute(self, query: Any, demonstrations: Sequence[Any]) -> Any:
        ...


class MemoryBankProtocol(Protocol):
    def retrieve(self, query: Any, top_k: int) -> Sequence[Any]:
        ...

    def add(self, query: Any, trajectory: Any, ground_truth: Optional[Any] = None) -> bool:
        ...

    def update_utility(self, retrieved_records: Sequence[Any], utility_score: float) -> None:
        ...

    def prune(self, current_step: int) -> List[Any]:
        ...

    def size(self) -> int:
        ...

    def get_all_records(self) -> Sequence[Any]:
        ...


class ExperimentRunner:
    """Benchmark runner executing task streams against memory agents."""

    def __init__(
        self,
        evaluator: Optional[BaseEvaluator] = None,
        oracle_evaluator: Optional[BaseEvaluator] = None,
        top_k: int = 6,
        track_error_free_twin: bool = True,
        output_dir: Optional[Union[str, Path]] = None,
    ):
        self.evaluator = evaluator or StrictOracleEvaluator(task_type="regression")
        self.oracle_evaluator = oracle_evaluator or StrictOracleEvaluator(task_type="regression")
        self.top_k = top_k
        self.track_error_free_twin = track_error_free_twin
        self.output_dir = Path(output_dir) if output_dir else Path("results")

    def run_stream(
        self,
        experiment_name: str,
        queries: Sequence[Any],
        ground_truths: Sequence[Any],
        agent_fn: Callable[[Any, Sequence[Any]], Any],
        memory_bank: Any,
        seed: int = 42,
        ef_agent_fn: Optional[Callable[[Any, Sequence[Any]], Any]] = None,
        input_sim_fn: Optional[Callable[[Any, Any], float]] = None,
        output_sim_fn: Optional[Callable[[Any, Any], float]] = None,
        addition_gating: bool = True,
        addition_predicate: Optional[Callable[[EvaluationResult], bool]] = None,
        periodic_prune_interval: int = 0,
    ) -> ExperimentResult:
        """Execute a full sequential task stream.

        At each step t:
        1. Retrieve top-K demonstrations from memory bank.
        2. Agent executes trajectory: e_t = agent_fn(q_t, xi_K).
        3. Parallel Error-Free Twin executes trajectory: e_t_ef = ef_agent_fn(q_t, xi_K_ef).
        4. Trajectory evaluator scores e_t -> score, passed.
        5. Calculate input similarity S_in(q_t, q_k*) and output similarity S_out(e_t, e_k*).
        6. If addition_gating passes, add (q_t, e_t) to memory bank.
        7. Update downstream utility score for retrieved records.
        8. Check and run deletion policies (periodic / history / bounded).
        9. Record step metrics.
        """
        if len(queries) != len(ground_truths):
            raise ValueError(f"Stream length mismatch: {len(queries)} queries vs {len(ground_truths)} ground truths")

        total_steps = len(queries)
        initial_mem_size = memory_bank.size() if hasattr(memory_bank, "size") else 0
        total_added = 0
        total_deleted = 0

        step_metrics: List[StepMetric] = []
        in_sims: List[float] = []
        out_sims: List[float] = []
        actual_passed_list: List[bool] = []
        ef_passed_list: List[bool] = []
        deleted_record_errors: List[float] = []

        # Default similarity helpers
        _in_sim_fn = input_sim_fn or (lambda q1, q2: compute_cosine_similarity(q1, q2) if hasattr(q1, "__len__") else 1.0)
        _out_sim_fn = output_sim_fn or (lambda e1, e2: compute_rbf_similarity(
            parse_regagent_prediction(e1) or 0.0,
            parse_regagent_prediction(e2) or 0.0,
            gamma=1.0,
        ))

        for t in range(total_steps):
            q_t = queries[t]
            gt_t = ground_truths[t]

            # 1. Retrieve top-K demonstrations
            retrieved_demos: List[Any] = []
            if hasattr(memory_bank, "retrieve"):
                retrieved_demos = list(memory_bank.retrieve(q_t, top_k=self.top_k))

            # 2. Agent Execution
            trajectory = agent_fn(q_t, retrieved_demos)

            # 3. Evaluate Trajectory (Evaluator score Phi and addition gate)
            eval_res = self.evaluator.evaluate(q_t, trajectory, ground_truth=gt_t)
            oracle_res = self.oracle_evaluator.evaluate(q_t, trajectory, ground_truth=gt_t)
            actual_passed = oracle_res.passed
            actual_passed_list.append(actual_passed)

            # 4. Error-Free Oracle Twin Execution
            ef_trajectory = None
            ef_passed = None
            ef_err_mag = None
            if self.track_error_free_twin:
                # Prepare EF demonstrations (replacing demonstration trajectories with ground truths)
                ef_demos = []
                for demo in retrieved_demos:
                    ef_demo = demo
                    if hasattr(demo, "ground_truth") and demo.ground_truth is not None:
                        # Construct twin demonstration with ground truth
                        if isinstance(demo, tuple) and len(demo) >= 2:
                            ef_demo = (demo[0], demo.ground_truth if hasattr(demo, "ground_truth") else demo[1])
                    ef_demos.append(ef_demo)

                ef_agent = ef_agent_fn or agent_fn
                ef_trajectory = ef_agent(q_t, ef_demos)
                ef_oracle_res = self.oracle_evaluator.evaluate(q_t, ef_trajectory, ground_truth=gt_t)
                ef_passed = ef_oracle_res.passed
                ef_err_mag = ef_oracle_res.error_magnitude
                ef_passed_list.append(ef_passed)

            # 5. Experience-Following Similarities
            s_in_max = 0.0
            s_out_top = 0.0
            retrieved_has_error = False

            if retrieved_demos:
                best_sim = -float("inf")
                best_demo = None
                for demo in retrieved_demos:
                    demo_q = demo[0] if isinstance(demo, tuple) else getattr(demo, "query", None)
                    demo_e = demo[1] if isinstance(demo, tuple) else getattr(demo, "trajectory", None)
                    demo_gt = getattr(demo, "ground_truth", None)

                    if demo_gt is not None:
                        demo_eval = self.oracle_evaluator.evaluate(demo_q, demo_e, ground_truth=demo_gt)
                        if not demo_eval.passed:
                            retrieved_has_error = True

                    if demo_q is not None:
                        sim = _in_sim_fn(q_t, demo_q)
                        if sim > best_sim:
                            best_sim = sim
                            best_demo = demo

                s_in_max = max(0.0, float(best_sim))
                if best_demo is not None:
                    best_e = best_demo[1] if isinstance(best_demo, tuple) else getattr(best_demo, "trajectory", None)
                    if best_e is not None:
                        s_out_top = max(0.0, float(_out_sim_fn(trajectory, best_e)))

            in_sims.append(s_in_max)
            out_sims.append(s_out_top)

            # 6. Memory Addition
            should_add = True
            if addition_gating:
                if addition_predicate is not None:
                    should_add = addition_predicate(eval_res)
                else:
                    should_add = eval_res.passed

            if should_add and hasattr(memory_bank, "add"):
                added = memory_bank.add(q_t, trajectory, ground_truth=gt_t)
                if added:
                    total_added += 1

            # 7. Memory Utility Update
            if retrieved_demos and hasattr(memory_bank, "update_utility"):
                # Utility signal Phi is the downstream task success / score
                utility_signal = 1.0 if actual_passed else 0.0
                memory_bank.update_utility(retrieved_demos, utility_score=utility_signal)

            # 8. Deletion / Pruning
            if hasattr(memory_bank, "prune"):
                if periodic_prune_interval <= 0 or (t + 1) % periodic_prune_interval == 0:
                    pruned = memory_bank.prune(current_step=t)
                    if pruned:
                        total_deleted += len(pruned)
                        for p in pruned:
                            p_gt = getattr(p, "ground_truth", None)
                            p_e = getattr(p, "trajectory", None)
                            if p_gt is not None and p_e is not None:
                                p_eval = self.oracle_evaluator.evaluate(getattr(p, "query", None), p_e, ground_truth=p_gt)
                                if p_eval.error_magnitude is not None:
                                    deleted_record_errors.append(p_eval.error_magnitude)

            # 9. Track Running Metrics
            running_sr = float(np.mean(actual_passed_list))
            ef_running_sr = float(np.mean(ef_passed_list)) if ef_passed_list else None
            cum_r = compute_experience_following_correlation(in_sims, out_sims) if len(in_sims) >= 2 else 0.0
            ep_gap = (ef_running_sr - running_sr) if ef_running_sr is not None else None
            current_mem_sz = memory_bank.size() if hasattr(memory_bank, "size") else 0

            step_m = StepMetric(
                step=t,
                query=q_t,
                ground_truth=gt_t,
                trajectory=trajectory,
                passed=actual_passed,
                score=eval_res.score,
                error_magnitude=oracle_res.error_magnitude,
                retrieved_count=len(retrieved_demos),
                input_similarity=s_in_max,
                output_similarity=s_out_top,
                memory_size=current_mem_sz,
                running_success_rate=running_sr,
                cumulative_r_ef=cum_r,
                is_erroneous=not actual_passed,
                retrieved_has_error=retrieved_has_error,
                ef_trajectory=ef_trajectory,
                ef_passed=ef_passed,
                ef_error_magnitude=ef_err_mag,
                ef_running_success_rate=ef_running_sr,
                error_propagation_gap=ep_gap,
                metadata=eval_res.metadata,
            )
            step_metrics.append(step_m)

        # Collect retained records error distribution for KDE analysis
        retained_record_errors: List[float] = []
        if hasattr(memory_bank, "get_all_records"):
            for rec in memory_bank.get_all_records():
                rec_gt = getattr(rec, "ground_truth", None)
                rec_e = getattr(rec, "trajectory", None)
                rec_q = getattr(rec, "query", None)
                if rec_gt is not None and rec_e is not None:
                    rec_eval = self.oracle_evaluator.evaluate(rec_q, rec_e, ground_truth=rec_gt)
                    if rec_eval.error_magnitude is not None:
                        retained_record_errors.append(rec_eval.error_magnitude)

        final_sr = float(np.mean(actual_passed_list)) if actual_passed_list else 0.0
        final_ef_sr = float(np.mean(ef_passed_list)) if ef_passed_list else None
        final_mem_size = memory_bank.size() if hasattr(memory_bank, "size") else 0
        pearson_r = compute_experience_following_correlation(in_sims, out_sims)
        err_rate = compute_error_replication_rate(
            [not p for p in actual_passed_list],
            [m.retrieved_has_error for m in step_metrics],
            out_sims,
        )
        retention_ratio = compute_memory_retention_ratio(final_mem_size, total_added, initial_mem_size)
        mean_ep_gap = (final_ef_sr - final_sr) if final_ef_sr is not None else None

        result = ExperimentResult(
            experiment_name=experiment_name,
            seed=seed,
            total_steps=total_steps,
            final_success_rate=final_sr,
            final_memory_size=final_mem_size,
            initial_memory_size=initial_mem_size,
            total_added=total_added,
            total_deleted=total_deleted,
            pearson_r_ef=pearson_r,
            error_replication_rate=err_rate,
            memory_retention_ratio=retention_ratio,
            mean_error_propagation_gap=mean_ep_gap,
            ef_final_success_rate=final_ef_sr,
            step_metrics=step_metrics,
            deleted_record_errors=deleted_record_errors,
            retained_record_errors=retained_record_errors,
            metadata={"top_k": self.top_k, "tracked_ef": self.track_error_free_twin},
        )

        return result

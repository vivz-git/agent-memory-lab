"""CLI Benchmark Runner for Memory Management & Experience-Following Experiments.

Supports launching Protocol A, Protocol B, Protocol C, Protocol D, and Protocol E
across multiple seeds and horizon lengths.

Usage:
    python benchmarks/run_benchmark.py --protocol A --env reg_agent -T 500 --seeds 42 128
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
import logging
import os
from pathlib import Path
import sys
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.config import (
    AddAllConfig,
    BenchmarkConfig,
    BoundedCapacityConfig,
    CoarseAdditionConfig,
    CombinedDeletionConfig,
    FixedMemoryConfig,
    HistoryDeletionConfig,
    PeriodicDeletionConfig,
    ProtocolAConfig,
    ProtocolBConfig,
    ProtocolCConfig,
    ProtocolDConfig,
    ProtocolEConfig,
    StrictAdditionConfig,
)
from src.evaluation.evaluator import (
    CoarseLLMJudgeEvaluator,
    RegAgentCoarseEvaluator,
    RegAgentStrictEvaluator,
    StrictOracleEvaluator,
)
from src.evaluation.leakage import LeakageChecker
from src.evaluation.metrics import (
    compute_accuracy,
    compute_cosine_similarity,
    compute_experience_following_correlation,
    compute_rbf_similarity,
    compute_regression_success_rate,
)
from src.evaluation.runner import ExperimentResult, ExperimentRunner


# =====================================================================
# Standalone Benchmark Harness (RegAgent Synthetic Task Generator & Memory)
# =====================================================================

@dataclass
class SyntheticMemoryRecord:
    id: int
    query: np.ndarray
    trajectory: str
    ground_truth: float
    retrieval_count: int = 0
    utility_history: List[float] = field(default_factory=list)
    entry_step: int = 0
    last_retrieved_step: int = 0

    @property
    def mean_utility(self) -> float:
        return float(np.mean(self.utility_history)) if self.utility_history else 0.0


class SyntheticMemoryBank:
    """Vectorized episodic memory bank supporting additions, periodic, history, and bounded deletions."""

    def __init__(
        self,
        addition_policy: str = "strict",
        deletion_policy: str = "none",
        coarse_level: str = "C1",
        history_min_retrievals: int = 5,
        history_beta: float = 0.5,
        periodic_period: int = 500,
        periodic_alpha: int = 0,
        max_capacity: Optional[int] = None,
        regression_evaluator: Optional[Any] = None,
    ):
        self.addition_policy = addition_policy.lower()
        self.deletion_policy = deletion_policy.lower()
        self.coarse_level = coarse_level
        self.history_min_retrievals = history_min_retrievals
        self.history_beta = history_beta
        self.periodic_period = periodic_period
        self.periodic_alpha = periodic_alpha
        self.max_capacity = max_capacity
        self.records: List[SyntheticMemoryRecord] = []
        self._next_id = 0
        self.evaluator = regression_evaluator or RegAgentStrictEvaluator(threshold=1.0)
        self.coarse_evaluator = RegAgentCoarseEvaluator(level=coarse_level)

    def size(self) -> int:
        return len(self.records)

    def get_all_records(self) -> List[SyntheticMemoryRecord]:
        return self.records

    def add(self, query: np.ndarray, trajectory: str, ground_truth: Optional[float] = None) -> bool:
        """Evaluate addition gate and append if admitted."""
        if self.addition_policy == "fixed":
            return False
        elif self.addition_policy == "add_all":
            pass  # always admitted
        elif self.addition_policy == "coarse":
            res = self.coarse_evaluator.evaluate(query, trajectory, ground_truth=ground_truth)
            if not res.passed:
                return False
        elif self.addition_policy == "strict":
            res = self.evaluator.evaluate(query, trajectory, ground_truth=ground_truth)
            if not res.passed:
                return False

        rec = SyntheticMemoryRecord(
            id=self._next_id,
            query=np.asarray(query, dtype=np.float64),
            trajectory=trajectory,
            ground_truth=float(ground_truth) if ground_truth is not None else 0.0,
            entry_step=0,
        )
        self._next_id += 1
        self.records.append(rec)

        # Handle hard capacity limit if bounded
        if self.max_capacity is not None and len(self.records) > self.max_capacity:
            self._evict_lowest_utility()

        return True

    def retrieve(self, query: np.ndarray, top_k: int = 6) -> List[SyntheticMemoryRecord]:
        if not self.records:
            return []
        q = np.asarray(query, dtype=np.float64)
        sims = []
        for rec in self.records:
            sim = compute_cosine_similarity(q, rec.query)
            sims.append(sim)

        top_indices = np.argsort(sims)[::-1][: min(top_k, len(self.records))]
        retrieved = [self.records[idx] for idx in top_indices]
        for rec in retrieved:
            rec.retrieval_count += 1
        return retrieved

    def update_utility(self, retrieved_records: Sequence[SyntheticMemoryRecord], utility_score: float) -> None:
        for rec in retrieved_records:
            rec.utility_history.append(float(utility_score))

    def prune(self, current_step: int) -> List[SyntheticMemoryRecord]:
        if self.deletion_policy == "none":
            return []

        pruned: List[SyntheticMemoryRecord] = []
        retained: List[SyntheticMemoryRecord] = []

        for rec in self.records:
            should_delete = False

            # Periodic deletion check
            if self.deletion_policy in ("periodic", "combined", "bounded"):
                if (current_step + 1) % self.periodic_period == 0:
                    if rec.retrieval_count <= self.periodic_alpha:
                        should_delete = True

            # History-based utility deletion check
            if self.deletion_policy in ("history", "combined", "bounded") and not should_delete:
                if rec.retrieval_count >= self.history_min_retrievals:
                    if rec.mean_utility <= self.history_beta:
                        should_delete = True

            if should_delete:
                pruned.append(rec)
            else:
                retained.append(rec)

        self.records = retained
        return pruned

    def _evict_lowest_utility(self) -> Optional[SyntheticMemoryRecord]:
        if not self.records:
            return None
        # Sort by mean utility ascending, evict lowest
        lowest_idx = min(range(len(self.records)), key=lambda i: (self.records[i].mean_utility, self.records[i].retrieval_count))
        evicted = self.records.pop(lowest_idx)
        return evicted


def generate_regagent_dataset(
    seed: int,
    num_samples: int,
    w_dim: int = 6,
    noise_range: float = 1.0,
    cluster_shifts: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic 6D regression data y = w^T x + noise."""
    rng = np.random.default_rng(seed)
    w = rng.standard_normal(w_dim)

    if cluster_shifts:
        # Generate GMM-like cluster shifts (mu in {-0.5, 0.0, 0.5})
        n_per_cluster = num_samples // 3
        remainder = num_samples - n_per_cluster * 3
        means = [-0.5, 0.0, 0.5]
        x_list = []
        for i, mu in enumerate(means):
            size = n_per_cluster + (remainder if i == 2 else 0)
            x_cluster = rng.normal(loc=mu, scale=1.0, size=(size, w_dim))
            x_list.append(x_cluster)
        x = np.vstack(x_list)
    else:
        x = rng.standard_normal(size=(num_samples, w_dim))

    noise = rng.uniform(-noise_range, noise_range, size=num_samples)
    y = np.dot(x, w) + noise
    return x, y, w


def synthetic_regagent_policy(
    query: np.ndarray,
    demonstrations: Sequence[SyntheticMemoryRecord],
    w_true: np.ndarray,
    noise_std: float = 0.3,
    rng: Optional[np.random.Generator] = None,
) -> str:
    """Simulate RegAgent in-context experience-following policy.

    If demonstrations exist, agent interpolates heavily towards closest demo output
    (exhibiting experience-following behavior r ~ 0.9).
    """
    _rng = rng or np.random.default_rng()
    if not demonstrations:
        # Pure prior estimate with noise
        guess = float(np.dot(query, w_true) + _rng.normal(0, noise_std))
        return f"Guess: boxed{{{guess:.4f}}}"

    # Find closest demo by cosine similarity
    best_sim = -float("inf")
    best_demo = demonstrations[0]
    for d in demonstrations:
        sim = compute_cosine_similarity(query, d.query)
        if sim > best_sim:
            best_sim = sim
            best_demo = d

    # Parse demo output
    import re
    demo_match = re.search(r"boxed\{([+-]?\d*\.?\d+)\}", best_demo.trajectory)
    demo_val = float(demo_match.group(1)) if demo_match else best_demo.ground_truth

    # Agent execution: conditions on retrieved demo with Experience-Following weight alpha = sim^2
    alpha = float(np.clip(best_sim, 0.0, 1.0)) ** 2
    delta = float(np.dot(query - best_demo.query, w_true))
    pred = alpha * (demo_val + delta) + (1.0 - alpha) * float(np.dot(query, w_true)) + float(_rng.normal(0, 0.2))

    return f"Guess: boxed{{{pred:.4f}}}"


# =====================================================================
# Benchmark Protocol Runners
# =====================================================================

def run_protocol_a(config: ProtocolAConfig, output_dir: Path, verbose: bool = True) -> Dict[str, Any]:
    """Execute Protocol A: Long-term memory growth."""
    print(f"\n🚀 Running Protocol A: Long-Term Memory Growth (T={config.stream_length})")
    results = {}
    leakage_checker = LeakageChecker()

    for seed in config.seeds:
        x_init, y_init, w = generate_regagent_dataset(seed=seed, num_samples=config.initial_memory_size)
        x_test, y_test, _ = generate_regagent_dataset(seed=seed + 10000, num_samples=config.stream_length)

        # Leakage verification
        leakage_report = leakage_checker.verify_split_isolation(x_init, x_test, min_distance=1e-4)
        if not leakage_report.is_clean:
            print(f"⚠️ Leakage warning on seed {seed}: {leakage_report.summary()}")

        for strategy in config.addition_strategies:
            mem = SyntheticMemoryBank(
                addition_policy=strategy.replace("coarse_c1", "coarse").replace("coarse_c2", "coarse").replace("coarse_c3", "coarse"),
                coarse_level="C1" if "c1" in strategy else ("C2" if "c2" in strategy else "C3"),
            )
            # Populate initial memory D_0
            for i in range(len(x_init)):
                mem.records.append(
                    SyntheticMemoryRecord(
                        id=i,
                        query=x_init[i],
                        trajectory=f"Guess: boxed{{{y_init[i]:.4f}}}",
                        ground_truth=y_init[i],
                    )
                )

            runner = ExperimentRunner(top_k=6, track_error_free_twin=config.evaluate_error_free_twin)
            rng = np.random.default_rng(seed)
            agent_fn = lambda q, demos: synthetic_regagent_policy(q, demos, w_true=w, rng=rng)

            exp_name = f"ProtocolA_{strategy}_seed{seed}"
            res = runner.run_stream(
                experiment_name=exp_name,
                queries=x_test,
                ground_truths=y_test,
                agent_fn=agent_fn,
                memory_bank=mem,
                seed=seed,
                addition_gating=strategy != "add_all",
            )
            res.to_json(output_dir / f"{exp_name}.json", include_step_metrics=False)
            res.to_csv(output_dir / f"{exp_name}.csv")
            results[exp_name] = res.to_dict()

            if verbose:
                print(f"  [{strategy.upper():<10}] Seed {seed:<4} -> SR: {res.final_success_rate*100:.2f}% | r_EF: {res.pearson_r_ef:.4f} | Mem: {res.final_memory_size}")

    return results


def run_protocol_b(config: ProtocolBConfig, output_dir: Path, verbose: bool = True) -> Dict[str, Any]:
    """Execute Protocol B: Deletion KDE Utility Evaluation."""
    print(f"\n🚀 Running Protocol B: Memory Deletion & Utility KDE (T={config.stream_length})")
    results = {}

    for seed in config.seeds:
        x_init, y_init, w = generate_regagent_dataset(seed=seed, num_samples=config.initial_memory_size)
        x_test, y_test, _ = generate_regagent_dataset(seed=seed + 10000, num_samples=config.stream_length)

        for del_strat in config.deletion_strategies:
            mem = SyntheticMemoryBank(
                addition_policy="strict",
                deletion_policy=del_strat,
                history_min_retrievals=config.history_min_retrievals,
                history_beta=0.5,
                periodic_period=config.periodic_window,
            )
            for i in range(len(x_init)):
                mem.records.append(
                    SyntheticMemoryRecord(
                        id=i,
                        query=x_init[i],
                        trajectory=f"Guess: boxed{{{y_init[i]:.4f}}}",
                        ground_truth=y_init[i],
                    )
                )

            runner = ExperimentRunner(top_k=6)
            rng = np.random.default_rng(seed)
            agent_fn = lambda q, demos: synthetic_regagent_policy(q, demos, w_true=w, rng=rng)

            exp_name = f"ProtocolB_{del_strat}_seed{seed}"
            res = runner.run_stream(
                experiment_name=exp_name,
                queries=x_test,
                ground_truths=y_test,
                agent_fn=agent_fn,
                memory_bank=mem,
                seed=seed,
            )
            res.to_json(output_dir / f"{exp_name}.json", include_step_metrics=False)
            results[exp_name] = res.to_dict()

            if verbose:
                del_mean = f"{np.mean(res.deleted_record_errors):.4f}" if res.deleted_record_errors else "N/A"
                ret_mean = f"{np.mean(res.retained_record_errors):.4f}" if res.retained_record_errors else "N/A"
                print(f"  [{del_strat.upper():<10}] Seed {seed:<4} -> SR: {res.final_success_rate*100:.2f}% | Del Error Mean: {del_mean} | Ret Error Mean: {ret_mean}")

    return results


def run_protocol_c(config: ProtocolCConfig, output_dir: Path, verbose: bool = True) -> Dict[str, Any]:
    """Execute Protocol C: Task Distribution Shift."""
    print(f"\n🚀 Running Protocol C: Task Distribution Shift (GMM Clusters, T={config.stream_length})")
    results = {}

    for seed in config.seeds:
        x_init, y_init, w = generate_regagent_dataset(seed=seed, num_samples=config.initial_memory_size)
        x_test, y_test, _ = generate_regagent_dataset(seed=seed + 10000, num_samples=config.stream_length, cluster_shifts=True)

        for strat in ["fixed", "strict_no_del", "strict_combined"]:
            add_pol = "fixed" if strat == "fixed" else "strict"
            del_pol = "combined" if "combined" in strat else "none"

            mem = SyntheticMemoryBank(addition_policy=add_pol, deletion_policy=del_pol, periodic_period=200)
            for i in range(len(x_init)):
                mem.records.append(
                    SyntheticMemoryRecord(id=i, query=x_init[i], trajectory=f"Guess: boxed{{{y_init[i]:.4f}}}", ground_truth=y_init[i])
                )

            runner = ExperimentRunner(top_k=6)
            rng = np.random.default_rng(seed)
            agent_fn = lambda q, demos: synthetic_regagent_policy(q, demos, w_true=w, rng=rng)

            exp_name = f"ProtocolC_{strat}_seed{seed}"
            res = runner.run_stream(
                experiment_name=exp_name,
                queries=x_test,
                ground_truths=y_test,
                agent_fn=agent_fn,
                memory_bank=mem,
                seed=seed,
            )
            res.to_json(output_dir / f"{exp_name}.json", include_step_metrics=False)
            results[exp_name] = res.to_dict()

            if verbose:
                print(f"  [{strat:<16}] Seed {seed:<4} -> SR: {res.final_success_rate*100:.2f}% | Final Mem: {res.final_memory_size}")

    return results


def run_protocol_d(config: ProtocolDConfig, output_dir: Path, verbose: bool = True) -> Dict[str, Any]:
    """Execute Protocol D: Resource-Constrained Bounded Memory."""
    print(f"\n🚀 Running Protocol D: Resource-Constrained Bounded Memory (T={config.stream_length})")
    results = {}

    for seed in config.seeds:
        x_init, y_init, w = generate_regagent_dataset(seed=seed, num_samples=config.initial_memory_size)
        x_test, y_test, _ = generate_regagent_dataset(seed=seed + 10000, num_samples=config.stream_length)

        for cap in config.capacity_limits:
            mem = SyntheticMemoryBank(addition_policy="strict", deletion_policy="bounded", max_capacity=cap)
            for i in range(min(len(x_init), cap)):
                mem.records.append(
                    SyntheticMemoryRecord(id=i, query=x_init[i], trajectory=f"Guess: boxed{{{y_init[i]:.4f}}}", ground_truth=y_init[i])
                )

            runner = ExperimentRunner(top_k=6)
            rng = np.random.default_rng(seed)
            agent_fn = lambda q, demos: synthetic_regagent_policy(q, demos, w_true=w, rng=rng)

            exp_name = f"ProtocolD_Cap{cap}_seed{seed}"
            res = runner.run_stream(
                experiment_name=exp_name,
                queries=x_test,
                ground_truths=y_test,
                agent_fn=agent_fn,
                memory_bank=mem,
                seed=seed,
            )
            res.to_json(output_dir / f"{exp_name}.json", include_step_metrics=False)
            results[exp_name] = res.to_dict()

            if verbose:
                print(f"  [CAPACITY {cap:<4}] Seed {seed:<4} -> SR: {res.final_success_rate*100:.2f}% | Final Mem: {res.final_memory_size} (Max: {cap})")

    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Memory Management & Experience-Following Benchmarks.")
    parser.add_argument("--protocol", choices=["A", "B", "C", "D", "all"], default="A", help="Benchmark protocol to execute.")
    parser.add_argument("--env", default="reg_agent", help="Target agent environment (default: reg_agent).")
    parser.add_argument("-T", "--stream-length", type=int, default=500, help="Stream horizon length (default: 500).")
    parser.add_argument("-N0", "--initial-memory-size", type=int, default=100, help="Seed memory size N_0 (default: 100).")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 128], help="Random seeds for replication.")
    parser.add_argument("-M", "--max-memory", type=int, default=100, help="Capacity limit for Protocol D.")
    parser.add_argument("--output-dir", default="results", help="Directory to save experimental results.")
    parser.add_argument("--verbose", action="store_true", default=True, help="Print detailed step outputs.")

    args = parser.parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("🧠 AGENT MEMORY LAB: BENCHMARK SUITE")
    print(f"Protocol: {args.protocol} | Env: {args.env} | T: {args.stream_length} | N0: {args.initial_memory_size}")
    print(f"Seeds: {args.seeds} | Output: {out_dir}")
    print("=" * 70)

    all_results = {}

    if args.protocol in ("A", "all"):
        proto_a_cfg = ProtocolAConfig(
            stream_length=args.stream_length,
            initial_memory_size=args.initial_memory_size,
            seeds=args.seeds,
        )
        res_a = run_protocol_a(proto_a_cfg, output_dir=out_dir / "protocol_a", verbose=args.verbose)
        all_results["protocol_a"] = res_a

    if args.protocol in ("B", "all"):
        proto_b_cfg = ProtocolBConfig(
            stream_length=args.stream_length,
            initial_memory_size=args.initial_memory_size,
            seeds=args.seeds,
        )
        res_b = run_protocol_b(proto_b_cfg, output_dir=out_dir / "protocol_b", verbose=args.verbose)
        all_results["protocol_b"] = res_b

    if args.protocol in ("C", "all"):
        proto_c_cfg = ProtocolCConfig(
            stream_length=args.stream_length,
            initial_memory_size=args.initial_memory_size,
            seeds=args.seeds,
        )
        res_c = run_protocol_c(proto_c_cfg, output_dir=out_dir / "protocol_c", verbose=args.verbose)
        all_results["protocol_c"] = res_c

    if args.protocol in ("D", "all"):
        proto_d_cfg = ProtocolDConfig(
            stream_length=args.stream_length,
            initial_memory_size=args.initial_memory_size,
            capacity_limits=[args.max_memory] if args.max_memory else [50, 100, 180],
            seeds=args.seeds,
        )
        res_d = run_protocol_d(proto_d_cfg, output_dir=out_dir / "protocol_d", verbose=args.verbose)
        all_results["protocol_d"] = res_d

    # Serialize global summary
    with open(out_dir / "benchmark_summary.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=str)

    print("\n✅ Benchmark execution completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

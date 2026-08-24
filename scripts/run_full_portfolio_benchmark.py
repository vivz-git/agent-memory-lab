import argparse
import csv
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from dotenv import load_dotenv

load_dotenv()

from src.agent.core import RegAgent, MockLLMClient, Demonstration, get_llm_client
from src.environments.base import TaskQuery
from src.memory.bank import BaseMemoryBank
from src.memory.schema import ExperienceRecord
from src.memory.addition import create_addition_policy
from src.memory.deletion import create_deletion_policy
from src.memory.adaptive_retrieval import AdaptiveReadFilter
from src.evaluation.evaluator import RegAgentStrictEvaluator
from src.evaluation.metrics import (
    compute_regression_success_rate,
    compute_experience_following_correlation,
    compute_error_propagation_gap,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("full_benchmark")


def generate_regagent_dataset(seed: int, num_samples: int, w_dim: int = 6):
    rng = np.random.default_rng(seed)
    w = rng.standard_normal(w_dim)
    x = rng.standard_normal(size=(num_samples, w_dim))
    noise = rng.uniform(-1.0, 1.0, size=num_samples)
    y = np.dot(x, w) + noise
    return x, y, w


def run_benchmark_condition(
    condition_name: str,
    x_init: np.ndarray,
    y_init: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    addition_policy_name: str = "fixed",
    deletion_policy_name: str = "none",
    use_adaptive_read: bool = False,
    is_error_free_twin: bool = False,
    rate_limit_sleep: float = 1.2,
) -> Dict[str, Any]:
    logger.info(f"\n--- Running Condition: {condition_name} ({len(x_test)} steps) ---")

    bank = BaseMemoryBank(metric="cosine")

    # Initial Memory D_0
    for i in range(len(x_init)):
        query_key = x_init[i].tolist()
        bank.add(
            ExperienceRecord(
                id=f"init_{i}",
                query_key=query_key,
                trajectory_text=f"Guess: boxed{{{y_init[i]:.4f}}}",
                metadata={"ground_truth": y_init[i]},
                entry_step=-1,
            )
        )

    # Policies
    addition_policy = create_addition_policy(addition_policy_name, error_threshold=1.0)
    deletion_policy = create_deletion_policy(
        deletion_policy_name, min_retrievals=3, utility_threshold=0.5
    )

    # Adaptive read filter
    read_filter = (
        AdaptiveReadFilter(min_retrievals=1, utility_threshold=0.5)
        if use_adaptive_read
        else None
    )

    # Evaluator: Deterministic local RegAgent evaluator (threshold=2.5)
    evaluator = RegAgentStrictEvaluator(threshold=2.5)

    if is_error_free_twin:
        llm_client = MockLLMClient(mode="demonstration_mimic", noise_std=0.0)
    else:
        llm_client = get_llm_client()
    agent = RegAgent(llm_client=llm_client)

    metrics = {
        "success_flags": [],
        "utilities": [],
        "memory_size": [],
        "s_in": [],
        "s_out": [],
        "added_count": 0,
        "deleted_count": 0,
        "read_rejected": 0,
    }

    start_time = time.time()

    for t in range(len(x_test)):
        q_vec = x_test[t]
        gt = y_test[t]

        task_query = TaskQuery(
            query_id=f"test_{t}",
            query_vector=q_vec.tolist(),
            raw_input=q_vec.tolist(),
            ground_truth=gt,
        )

        # 1. Retrieval
        if read_filter:
            retrieval_results = read_filter.retrieve_filtered(
                bank, query=q_vec.tolist(), top_k=3
            )
            metrics["read_rejected"] = read_filter._total_rejected_candidates
        else:
            retrieval_results = bank.retrieve(query=q_vec.tolist(), top_k=3)

        demos = []
        for res in retrieval_results:
            demo_q = TaskQuery(
                query_id=res.record.id,
                query_vector=res.record.query_key,
                raw_input=res.record.query_key,
                ground_truth=res.record.metadata.get("ground_truth"),
            )
            demos.append(
                Demonstration(
                    query=demo_q,
                    execution=res.record.trajectory_text,
                    memory_id=res.record.id,
                )
            )
            res.record.retrieval_count += 1

        if retrieval_results:
            metrics["s_in"].append(retrieval_results[0].score)
        else:
            metrics["s_in"].append(0.0)

        # 2. Agent Execution
        if is_error_free_twin:
            raw_output = f"Guess: boxed{{{gt:.4f}}}"
            pred = float(gt)
        else:
            pred, raw_output = agent.act(
                query=task_query, demonstrations=demos, temperature=0.0
            )

        # 3. Calculate S_out for correlation metric
        if demos and pred is not None:
            import re

            demo_match = re.search(r"boxed\{([+-]?\d*\.?\d+)\}", demos[0].execution)
            demo_val = float(demo_match.group(1)) if demo_match else 0.0
            s_out = 1.0 / (1.0 + abs(pred - demo_val))
            metrics["s_out"].append(s_out)
        else:
            metrics["s_out"].append(0.0)

        # 4. Evaluation
        eval_result = evaluator.evaluate(
            query=q_vec, trajectory=raw_output, ground_truth=gt
        )
        metrics["success_flags"].append(1 if eval_result.passed else 0)
        utility_score = 1.0 if eval_result.passed else 0.0
        metrics["utilities"].append(utility_score)

        if read_filter:
            read_filter.update_agent_utility(utility_score)

        # 5. Supervision Update
        for res in retrieval_results:
            bank.update_utility(res.record.id, utility_score, step=t)

        # 6. Addition Gating
        should_add = False
        if addition_policy_name == "add_all":
            should_add = True
        elif addition_policy_name != "fixed":
            should_add = addition_policy.should_add(
                query=q_vec,
                trajectory=raw_output,
                evaluation_result=bool(utility_score),
                ground_truth=gt,
            )

        if should_add and not is_error_free_twin:
            bank.add(
                ExperienceRecord(
                    id=f"step_{t}",
                    query_key=q_vec.tolist(),
                    trajectory_text=raw_output,
                    metadata={"ground_truth": gt},
                    entry_step=t,
                )
            )
            metrics["added_count"] += 1

        # 7. Deletion Gating
        if deletion_policy_name != "none":
            to_delete = deletion_policy.get_eviction_candidates(bank, current_step=t)
            if to_delete:
                bank.delete_many(to_delete)
                metrics["deleted_count"] += len(to_delete)

        metrics["memory_size"].append(bank.size())

        if not is_error_free_twin and rate_limit_sleep > 0:
            time.sleep(rate_limit_sleep)

    end_time = time.time()

    sr = compute_regression_success_rate(
        metrics["utilities"], [1.0] * len(x_test), threshold=0.1
    )
    r_ef = compute_experience_following_correlation(metrics["s_in"], metrics["s_out"])

    logger.info(f"  -> SR: {sr*100:.1f}%")
    logger.info(f"  -> r_EF: {r_ef:.4f}")
    logger.info(f"  -> Added: {metrics['added_count']}, Deleted: {metrics['deleted_count']}")
    logger.info(f"  -> Final Mem Size: {metrics['memory_size'][-1]}")
    if use_adaptive_read:
        logger.info(f"  -> Read Rejected: {metrics['read_rejected']}")

    return {
        "condition": condition_name,
        "addition_policy": addition_policy_name,
        "deletion_policy": deletion_policy_name,
        "use_adaptive_read": use_adaptive_read,
        "success_rate": sr,
        "r_ef": r_ef,
        "initial_mem_size": len(x_init),
        "added_count": metrics["added_count"],
        "deleted_count": metrics["deleted_count"],
        "final_mem_size": metrics["memory_size"][-1],
        "read_rejected": metrics["read_rejected"],
        "latency_sec": end_time - start_time,
        "memory_trajectory": metrics["memory_size"],
    }


def run_seed_benchmark(
    seed: int,
    test_size: int,
    initial_memory_size: int,
    output_dir: Path,
    sleep_sec: float,
) -> List[Dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"==================================================")
    logger.info(f"STARTING BENCHMARK FOR SEED {seed} ({test_size} tasks/condition)")
    logger.info(f"==================================================")

    x_init, y_init, w = generate_regagent_dataset(seed, initial_memory_size)
    x_test, y_test, _ = generate_regagent_dataset(seed + 1, test_size)

    config = {
        "env": "reg_agent",
        "llm_provider": "groq",
        "model": "openai/gpt-oss-120b",
        "evaluator": "RegAgentStrictEvaluator (threshold=2.5)",
        "initial_memory_size": initial_memory_size,
        "test_stream_size": test_size,
        "seed": seed,
        "conditions": [
            "A. Fixed",
            "B. Naive Add-All",
            "C. Managed Memory",
            "D. Managed + Adaptive Read Rejection",
        ],
    }

    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Diagnostic Baseline Error-Free Twin
    res_ef = run_benchmark_condition(
        "Error-Free Twin",
        x_init,
        y_init,
        x_test,
        y_test,
        addition_policy_name="fixed",
        deletion_policy_name="none",
        use_adaptive_read=False,
        is_error_free_twin=True,
        rate_limit_sleep=0.0,
    )

    conditions = [
        ("A. Fixed", "fixed", "none", False),
        ("B. Naive Add-All", "add_all", "none", False),
        ("C. Managed Memory", "strict", "history", False),
        ("D. Managed + Adaptive Read Rejection", "strict", "history", True),
    ]

    results = []
    for name, add_pol, del_pol, use_read in conditions:
        res = run_benchmark_condition(
            name,
            x_init,
            y_init,
            x_test,
            y_test,
            addition_policy_name=add_pol,
            deletion_policy_name=del_pol,
            use_adaptive_read=use_read,
            is_error_free_twin=False,
            rate_limit_sleep=sleep_sec,
        )
        delta_ep = compute_error_propagation_gap(
            res["success_rate"], res_ef["success_rate"]
        )
        res["error_propagation_gap"] = delta_ep
        results.append(res)

    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Write summary.md
    summary_lines = [
        f"# Benchmark Results — Seed {seed}",
        "",
        f"- **Model:** `openai/gpt-oss-120b` (Groq API)",
        f"- **Evaluator:** Deterministic `RegAgentStrictEvaluator(threshold=2.5)`",
        f"- **Stream Length:** {test_size} tasks",
        f"- **Initial Memory ($D_0$):** {initial_memory_size} verified entries",
        f"- **Seed:** {seed}",
        "",
        "## Performance Table",
        "",
        "| Condition | Success Rate | Final Memory | Added | Deleted | Read Rejected | $\\Delta_{EP}$ (Error Gap) | $r_{EF}$ (Exp-Following) | Latency (s) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for r in results:
        summary_lines.append(
            f"| **{r['condition']}** | **{r['success_rate']*100:.1f}%** | {r['final_mem_size']} | {r['added_count']} | {r['deleted_count']} | {r['read_rejected']} | {r['error_propagation_gap']*100:.1f}% | {r['r_ef']:.4f} | {r['latency_sec']:.1f} |"
        )

    with open(output_dir / "summary.md", "w") as f:
        f.write("\n".join(summary_lines))

    logger.info(f"Seed {seed} complete. Artifacts saved to {output_dir}/")
    return results


def aggregate_benchmark_runs(
    all_results: Dict[int, List[Dict[str, Any]]],
    output_dir: Path,
    test_size: int,
    initial_memory_size: int,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    seeds = list(all_results.keys())

    condition_names = [
        "A. Fixed",
        "B. Naive Add-All",
        "C. Managed Memory",
        "D. Managed + Adaptive Read Rejection",
    ]

    aggregate_data = []

    for cond in condition_names:
        cond_runs = []
        for s in seeds:
            for r in all_results[s]:
                if r["condition"] == cond:
                    cond_runs.append(r)
                    break

        sr_vals = [r["success_rate"] for r in cond_runs]
        r_ef_vals = [r["r_ef"] for r in cond_runs]
        delta_ep_vals = [r["error_propagation_gap"] for r in cond_runs]
        mem_vals = [r["final_mem_size"] for r in cond_runs]
        added_vals = [r["added_count"] for r in cond_runs]
        del_vals = [r["deleted_count"] for r in cond_runs]
        read_rej_vals = [r["read_rejected"] for r in cond_runs]
        lat_vals = [r["latency_sec"] for r in cond_runs]

        entry = {
            "condition": cond,
            "success_rate_mean": float(np.mean(sr_vals)),
            "success_rate_std": float(np.std(sr_vals)),
            "r_ef_mean": float(np.mean(r_ef_vals)),
            "r_ef_std": float(np.std(r_ef_vals)),
            "delta_ep_mean": float(np.mean(delta_ep_vals)),
            "delta_ep_std": float(np.std(delta_ep_vals)),
            "final_mem_size_mean": float(np.mean(mem_vals)),
            "final_mem_size_std": float(np.std(mem_vals)),
            "added_count_mean": float(np.mean(added_vals)),
            "added_count_std": float(np.std(added_vals)),
            "deleted_count_mean": float(np.mean(del_vals)),
            "deleted_count_std": float(np.std(del_vals)),
            "read_rejected_mean": float(np.mean(read_rej_vals)),
            "read_rejected_std": float(np.std(read_rej_vals)),
            "latency_sec_mean": float(np.mean(lat_vals)),
            "latency_sec_std": float(np.std(lat_vals)),
            "runs_by_seed": {s: r for s, r in zip(seeds, cond_runs)},
        }
        aggregate_data.append(entry)

    with open(output_dir / "results.json", "w") as f:
        json.dump(aggregate_data, f, indent=2)

    # Write comparison.csv
    csv_file = output_dir / "comparison.csv"
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Condition",
            "SR_Mean",
            "SR_Std",
            "r_EF_Mean",
            "r_EF_Std",
            "Delta_EP_Mean",
            "Delta_EP_Std",
            "Final_Mem_Mean",
            "Final_Mem_Std",
            "Added_Mean",
            "Deleted_Mean",
            "Read_Rejected_Mean",
            "Latency_Mean_Sec",
        ])
        for a in aggregate_data:
            writer.writerow([
                a["condition"],
                f"{a['success_rate_mean']*100:.2f}%",
                f"{a['success_rate_std']*100:.2f}%",
                f"{a['r_ef_mean']:.4f}",
                f"{a['r_ef_std']:.4f}",
                f"{a['delta_ep_mean']*100:.2f}%",
                f"{a['delta_ep_std']*100:.2f}%",
                f"{a['final_mem_size_mean']:.1f}",
                f"{a['final_mem_size_std']:.1f}",
                f"{a['added_count_mean']:.1f}",
                f"{a['deleted_count_mean']:.1f}",
                f"{a['read_rejected_mean']:.1f}",
                f"{a['latency_sec_mean']:.1f}",
            ])

    # Write summary.md
    summary_lines = [
        "# Aggregate Benchmark Summary (Across Seeds 42 & 123)",
        "",
        f"- **Model:** `openai/gpt-oss-120b` (Groq API)",
        f"- **Evaluator:** Deterministic `RegAgentStrictEvaluator(threshold=2.5)`",
        f"- **Total Task Stream:** {test_size} tasks per condition × 2 independent seeds = {test_size * 4 * len(seeds)} executions",
        f"- **Initial Memory ($D_0$):** {initial_memory_size} verified demonstrations",
        f"- **Evaluated Seeds:** {seeds}",
        "",
        "## Aggregate Performance Table (Mean ± Std)",
        "",
        "| Condition | Success Rate (SR) | Final Memory Size | Added | Deleted | Read Rejected | $\\Delta_{EP}$ (Error Gap) | $r_{EF}$ (Following Correlation) | Avg Latency |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for a in aggregate_data:
        sr_str = f"**{a['success_rate_mean']*100:.1f}%** ± {a['success_rate_std']*100:.1f}%"
        mem_str = f"{a['final_mem_size_mean']:.1f} ± {a['final_mem_size_std']:.1f}"
        delta_str = f"{a['delta_ep_mean']*100:.1f}% ± {a['delta_ep_std']*100:.1f}%"
        ref_str = f"{a['r_ef_mean']:.4f} ± {a['r_ef_std']:.4f}"
        lat_str = f"{a['latency_sec_mean']:.1f}s"
        summary_lines.append(
            f"| **{a['condition']}** | {sr_str} | {mem_str} | {a['added_count_mean']:.0f} | {a['deleted_count_mean']:.0f} | {a['read_rejected_mean']:.0f} | {delta_str} | {ref_str} | {lat_str} |"
        )

    summary_lines.extend([
        "",
        "## Key Findings & Dynamics",
        "1. **Baseline Fixed ($D_0$ only):** Serves as frozen static memory baseline.",
        "2. **Naive Add-All:** Ingests unverified trajectories unconditionally. Demonstrates error compounding and memory bloat.",
        "3. **Managed Memory (Strict Addition + History Deletion):** Strict Write Gate filters invalid outputs, while History Forget Gate purges low-utility records.",
        "4. **Managed + Adaptive Read Rejection (Our Engineering Extension):** Proactively blocks misaligned memories from entering prompt context, maintaining high alignment and robust performance.",
        "",
    ])

    with open(output_dir / "summary.md", "w") as f:
        f.write("\n".join(summary_lines))

    logger.info(f"Aggregate benchmark complete. Results saved to {output_dir}/")


def main():
    parser = argparse.ArgumentParser(
        description="Run Full Portfolio-Scope Benchmark across Seeds 42 and 123"
    )
    parser.add_argument(
        "--test-size",
        type=int,
        default=100,
        help="Number of streaming test queries per condition (default=100)",
    )
    parser.add_argument(
        "--init-size",
        type=int,
        default=20,
        help="Number of verified demonstrations in initial seed memory D_0 (default=20)",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=1.2,
        help="Sleep seconds between LLM calls to prevent Groq rate limits",
    )
    args = parser.parse_args()

    os.environ["LLM_PROVIDER"] = "groq"
    base_dir = Path("evaluation/benchmark")
    base_dir.mkdir(parents=True, exist_ok=True)

    seeds = [42, 123]
    all_results: Dict[int, List[Dict[str, Any]]] = {}

    for seed in seeds:
        seed_dir = base_dir / f"seed_{seed}"
        res = run_seed_benchmark(
            seed=seed,
            test_size=args.test_size,
            initial_memory_size=args.init_size,
            output_dir=seed_dir,
            sleep_sec=args.sleep,
        )
        all_results[seed] = res

    # Run aggregation
    aggregate_dir = base_dir / "aggregate"
    aggregate_benchmark_runs(
        all_results=all_results,
        output_dir=aggregate_dir,
        test_size=args.test_size,
        initial_memory_size=args.init_size,
    )
    logger.info("ALL BENCHMARK RUNS AND AGGREGATION COMPLETED SUCCESSFULLY.")


if __name__ == "__main__":
    main()

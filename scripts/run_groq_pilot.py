import json
import logging
import time
from pathlib import Path

import numpy as np

import os
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

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("pilot")

def generate_regagent_dataset(seed: int, num_samples: int, w_dim: int = 6):
    rng = np.random.default_rng(seed)
    w = rng.standard_normal(w_dim)
    x = rng.standard_normal(size=(num_samples, w_dim))
    noise = rng.uniform(-1.0, 1.0, size=num_samples)
    y = np.dot(x, w) + noise
    return x, y, w

def run_pilot_condition(
    condition_name: str,
    x_init: np.ndarray,
    y_init: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    addition_policy_name: str = "fixed",
    deletion_policy_name: str = "none",
    use_adaptive_read: bool = False,
    is_error_free_twin: bool = False,
):
    logger.info(f"\n--- Running Condition: {condition_name} ---")
    
    # Initialize real memory bank
    bank = BaseMemoryBank(metric="cosine")
    
    # Pre-fill initial memory D_0
    for i in range(len(x_init)):
        query_key = x_init[i].tolist()
        bank.add(ExperienceRecord(
            id=f"init_{i}",
            query_key=query_key,
            trajectory_text=f"Guess: boxed{{{y_init[i]:.4f}}}",
            metadata={"ground_truth": y_init[i]},
            entry_step=-1,
        ))
    
    # Policies
    addition_policy = create_addition_policy(addition_policy_name, error_threshold=1.0)
    deletion_policy = create_deletion_policy(deletion_policy_name, min_retrievals=3, utility_threshold=0.5)
    
    # Adaptive read filter
    read_filter = AdaptiveReadFilter(min_retrievals=1, utility_threshold=0.5) if use_adaptive_read else None
    
    # Evaluator
    evaluator = RegAgentStrictEvaluator(threshold=2.5)
    
    # Real Agent with Groq LLM
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
        
        # 1. Retrieval (with optional adaptive read rejection)
        if read_filter:
            retrieval_results = read_filter.retrieve_filtered(bank, query=q_vec.tolist(), top_k=3)
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
            demos.append(Demonstration(
                query=demo_q, 
                execution=res.record.trajectory_text,
                memory_id=res.record.id
            ))
            # Track retrieval count
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
            pred, raw_output = agent.act(query=task_query, demonstrations=demos, temperature=0.0)
        
        # Calculate S_out for correlation metric (against top retrieved demo)
        if demos and pred is not None:
            # simple scalar distance similarity for regagent S_out = 1 / (1 + |pred - demo_val|)
            import re
            demo_match = re.search(r"boxed\{([+-]?\d*\.?\d+)\}", demos[0].execution)
            demo_val = float(demo_match.group(1)) if demo_match else 0.0
            s_out = 1.0 / (1.0 + abs(pred - demo_val))
            metrics["s_out"].append(s_out)
        else:
            metrics["s_out"].append(0.0)
            
        # 3. Evaluation
        eval_result = evaluator.evaluate(query=q_vec, trajectory=raw_output, ground_truth=gt)
        metrics["success_flags"].append(1 if eval_result.passed else 0)
        utility_score = 1.0 if eval_result.passed else 0.0
        metrics["utilities"].append(utility_score)
        
        if read_filter:
            read_filter.update_agent_utility(utility_score)
            
        # 4. Utility Update (Supervision)
        for res in retrieval_results:
            bank.update_utility(res.record.id, utility_score, step=t)
            
        # 5. Addition Gating
        should_add = False
        if addition_policy_name == "add_all":
            should_add = True
        elif addition_policy_name != "fixed":
            should_add = addition_policy.should_add(
                query=q_vec, trajectory=raw_output, evaluation_result=bool(utility_score), ground_truth=gt
            )
            
        if should_add and not is_error_free_twin:
            bank.add(ExperienceRecord(
                id=f"step_{t}",
                query_key=q_vec.tolist(),
                trajectory_text=raw_output,
                metadata={"ground_truth": gt},
                entry_step=t
            ))
            
        # 6. Deletion Gating
        if deletion_policy_name != "none":
            to_delete = deletion_policy.get_eviction_candidates(bank, current_step=t)
            if to_delete:
                bank.delete_many(to_delete)
                metrics["deleted_count"] += len(to_delete)
                
        metrics["memory_size"].append(bank.size())

        if not is_error_free_twin:

            time.sleep(1)
        if not is_error_free_twin:
            time.sleep(1)
        
    end_time = time.time()
    
    sr = compute_regression_success_rate(metrics["utilities"], [1.0]*len(x_test), threshold=0.1)
    r_ef = compute_experience_following_correlation(metrics["s_in"], metrics["s_out"])
    
    logger.info(f"  -> SR: {sr*100:.1f}%")
    logger.info(f"  -> r_EF: {r_ef:.4f}")
    logger.info(f"  -> Final Mem Size: {metrics['memory_size'][-1]}")
    logger.info(f"  -> Deleted: {metrics['deleted_count']}")
    if use_adaptive_read:
        logger.info(f"  -> Read Rejected: {metrics['read_rejected']}")
        
    return {
        "condition": condition_name,
        "success_rate": sr,
        "r_ef": r_ef,
        "final_mem_size": metrics["memory_size"][-1],
        "deleted_count": metrics["deleted_count"],
        "read_rejected": metrics["read_rejected"],
        "latency_sec": end_time - start_time,
        "raw_utilities": metrics["utilities"],
    }

def main():
    os.environ["LLM_PROVIDER"] = "groq"
    out_dir = Path("evaluation/pilot_groq_v2")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Config
    seed = 42
    init_size = 20
    test_size = 30
    
    # Datasets
    x_init, y_init, w = generate_regagent_dataset(seed, init_size)
    x_test, y_test, _ = generate_regagent_dataset(seed + 1, test_size)
    
    config = {
        "env": "reg_agent",
        "init_mem_size": init_size,
        "test_stream_size": test_size,
        "seed": seed,
        "model": "openai/gpt-oss-120b",
    }
    
    with open(out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
        
    results = []
    
    # Error-free twin for Error Propagation baseline
    res_ef = run_pilot_condition("Error-Free Twin", x_init, y_init, x_test, y_test, is_error_free_twin=True)
    
    # A. Fixed
    res_a = run_pilot_condition("A. Fixed", x_init, y_init, x_test, y_test, "fixed", "none")
    results.append(res_a)
    
    # B. Add-All
    res_b = run_pilot_condition("B. Add-All", x_init, y_init, x_test, y_test, "add_all", "none")
    results.append(res_b)
    
    # C. Selective addition
    res_c = run_pilot_condition("C. Strict Addition", x_init, y_init, x_test, y_test, "strict", "none")
    results.append(res_c)
    
    # D. Selective addition + deletion
    res_d = run_pilot_condition("D. Strict + History Deletion", x_init, y_init, x_test, y_test, "strict", "history")
    results.append(res_d)
    
    # E. Adaptive Read Rejection
    res_e = run_pilot_condition("E. Adaptive Read Rejection", x_init, y_init, x_test, y_test, "strict", "history", use_adaptive_read=True)
    results.append(res_e)
    
    # Compute error propagation
    for r in results:
        delta_ep = compute_error_propagation_gap(r["success_rate"], res_ef["success_rate"])
        r["error_propagation_gap"] = delta_ep
        del r["raw_utilities"] # keep json clean
        
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
        
    print(f"\nPilot complete. Results written to {out_dir}/")

if __name__ == "__main__":
    main()

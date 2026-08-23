"""Unit tests for Evaluation Framework, Metrics, Evaluators, Leakage Safeguards, and Runner.

Validates compliance with research/RESEARCH_SPEC.md.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import tempfile
import numpy as np
import pytest

from src.evaluation.config import (
    AddAllConfig,
    BaselineConfig,
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
    StrictDeletionConfig,
)
from src.evaluation.evaluator import (
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


# =====================================================================
# 1. Metrics Unit Tests
# =====================================================================

class TestMetrics:
    def test_compute_accuracy(self):
        # Exact string match with normalization
        preds = ["DDoS-ICMP_Flood", " BenignTraffic ", "DDoS-UDP_Flood"]
        gts = ["ddos-icmp_flood", "BenignTraffic", "DDoS-TCP_Flood"]
        acc = compute_accuracy(preds, gts)
        assert pytest.approx(acc) == 2 / 3

        # Empty predictions
        assert compute_accuracy([], []) == 0.0

        # Length mismatch
        with pytest.raises(ValueError, match="Length mismatch"):
            compute_accuracy([1, 2], [1])

    def test_compute_regression_success_rate(self):
        preds = [10.0, 10.8, 11.2, 12.5]
        gts = [10.0, 10.0, 10.0, 10.0]
        # threshold 1.0: diffs are [0.0, 0.8, 1.2, 2.5] -> 2 pass out of 4
        sr = compute_regression_success_rate(preds, gts, threshold=1.0)
        assert pytest.approx(sr) == 0.5

        # Custom threshold 1.5 -> 3 pass
        sr_15 = compute_regression_success_rate(preds, gts, threshold=1.5)
        assert pytest.approx(sr_15) == 0.75

        # Empty
        assert compute_regression_success_rate([], []) == 0.0

    def test_compute_experience_following_correlation(self):
        # Perfect positive correlation (r = 1.0)
        x = [0.1, 0.2, 0.3, 0.4, 0.5]
        y = [1.0, 2.0, 3.0, 4.0, 5.0]
        r = compute_experience_following_correlation(x, y)
        assert pytest.approx(r, abs=1e-5) == 1.0

        # Perfect negative correlation (r = -1.0)
        y_neg = [5.0, 4.0, 3.0, 2.0, 1.0]
        r_neg = compute_experience_following_correlation(x, y_neg)
        assert pytest.approx(r_neg, abs=1e-5) == -1.0

        # Uncorrelated or orthogonal
        x_uncorr = [1.0, 0.0, -1.0, 0.0]
        y_uncorr = [0.0, 1.0, 0.0, -1.0]
        r_uncorr = compute_experience_following_correlation(x_uncorr, y_uncorr)
        assert pytest.approx(r_uncorr, abs=1e-5) == 0.0

        # Zero variance edge case (constant sequence) -> returns 0.0 without ZeroDivisionError
        x_const = [1.0, 1.0, 1.0, 1.0]
        y_norm = [1.0, 2.0, 3.0, 4.0]
        assert compute_experience_following_correlation(x_const, y_norm) == 0.0
        assert compute_experience_following_correlation(y_norm, x_const) == 0.0

        # Less than 2 samples -> 0.0
        assert compute_experience_following_correlation([1.0], [2.0]) == 0.0
        assert compute_experience_following_correlation([], []) == 0.0

        # Length mismatch raises ValueError
        with pytest.raises(ValueError):
            compute_experience_following_correlation([1.0, 2.0], [1.0])

    def test_compute_error_propagation_gap(self):
        # Scalar gap: Delta_EP = Metric(EF) - Metric(Actual)
        gap_scalar = compute_error_propagation_gap(actual_metrics=0.65, error_free_metrics=0.80)
        assert pytest.approx(gap_scalar) == 0.15

        # Sequence gap
        act_seq = [0.5, 0.6, 0.7]
        ef_seq = [0.6, 0.8, 0.9]
        gap_seq = compute_error_propagation_gap(act_seq, ef_seq)
        assert [pytest.approx(g) for g in gap_seq] == [0.1, 0.2, 0.2]

    def test_compute_memory_retention_ratio(self):
        # M(t) / (N_0 + total_added) = 80 / (100 + 100) = 0.4
        ratio = compute_memory_retention_ratio(current_mem_size=80, total_added=100, initial_mem_size=100)
        assert pytest.approx(ratio) == 0.4

        # Zero total capacity attempted
        assert compute_memory_retention_ratio(0, 0, 0) == 1.0

    def test_compute_l2_error(self):
        preds = [[1.0, 2.0], [3.0, 4.0]]
        gts = [[1.0, 2.0], [0.0, 0.0]]  # dists: 0.0 and 5.0 -> mean 2.5
        l2 = compute_l2_error(preds, gts)
        assert pytest.approx(l2) == 2.5

    def test_compute_error_replication_rate(self):
        # Step 0: ret_err=True, act_err=True, s_out=0.9 -> Replicated
        # Step 1: ret_err=True, act_err=False, s_out=0.9 -> Not Replicated
        # Step 2: ret_err=False, act_err=True, s_out=0.9 -> Condition not met
        # ERR = 1 / 2 = 0.5
        act_errs = [True, False, True]
        ret_errs = [True, True, False]
        s_outs = [0.9, 0.9, 0.9]
        err_rate = compute_error_replication_rate(act_errs, ret_errs, s_outs, mimic_threshold=0.8)
        assert pytest.approx(err_rate) == 0.5

    def test_similarity_helpers(self):
        # Cosine similarity
        v1 = [1.0, 0.0]
        v2 = [0.0, 1.0]
        assert pytest.approx(compute_cosine_similarity(v1, v2)) == 0.0
        assert pytest.approx(compute_cosine_similarity(v1, v1)) == 1.0

        # RBF similarity
        assert pytest.approx(compute_rbf_similarity(2.0, 2.0)) == 1.0
        assert compute_rbf_similarity(2.0, 5.0) < 1.0


# =====================================================================
# 2. Evaluators Unit Tests
# =====================================================================

class TestEvaluators:
    def test_parse_regagent_prediction(self):
        assert parse_regagent_prediction("Guess: boxed{12.34}") == 12.34
        assert parse_regagent_prediction("Guess: \\boxed{-5.67}") == -5.67
        assert parse_regagent_prediction("boxed{0.0}") == 0.0
        assert parse_regagent_prediction("Guess: 42.1") == 42.1
        assert parse_regagent_prediction(3.1415) == 3.1415
        assert parse_regagent_prediction("unparseable garbage text") is None

    def test_parse_ciciot_prediction(self):
        traj = "ANALYSIS: High packet count.\nANSWER: DDoS-ICMP_Flood"
        assert parse_ciciot_prediction(traj) == "DDoS-ICMP_Flood"

    def test_parse_judge_decision(self):
        # Positive signals
        assert parse_judge_decision("CORRECT\nThe answer matches.")[0] is True
        assert parse_judge_decision("yes\nTrajectory stays on road.")[0] is True

        # Negative signals
        assert parse_judge_decision("INCORRECT\nLabel mismatch.")[0] is False
        assert parse_judge_decision("no\nCollision detected.")[0] is False

        # Edge cases
        assert parse_judge_decision("")[0] is False

    def test_strict_oracle_evaluator_regression(self):
        evaluator = RegAgentStrictEvaluator(threshold=1.0)

        # Within threshold <= 1.0
        res_pass = evaluator.evaluate(query=[1, 2], trajectory="Guess: boxed{4.5}", ground_truth=4.0)
        assert res_pass.passed is True
        assert res_pass.score == 1.0
        assert pytest.approx(res_pass.error_magnitude) == 0.5

        # Outside threshold > 1.0
        res_fail = evaluator.evaluate(query=[1, 2], trajectory="Guess: boxed{6.5}", ground_truth=4.0)
        assert res_fail.passed is False
        assert res_fail.score == 0.0
        assert pytest.approx(res_fail.error_magnitude) == 2.5

    def test_strict_oracle_evaluator_classification(self):
        evaluator = CICIOTStrictEvaluator()
        res_pass = evaluator.evaluate(
            query="packet flow",
            trajectory="ANALYSIS: Flooding.\nANSWER: DDoS-UDP_Flood",
            ground_truth="DDoS-UDP_Flood",
        )
        assert res_pass.passed is True

        res_fail = evaluator.evaluate(
            query="packet flow",
            trajectory="ANALYSIS: Flooding.\nANSWER: BenignTraffic",
            ground_truth="DDoS-UDP_Flood",
        )
        assert res_fail.passed is False

    def test_coarse_evaluator_thresholds(self):
        # C1 threshold is 1.6
        c1_eval = RegAgentCoarseEvaluator(level="C1")
        # diff 1.5 passes C1 (<= 1.6) but fails Strict (<= 1.0)
        res_c1 = c1_eval.evaluate(query=None, trajectory=5.5, ground_truth=4.0)
        assert res_c1.passed is True

        # C2 threshold is 1.4 -> diff 1.5 fails C2
        c2_eval = RegAgentCoarseEvaluator(level="C2")
        res_c2 = c2_eval.evaluate(query=None, trajectory=5.5, ground_truth=4.0)
        assert res_c2.passed is False

        # C3 threshold is 1.2 -> diff 1.3 fails C3
        c3_eval = RegAgentCoarseEvaluator(level="C3")
        res_c3 = c3_eval.evaluate(query=None, trajectory=5.3, ground_truth=4.0)
        assert res_c3.passed is False

    def test_coarse_evaluator_with_mock_llm(self):
        # Mock LLM callable returning structured verdict
        def mock_llm_judge(prompt: str) -> str:
            if "DDoS-SYN_Flood" in prompt:
                return "CORRECT\nInferred gold label matches model answer."
            return "INCORRECT\nInferred gold label does not match."

        evaluator = CICIOTCoarseEvaluator(level="C1", llm_callable=mock_llm_judge)

        res_pass = evaluator.evaluate(
            query="Flow duration...",
            trajectory="ANALYSIS: SYN count high.\nANSWER: DDoS-SYN_Flood",
        )
        assert res_pass.passed is True
        assert res_pass.score == 1.0

        res_fail = evaluator.evaluate(
            query="Flow duration...",
            trajectory="ANALYSIS: SYN count high.\nANSWER: BenignTraffic",
        )
        assert res_fail.passed is False


# =====================================================================
# 3. Leakage Checker Unit Tests
# =====================================================================

class TestLeakageSafeguards:
    def test_leakage_checker_clean(self):
        checker = LeakageChecker(min_embedding_distance=1e-2)
        init_queries = [[1.0, 0.0], [0.0, 1.0]]
        test_queries = [[5.0, 5.0], [-1.0, -1.0]]

        report = checker.verify_split_isolation(
            init_queries=init_queries,
            test_queries=test_queries,
            init_embeddings=np.array(init_queries),
            test_embeddings=np.array(test_queries),
        )
        assert report.is_clean is True
        assert report.hash_collisions == 0
        assert report.exact_overlaps == 0
        assert report.distance_violations == 0

    def test_leakage_checker_detects_hash_and_exact_overlap(self):
        checker = LeakageChecker()
        init_queries = ["query_alpha", "query_beta"]
        test_queries = ["query_beta", "query_gamma"]  # "query_beta" is leaked

        report = checker.verify_split_isolation(init_queries, test_queries)
        assert report.is_clean is False
        assert report.hash_collisions == 1
        assert len(report.violations) >= 1

    def test_leakage_checker_detects_embedding_distance_violation(self):
        checker = LeakageChecker(min_embedding_distance=0.1)
        init_emb = np.array([[1.0, 1.0], [0.0, 0.0]])
        test_emb = np.array([[1.02, 1.01], [5.0, 5.0]])  # test_emb[0] is dist ~0.022 < 0.1

        report = checker.verify_split_isolation(
            init_queries=["q1", "q2"],
            test_queries=["q_test1", "q_test2"],
            init_embeddings=init_emb,
            test_embeddings=test_emb,
            min_distance=0.1,
        )
        assert report.is_clean is False
        assert report.distance_violations == 1

    def test_leakage_raise_on_error(self):
        checker = LeakageChecker()
        with pytest.raises(AssertionError, match="Data leakage assertion failed"):
            checker.verify_split_isolation(
                init_queries=["dup"],
                test_queries=["dup"],
                raise_on_leakage=True,
            )


# =====================================================================
# 4. Configuration Dataclass Unit Tests
# =====================================================================

class TestConfigDataclasses:
    def test_baseline_configs(self):
        fixed = FixedMemoryConfig(top_k=6, initial_memory_size=100)
        assert fixed.addition_policy == "fixed"
        assert fixed.deletion_policy == "none"

        add_all = AddAllConfig()
        assert add_all.addition_policy == "add_all"

        coarse = CoarseAdditionConfig(coarse_level="C2")
        assert coarse.addition_policy == "coarse"
        assert coarse.coarse_level == "C2"

        strict = StrictAdditionConfig(strict_threshold=1.0)
        assert strict.addition_policy == "strict"

    def test_deletion_configs(self):
        hist = HistoryDeletionConfig(min_retrievals=5, beta=0.5)
        assert hist.deletion_policy == "history"
        assert hist.min_retrievals == 5
        assert hist.beta == 0.5

        comb = CombinedDeletionConfig(period=200, min_retrievals=3, beta=0.7)
        assert comb.deletion_policy == "combined"
        assert comb.period == 200

        bounded = BoundedCapacityConfig(max_capacity=180)
        assert bounded.max_capacity == 180

    def test_protocol_configs(self):
        proto_a = ProtocolAConfig(stream_length=500, seeds=[42, 128])
        assert proto_a.stream_length == 500
        assert len(proto_a.seeds) == 2
        d_a = proto_a.to_dict()
        assert d_a["protocol_name"] == "Protocol_A_Long_Term_Growth"

        bench = BenchmarkConfig(environment="reg_agent")
        assert bench.benchmark_name == "agent_memory_management_benchmark"
        assert "A" in bench.active_protocols


# =====================================================================
# 5. Experiment Runner Unit Tests
# =====================================================================

class MockMemoryBank:
    """Mock memory bank for testing ExperimentRunner."""
    def __init__(self):
        self.records = []
        self._id = 0

    def size(self) -> int:
        return len(self.records)

    def retrieve(self, query, top_k: int = 6):
        return self.records[:top_k]

    def add(self, query, trajectory, ground_truth=None) -> bool:
        self.records.append((query, trajectory, ground_truth))
        return True

    def update_utility(self, retrieved_records, utility_score: float) -> None:
        pass

    def prune(self, current_step: int):
        return []

    def get_all_records(self):
        return self.records


class TestExperimentRunner:
    def test_runner_stream_execution(self):
        queries = [np.array([1.0, 2.0]), np.array([2.0, 3.0]), np.array([3.0, 4.0])]
        ground_truths = [5.0, 8.0, 11.0]

        # Mock agent that returns ground truth with small noise
        def mock_agent(query, demos):
            if demos:
                return "Guess: boxed{8.1}"
            return "Guess: boxed{5.1}"

        mem = MockMemoryBank()
        # Seed initial memory
        mem.add(np.array([0.0, 0.0]), "Guess: boxed{0.0}", ground_truth=0.0)

        runner = ExperimentRunner(top_k=2, track_error_free_twin=True)
        res = runner.run_stream(
            experiment_name="test_run",
            queries=queries,
            ground_truths=ground_truths,
            agent_fn=mock_agent,
            memory_bank=mem,
            seed=42,
            addition_gating=False,
        )

        assert res.total_steps == 3
        assert len(res.step_metrics) == 3
        assert res.final_memory_size == 4  # 1 initial + 3 added
        assert res.final_success_rate > 0.0
        assert isinstance(res.pearson_r_ef, float)
        assert res.mean_error_propagation_gap is not None

    def test_runner_serialization(self):
        queries = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]
        ground_truths = [1.0, 1.0]

        def mock_agent(q, demos):
            return "Guess: boxed{1.0}"

        mem = MockMemoryBank()
        runner = ExperimentRunner()
        res = runner.run_stream(
            experiment_name="test_serial",
            queries=queries,
            ground_truths=ground_truths,
            agent_fn=mock_agent,
            memory_bank=mem,
            seed=42,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "res.json"
            csv_path = Path(tmpdir) / "res.csv"

            res.to_json(json_path)
            res.to_csv(csv_path)

            assert json_path.exists()
            assert csv_path.exists()

            # Verify JSON readable
            with open(json_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
                assert loaded["experiment_name"] == "test_serial"
                assert len(loaded["step_metrics"]) == 2

            # Verify CSV readable and has lines
            with open(csv_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
                assert len(lines) == 3  # Header + 2 rows

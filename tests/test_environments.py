"""Pytest suite for RegAgent and CIC-IoT environments."""

import numpy as np
import pytest

from src.environments.base import TaskQuery, TaskResult
from src.environments.reg_agent_env import RegAgentEnvironment, DEFAULT_W, VALID_MUS
from src.environments.ciciot_env import (
    CICIOTEnvironment,
    CICIOT_CLASSES,
    CICIOT_CONTINUOUS_FEATURES,
    CICIOT_DISCRETE_FEATURES,
    canonical_label,
)


class TestRegAgentEnvironment:
    """Test suite for 6D Synthetic Gaussian Regression Environment."""

    def test_initialization_and_weights(self):
        env_default = RegAgentEnvironment()
        assert np.allclose(env_default.w, DEFAULT_W)
        assert env_default.dim == 6

        custom_w = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        env_custom = RegAgentEnvironment(w=custom_w)
        assert np.allclose(env_custom.w, custom_w)

        with pytest.raises(ValueError):
            RegAgentEnvironment(w=[1.0, 2.0])

    def test_sample_initial_memory(self):
        env = RegAgentEnvironment()
        samples = env.sample_initial_memory(n_samples=50, seed=42)
        assert len(samples) == 50
        for s in samples:
            assert isinstance(s, TaskQuery)
            assert len(s.query_vector) == 6
            assert len(s.raw_input) == 6
            assert isinstance(s.ground_truth, float)
            assert s.metadata["mu"] in VALID_MUS
            assert -1.0 <= s.metadata["noise"] <= 1.0

    def test_sample_stream_standard_and_shift(self):
        env = RegAgentEnvironment()
        stream = env.sample_stream(n_samples=100, seed=128, cluster_shift=False)
        assert len(stream) == 100

        shifted_stream = env.sample_stream(n_samples=90, seed=128, cluster_shift=True)
        assert len(shifted_stream) == 90
        first_30 = shifted_stream[:30]
        second_30 = shifted_stream[30:60]
        third_30 = shifted_stream[60:90]
        assert all(q.metadata["mu"] == -0.5 for q in first_30)
        assert all(q.metadata["mu"] == 0.0 for q in second_30)
        assert all(q.metadata["mu"] == 0.5 for q in third_30)

    def test_evaluate_success_and_failure(self):
        env = RegAgentEnvironment(success_threshold=1.0)
        query = TaskQuery(
            query_id="test_01",
            query_vector=[1.0] * 6,
            ground_truth=5.0,
        )

        res_succ = env.evaluate(query, prediction=4.5)
        assert res_succ.is_success is True
        assert res_succ.score == 1.0
        assert pytest.approx(res_succ.error) == 0.5

        res_fail = env.evaluate(query, prediction=6.5)
        assert res_fail.is_success is False
        assert res_fail.score == 0.0
        assert pytest.approx(res_fail.error) == 1.5

        res_bound = env.evaluate(query, prediction=6.0)
        assert res_bound.is_success is True
        assert res_bound.score == 1.0

        res_unparse = env.evaluate(query, prediction="not_a_number")
        assert res_unparse.is_success is False
        assert res_unparse.score == 0.0

    def test_compute_input_similarity(self):
        env = RegAgentEnvironment()
        v1 = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        v2 = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        v3 = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
        v4 = np.array([-1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        v_zero = np.zeros(6)

        assert pytest.approx(env.compute_input_similarity(v1, v2)) == 1.0
        assert pytest.approx(env.compute_input_similarity(v1, v3)) == 0.0
        assert pytest.approx(env.compute_input_similarity(v1, v4)) == -1.0
        assert pytest.approx(env.compute_input_similarity(v1, v_zero)) == 0.0

    def test_compute_output_similarity(self):
        env = RegAgentEnvironment(gamma=1.0)
        assert pytest.approx(env.compute_output_similarity(5.0, 5.0)) == 1.0
        assert pytest.approx(env.compute_output_similarity(5.0, 6.0), rel=1e-3) == np.exp(-1.0)
        assert pytest.approx(env.compute_output_similarity("boxed{5.0}", "Guess: boxed{6.0}"), rel=1e-3) == np.exp(-1.0)
        assert pytest.approx(env.compute_output_similarity("invalid", 5.0)) == 0.0


class TestCICIOTEnvironment:
    """Test suite for 8-class Tabular IoT Network Traffic Environment."""

    def test_initialization_and_classes(self):
        env = CICIOTEnvironment()
        assert len(env.classes) == 8
        assert "DDoS-ICMP_Flood" in env.classes
        assert "BenignTraffic" in env.classes
        assert len(env.continuous_features) == len(CICIOT_CONTINUOUS_FEATURES)
        assert len(env.discrete_features) == len(CICIOT_DISCRETE_FEATURES)

    def test_sample_initial_and_stream(self):
        env = CICIOTEnvironment()
        init_mems = env.sample_initial_memory(n_samples=20, seed=42)
        assert len(init_mems) == 20
        for m in init_mems:
            assert m.ground_truth in CICIOT_CLASSES
            assert len(m.query_vector) == len(env.all_features)
            assert "Rate" in m.features
            assert "Protocol_Type" in m.features

        stream = env.sample_stream(n_samples=30, seed=99)
        assert len(stream) == 30

    def test_all_classes_generation(self):
        env = CICIOTEnvironment()
        for cls_name in CICIOT_CLASSES:
            q = env.generate_single_query(f"q_{cls_name}", traffic_type=cls_name, rng=np.random.RandomState(42))
            assert q.ground_truth == cls_name
            assert len(q.query_vector) == len(env.all_features)

    def test_evaluate_accuracy(self):
        env = CICIOTEnvironment()
        q = TaskQuery(
            query_id="iot_01",
            ground_truth="DDoS-SYN_Flood",
        )

        res_exact = env.evaluate(q, prediction="DDoS-SYN_Flood")
        assert res_exact.is_success is True
        assert res_exact.score == 1.0

        res_norm = env.evaluate(q, prediction="['ddos-syn_flood']")
        assert res_norm.is_success is True
        assert res_norm.score == 1.0

        res_wrong = env.evaluate(q, prediction="BenignTraffic")
        assert res_wrong.is_success is False
        assert res_wrong.score == 0.0

    def test_feature_distance_and_similarity(self):
        env = CICIOTEnvironment()
        q1 = env.generate_single_query("q1", traffic_type="DDoS-ICMP_Flood", rng=np.random.RandomState(1))
        q2 = env.generate_single_query("q2", traffic_type="DDoS-ICMP_Flood", rng=np.random.RandomState(1))
        q3 = env.generate_single_query("q3", traffic_type="BenignTraffic", rng=np.random.RandomState(2))

        assert pytest.approx(env.compute_input_similarity(q1, q2)) == 1.0

        sim_diff = env.compute_input_similarity(q1, q3)
        assert 0.0 <= sim_diff < 0.9

    def test_output_similarity(self):
        env = CICIOTEnvironment()
        out1 = "ANALYSIS: ICMP Flood detected.\nANSWER: DDoS-ICMP_Flood"
        out2 = "ANSWER: DDoS-ICMP_Flood"
        out3 = "ANSWER: BenignTraffic"

        assert env.compute_output_similarity(out1, out2) == 1.0
        assert env.compute_output_similarity(out1, out3) == 0.0

    def test_canonical_label(self):
        assert canonical_label("DDoS-SYN_Flood") == "ddossynflood"
        assert canonical_label("['ddos_syn_flood']") == "ddossynflood"
        assert canonical_label("Benign Traffic") == "benigntraffic"
        assert canonical_label("") == ""

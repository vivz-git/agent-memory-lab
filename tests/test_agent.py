"""Pytest suite for Agent architectures, prompts, parsing, and orchestrator."""

import numpy as np
import pytest

from src.environments.base import TaskQuery
from src.environments.reg_agent_env import RegAgentEnvironment
from src.environments.ciciot_env import CICIOTEnvironment
from src.agent.core import (
    Demonstration,
    MockLLMClient,
    OpenAILLMClient,
    RegAgent,
    CICIOTAgent,
)
from src.agent.prompts import (
    format_regagent_prompt,
    format_ciciot_prompt,
    format_ciciot_features_block,
)
from src.agent.orchestrator import (
    AgentOrchestrator,
    AdaptiveReadFilter,
    SimpleEpisodicMemoryBank,
)


class TestAgentPromptsAndParsing:
    """Test prompt formatting and robust output parsing."""

    def test_regagent_prompt_formatting(self):
        q = TaskQuery(
            query_id="q1",
            raw_input=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            ground_truth=1.5,
        )
        d1 = Demonstration(
            query=TaskQuery(query_id="d1", raw_input=[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            execution="boxed{0.8000}",
        )
        prompt = format_regagent_prompt(q, [d1])

        assert "6-dimensional input vector x" in prompt
        assert "Demonstrations (K=1):" in prompt
        assert "Input: [1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000] Guess: boxed{0.8000}" in prompt
        assert "Input: [0.1000, 0.2000, 0.3000, 0.4000, 0.5000, 0.6000] Guess:" in prompt

    def test_regagent_prompt_zero_demos(self):
        q = TaskQuery(
            query_id="q0",
            raw_input=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            ground_truth=1.5,
        )
        prompt = format_regagent_prompt(q, [])
        assert "Demonstrations (K=0):" in prompt
        assert "None" in prompt

    def test_regagent_output_parsing(self):
        agent = RegAgent()

        assert agent.parse_output("Guess: boxed{3.1415}") == 3.1415
        assert agent.parse_output(r"Guess: \boxed{-2.5000}") == -2.5
        assert agent.parse_output("boxed{42.0}") == 42.0
        assert agent.parse_output("Guess: 1.234") == 1.234
        assert agent.parse_output("The estimated value is 9.81") == 9.81
        assert agent.parse_output("unparseable garbage text") is None
        assert agent.parse_output("") is None

    def test_ciciot_prompt_formatting(self):
        env = CICIOTEnvironment()
        q = env.generate_single_query("q_iot", traffic_type="DDoS-ICMP_Flood", rng=np.random.RandomState(42))
        d = Demonstration(
            query=env.generate_single_query("d_iot", traffic_type="BenignTraffic", rng=np.random.RandomState(43)),
            execution="ANALYSIS: Standard benign flow.\nANSWER: BenignTraffic",
        )
        prompt = format_ciciot_prompt(q, [d])

        assert "Demonstrations (K=1):" in prompt
        assert "Flow duration" in prompt
        assert "DDoS-ICMP_Flood" in prompt
        assert "BenignTraffic" in prompt

    def test_ciciot_prompt_zero_demos(self):
        env = CICIOTEnvironment()
        q = env.generate_single_query("q_iot", traffic_type="DDoS-ICMP_Flood", rng=np.random.RandomState(42))
        prompt = format_ciciot_prompt(q, [])
        assert "Required output format:" in prompt
        assert "### Traffic Types:" in prompt

    def test_ciciot_output_parsing(self):
        agent = CICIOTAgent()

        out1 = "ANALYSIS: High rate SYN packets observed.\nANSWER: DDoS-SYN_Flood"
        assert agent.parse_output(out1) == "DDoS-SYN_Flood"

        out2 = "ANSWER: ['DDoS-HTTP_Flood']"
        assert agent.parse_output(out2) == "DDoS-HTTP_Flood"

        out3 = "ANSWER: benign_traffic"
        assert agent.parse_output(out3) == "BenignTraffic"

        out4 = "No structured prefix, but it mentions DDoS-UDP_Flood clearly."
        assert agent.parse_output(out4) == "DDoS-UDP_Flood"

        assert agent.parse_output("") == "Unknown"


class TestMockLLMClient:
    """Test deterministic mock LLM client modes."""

    def test_rule_based_regagent_inference(self):
        client = MockLLMClient(mode="rule_based")
        q = TaskQuery(
            query_id="q1",
            raw_input=[1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        )
        d1 = Demonstration(query=TaskQuery(query_id="d1", raw_input=[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]), execution="boxed{1.0}")
        d2 = Demonstration(query=TaskQuery(query_id="d2", raw_input=[2.0, 0.0, 0.0, 0.0, 0.0, 0.0]), execution="boxed{2.0}")
        d3 = Demonstration(query=TaskQuery(query_id="d3", raw_input=[0.5, 0.0, 0.0, 0.0, 0.0, 0.0]), execution="boxed{0.5}")

        prompt = format_regagent_prompt(q, [d1, d2, d3])
        response = client.complete(prompt)
        assert "boxed{" in response

        agent = RegAgent(llm_client=client)
        pred = agent.parse_output(response)
        assert pred is not None
        assert abs(pred - 1.0) < 0.2

    def test_rule_based_ciciot_inference(self):
        client = MockLLMClient(mode="rule_based")
        env = CICIOTEnvironment()
        q = env.generate_single_query("q_icmp", traffic_type="DDoS-ICMP_Flood", rng=np.random.RandomState(42))

        prompt = format_ciciot_prompt(q, [])
        response = client.complete(prompt)
        assert "ANSWER: DDoS-ICMP_Flood" in response

    def test_mimic_mode(self):
        client = MockLLMClient(mode="demonstration_mimic")
        q = TaskQuery(query_id="q1", raw_input=[1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        d1 = Demonstration(query=TaskQuery(query_id="d1", raw_input=[0.9, 0.0, 0.0, 0.0, 0.0, 0.0]), execution="boxed{99.9}")
        prompt = format_regagent_prompt(q, [d1])
        response = client.complete(prompt)
        assert "boxed{99.9000}" in response

    def test_custom_responder_and_fallback(self):
        client_custom = MockLLMClient(mode="custom", custom_responder=lambda p: "CUSTOM_OUT")
        assert client_custom.complete("test prompt") == "CUSTOM_OUT"

        client_fixed = MockLLMClient(mode="fixed", fixed_response="FIXED_OUT")
        assert client_fixed.complete("test prompt") == "FIXED_OUT"

        openai_fallback = OpenAILLMClient(api_key=None, fallback_to_mock=True)
        resp = openai_fallback.complete("6-dimensional input vector x: [1,0,0,0,0,0]")
        assert "boxed{" in resp


class TestAgentOrchestrator:
    """Test full agent orchestration loop and memory update policies."""

    def test_fixed_addition_lifecycle(self):
        env = RegAgentEnvironment()
        agent = RegAgent(llm_client=MockLLMClient(mode="rule_based"))
        orchestrator = AgentOrchestrator(
            agent=agent,
            env=env,
            addition_policy=lambda q, e, res: False,
            top_k=6,
        )

        init_queries = env.sample_initial_memory(n_samples=20, seed=42)
        orchestrator.populate_initial_memory(init_queries)
        assert len(orchestrator.memory_bank) == 20

        stream = env.sample_stream(n_samples=10, seed=101)
        results = orchestrator.run_stream(stream)

        assert len(results) == 10
        assert len(orchestrator.memory_bank) == 20
        assert all(not r.added_to_memory for r in results)
        assert all(len(r.retrieved_demonstrations) <= 6 for r in results)

    def test_strict_addition_and_history_deletion(self):
        env = RegAgentEnvironment()
        agent = RegAgent(llm_client=MockLLMClient(mode="rule_based"))

        def strict_addition(q, e, res):
            return res.is_success

        def history_deletion(memory_bank, step):
            evicted = []
            if hasattr(memory_bank, "records"):
                for mid, rec in memory_bank.records.items():
                    if rec["retrieval_count"] >= 2 and rec["mean_utility"] < 0.5:
                        evicted.append(mid)
            return evicted

        orchestrator = AgentOrchestrator(
            agent=agent,
            env=env,
            addition_policy=strict_addition,
            deletion_policy=history_deletion,
            top_k=4,
        )

        init_queries = env.sample_initial_memory(n_samples=15, seed=42)
        orchestrator.populate_initial_memory(init_queries)

        stream = env.sample_stream(n_samples=20, seed=202)
        results = orchestrator.run_stream(stream)

        assert len(results) == 20
        r = orchestrator.compute_experience_following_pearson_r()
        assert isinstance(r, float)

    def test_periodic_deletion_policy(self):
        env = RegAgentEnvironment()
        agent = RegAgent(llm_client=MockLLMClient(mode="rule_based"))

        def periodic_deletion(memory_bank, step):
            if step % 5 == 0:
                # Evict records added in step 0 that were never retrieved
                evicted = [
                    mid for mid, rec in memory_bank.records.items()
                    if rec["retrieval_count"] == 0 and rec["step_added"] == 0
                ]
                return evicted
            return []

        orchestrator = AgentOrchestrator(
            agent=agent,
            env=env,
            addition_policy=lambda q, e, res: True,
            deletion_policy=periodic_deletion,
            top_k=2,
        )

        init_queries = env.sample_initial_memory(n_samples=10, seed=42)
        orchestrator.populate_initial_memory(init_queries)

        stream = env.sample_stream(n_samples=10, seed=303)
        results = orchestrator.run_stream(stream)
        assert len(results) == 10
        assert any(len(r.deleted_memory_ids) > 0 for r in results)

    def test_adaptive_read_filter(self):
        read_filter = AdaptiveReadFilter(utility_threshold=0.5, min_retrievals=2)

        d_good = Demonstration(
            query=TaskQuery(query_id="g1"),
            execution="boxed{1.0}",
            score=0.9,
            metadata={"retrieval_count": 5, "mean_utility": 0.9},
        )
        d_bad = Demonstration(
            query=TaskQuery(query_id="b1"),
            execution="boxed{-10.0}",
            score=0.2,
            metadata={"retrieval_count": 3, "mean_utility": 0.2},
        )
        d_new = Demonstration(
            query=TaskQuery(query_id="n1"),
            execution="boxed{0.0}",
            score=0.0,
            metadata={"retrieval_count": 1, "mean_utility": 0.0},
        )

        q = TaskQuery(query_id="curr")
        filtered = read_filter.filter(q, [d_good, d_bad, d_new])

        assert d_good in filtered
        assert d_bad not in filtered
        assert d_new in filtered

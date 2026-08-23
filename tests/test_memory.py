"""Comprehensive test suite for experience-memory subsystem."""

import pytest
import numpy as np

from src.memory.schema import (
    ExperienceRecord,
    MemoryQuery,
    RetrievalResult,
    SimilarityMetricType,
    AdditionPolicyType,
    DeletionPolicyType,
)
from src.memory.bank import BaseMemoryBank
from src.memory.addition import (
    BaseAdditionPolicy,
    FixedAdditionPolicy,
    AddAllAdditionPolicy,
    CoarseAdditionPolicy,
    StrictAdditionPolicy,
    create_addition_policy,
)
from src.memory.deletion import (
    BaseDeletionPolicy,
    PeriodicDeletionPolicy,
    HistoryBasedDeletionPolicy,
    CombinedDeletionPolicy,
    ConstrainedCapacityDeletionPolicy,
    create_deletion_policy,
)
from src.memory.adaptive_retrieval import AdaptiveReadFilter


# =====================================================================
# 1. ExperienceRecord Schema Tests
# =====================================================================

class TestExperienceRecord:
    """Test ExperienceRecord model methods and lifecycle tracking."""

    def test_record_creation_and_defaults(self):
        record = ExperienceRecord(
            query_key=[0.1, 0.2, 0.3],
            query_text="Sample query",
            trajectory_text="Guess: boxed{42.0}",
            entry_step=10,
        )
        assert record.id is not None
        assert record.query_key == [0.1, 0.2, 0.3]
        assert record.query_text == "Sample query"
        assert record.trajectory_text == "Guess: boxed{42.0}"
        assert record.retrieval_count == 0
        assert record.utility_history == []
        assert record.mean_utility == 0.0
        assert record.entry_step == 10
        assert record.last_retrieved_step == 0

    def test_record_retrieval_and_utility_update(self):
        record = ExperienceRecord(
            query_key=[1.0, 0.0],
            trajectory_text="output 1",
            entry_step=0,
        )
        # Step 5: retrieved with downstream success 1.0
        record.record_retrieval(step=5, utility=1.0)
        assert record.retrieval_count == 1
        assert record.last_retrieved_step == 5
        assert record.retrieval_steps == [5]
        assert record.utility_history == [1.0]
        assert record.mean_utility == pytest.approx(1.0)

        # Step 12: retrieved with downstream failure 0.0
        record.record_retrieval(step=12, utility=0.0)
        assert record.retrieval_count == 2
        assert record.last_retrieved_step == 12
        assert record.retrieval_steps == [5, 12]
        assert record.utility_history == [1.0, 0.0]
        assert record.mean_utility == pytest.approx(0.5)

        # Step 20: standalone utility update 0.5
        record.update_utility(0.5, step=20)
        assert record.utility_history == [1.0, 0.0, 0.5]
        assert record.mean_utility == pytest.approx(0.5)
        assert record.last_retrieved_step == 20

    def test_retrievals_in_window(self):
        record = ExperienceRecord(
            query_key=[1.0, 2.0],
            trajectory_text="out",
        )
        for s in [50, 120, 200, 350, 480, 520]:
            record.record_retrieval(step=s)

        assert record.retrievals_in_window(0, 100) == 1       # 50
        assert record.retrievals_in_window(100, 300) == 2     # 120, 200
        assert record.retrievals_in_window(300, 500) == 2     # 350, 480
        assert record.retrievals_in_window(0, 600) == 6

    def test_to_demonstration_text(self):
        r1 = ExperienceRecord(
            query_key=[1.0, 2.0],
            query_text="Find x",
            trajectory_text="x = 3",
        )
        assert r1.to_demonstration_text() == "Input: Find x\nTrajectory: x = 3"

        r2 = ExperienceRecord(
            query_key=[1.0, 2.0],
            trajectory_text="x = 3",
        )
        assert r2.to_demonstration_text() == "Input: [1.0, 2.0]\nTrajectory: x = 3"


# =====================================================================
# 2. BaseMemoryBank Vector Indexing and Retrieval Tests
# =====================================================================

class TestBaseMemoryBank:
    """Test memory bank vector retrieval and similarity calculations."""

    def test_basic_crud_operations(self):
        bank = BaseMemoryBank(metric=SimilarityMetricType.COSINE)
        assert bank.size() == 0
        assert len(bank) == 0
        assert bank.all_records() == []

        r1 = ExperienceRecord(id="rec-1", query_key=[1.0, 0.0], trajectory_text="t1")
        r2 = ExperienceRecord(id="rec-2", query_key=[0.0, 1.0], trajectory_text="t2")
        bank.add(r1)
        bank.add(r2)

        assert bank.size() == 2
        assert bank.get("rec-1") == r1
        assert bank.get("rec-2") == r2
        assert bank.get("non-existent") is None

        # Update utility
        updated = bank.update_utility("rec-1", 0.9, step=10)
        assert updated is True
        assert bank.get("rec-1").mean_utility == pytest.approx(0.9)
        assert bank.update_utility("non-existent", 1.0) is False

        # Delete single
        assert bank.delete("rec-1") is True
        assert bank.size() == 1
        assert bank.delete("rec-1") is False

        # Delete many
        bank.add(r1)
        assert bank.delete_many(["rec-1", "rec-2", "rec-missing"]) == 2
        assert bank.size() == 0

        # Clear
        bank.add_many([r1, r2])
        assert bank.size() == 2
        bank.clear()
        assert bank.size() == 0

    def test_cosine_similarity_retrieval(self):
        bank = BaseMemoryBank(metric=SimilarityMetricType.COSINE)
        r_east = ExperienceRecord(id="east", query_key=[1.0, 0.0], trajectory_text="heading east")
        r_north = ExperienceRecord(id="north", query_key=[0.0, 1.0], trajectory_text="heading north")
        r_northeast = ExperienceRecord(id="northeast", query_key=[1.0, 1.0], trajectory_text="heading northeast")
        bank.add_many([r_east, r_north, r_northeast])

        # Query aligned with east [1.0, 0.0]
        results = bank.retrieve([1.0, 0.0], top_k=3)
        assert len(results) == 3
        assert results[0].record.id == "east"
        assert results[0].score == pytest.approx(1.0, abs=1e-5)
        assert results[1].record.id == "northeast"
        assert results[1].score == pytest.approx(1.0 / np.sqrt(2), abs=1e-5)
        assert results[2].record.id == "north"
        assert results[2].score == pytest.approx(0.0, abs=1e-5)

        # Test filter_ids
        filtered = bank.retrieve([1.0, 0.0], top_k=2, filter_ids={"east"})
        assert len(filtered) == 2
        assert filtered[0].record.id == "northeast"
        assert filtered[1].record.id == "north"

    def test_rbf_kernel_similarity_retrieval(self):
        bank = BaseMemoryBank(metric=SimilarityMetricType.RBF, rbf_gamma=1.0)
        r0 = ExperienceRecord(id="origin", query_key=[0.0, 0.0], trajectory_text="at origin")
        r1 = ExperienceRecord(id="near", query_key=[1.0, 0.0], trajectory_text="near")
        r2 = ExperienceRecord(id="far", query_key=[3.0, 4.0], trajectory_text="far")
        bank.add_many([r0, r1, r2])

        # Query [0.0, 0.0]
        results = bank.retrieve([0.0, 0.0], top_k=3)
        assert results[0].record.id == "origin"
        assert results[0].score == pytest.approx(np.exp(0.0), abs=1e-5)  # 1.0

        assert results[1].record.id == "near"
        # distance^2 = 1.0 -> score = exp(-1.0)
        assert results[1].score == pytest.approx(np.exp(-1.0), abs=1e-5)

        assert results[2].record.id == "far"
        # distance^2 = 3^2 + 4^2 = 25.0 -> score = exp(-25.0)
        assert results[2].score == pytest.approx(np.exp(-25.0), abs=1e-5)

    def test_relative_feature_difference_retrieval(self):
        # Continuous feature 0, discrete feature 1 (protocol: 6 for TCP, 17 for UDP)
        bank = BaseMemoryBank(
            metric=SimilarityMetricType.RELATIVE_FEATURE,
            discrete_feature_indices=[1],
        )

        r_tcp_flow = ExperienceRecord(id="tcp1", query_key=[100.0, 6.0], trajectory_text="TCP flow 100 bytes")
        r_tcp_large = ExperienceRecord(id="tcp2", query_key=[200.0, 6.0], trajectory_text="TCP flow 200 bytes")
        r_udp_flow = ExperienceRecord(id="udp1", query_key=[100.0, 17.0], trajectory_text="UDP flow 100 bytes")
        bank.add_many([r_tcp_flow, r_tcp_large, r_udp_flow])

        # Query: [100.0, 6.0]
        # Match against tcp1: continuous diff = |100-100|/100 = 0; discrete diff = 0 -> sim = 1.0
        # Match against tcp2: continuous diff = |100-200|/200 = 0.5; discrete diff = 0 -> avg_diff = 0.25 -> sim = 0.75
        # Match against udp1: continuous diff = 0; discrete diff = 1 (6 != 17) -> avg_diff = 0.5 -> sim = 0.5
        results = bank.retrieve([100.0, 6.0], top_k=3)
        assert results[0].record.id == "tcp1"
        assert results[0].score == pytest.approx(1.0, abs=1e-4)

        assert results[1].record.id == "tcp2"
        assert results[1].score == pytest.approx(0.75, abs=1e-4)

        assert results[2].record.id == "udp1"
        assert results[2].score == pytest.approx(0.50, abs=1e-4)

    def test_string_key_retrieval(self):
        bank = BaseMemoryBank(metric=SimilarityMetricType.COSINE)
        r1 = ExperienceRecord(id="doc1", query_key="calculate total revenue for patient 123", trajectory_text="code 1")
        r2 = ExperienceRecord(id="doc2", query_key="calculate average heart rate for icu stays", trajectory_text="code 2")
        bank.add_many([r1, r2])

        results = bank.retrieve("calculate total revenue for patient 123", top_k=2)
        assert results[0].record.id == "doc1"
        assert results[0].score == pytest.approx(1.0)
        assert results[1].record.id == "doc2"
        assert results[1].score < 1.0

    def test_empty_and_edge_case_retrieval(self):
        bank = BaseMemoryBank()
        # Empty bank
        assert bank.retrieve([1.0, 2.0], top_k=5) == []

        # top_k <= 0
        r = ExperienceRecord(id="r", query_key=[1.0, 0.0], trajectory_text="t")
        bank.add(r)
        assert bank.retrieve([1.0, 0.0], top_k=0) == []
        assert bank.retrieve([1.0, 0.0], top_k=-1) == []

        # top_k > size
        results = bank.retrieve([1.0, 0.0], top_k=10)
        assert len(results) == 1

        # Zero vector handling
        z1 = ExperienceRecord(id="z1", query_key=[0.0, 0.0], trajectory_text="z")
        bank.add(z1)
        zero_res = bank.retrieve([0.0, 0.0], top_k=2)
        assert len(zero_res) == 2


# =====================================================================
# 3. Memory Addition Policy Tests
# =====================================================================

class TestAdditionPolicies:
    """Test admission policies: Fixed, Add-All, Coarse, and Strict."""

    def test_fixed_addition_policy(self):
        policy = FixedAdditionPolicy()
        assert policy.policy_type == AdditionPolicyType.FIXED
        assert policy.should_add(query=[1, 2], trajectory="pred", ground_truth="pred") is False
        assert policy(query="q", trajectory="t", evaluation_result=1.0) is False

    def test_add_all_addition_policy(self):
        policy = AddAllAdditionPolicy()
        assert policy.policy_type == AdditionPolicyType.ADD_ALL
        assert policy.should_add(query=[1, 2], trajectory="pred", ground_truth="wrong") is True
        assert policy(query="q", trajectory="t", evaluation_result=0.0) is True

    def test_coarse_addition_policy_numeric_and_strings(self):
        # Error-based coarse policy (RegAgent style: error <= 1.6)
        coarse_err = CoarseAdditionPolicy(threshold=1.6, error_based=True)
        assert coarse_err.should_add(query="q", trajectory="t", evaluation_result=1.2) is True
        assert coarse_err.should_add(query="q", trajectory="t", evaluation_result=1.6) is True
        assert coarse_err.should_add(query="q", trajectory="t", evaluation_result=1.7) is False

        # Score-based coarse policy (confidence >= 0.8)
        coarse_score = CoarseAdditionPolicy(threshold=0.8, error_based=False)
        assert coarse_score.should_add(query="q", trajectory="t", evaluation_result=0.85) is True
        assert coarse_score.should_add(query="q", trajectory="t", evaluation_result=0.75) is False

        # LLM Judge string responses
        coarse_judge = CoarseAdditionPolicy()
        assert coarse_judge.should_add(query="q", trajectory="t", evaluation_result="CORRECT\nReasoning here...") is True
        assert coarse_judge.should_add(query="q", trajectory="t", evaluation_result="yes\nTrajectory adheres to rules") is True
        assert coarse_judge.should_add(query="q", trajectory="t", evaluation_result="INCORRECT\nReasoning here...") is False
        assert coarse_judge.should_add(query="q", trajectory="t", evaluation_result="no\nCollision detected") is False

    def test_strict_addition_policy(self):
        # Numeric error threshold (RegAgent: |y_hat - y| <= 1.0)
        strict_reg = StrictAdditionPolicy(error_threshold=1.0)
        assert strict_reg.should_add(query="q", trajectory=10.5, ground_truth=11.2) is True   # diff 0.7
        assert strict_reg.should_add(query="q", trajectory=10.5, ground_truth=12.0) is False  # diff 1.5

        # Exact match (CIC-IoT: label equality)
        strict_exact = StrictAdditionPolicy(exact_match=True)
        assert strict_exact.should_add(query="q", trajectory="DDoS-SYN_Flood", ground_truth="DDoS-SYN_Flood") is True
        assert strict_exact.should_add(query="q", trajectory="BenignTraffic", ground_truth="DDoS-HTTP_Flood") is False

        # Custom oracle callable
        strict_custom = StrictAdditionPolicy(oracle_fn=lambda traj, gt: traj.startswith("VALID"))
        assert strict_custom.should_add(query="q", trajectory="VALID_PATH", ground_truth=None) is True
        assert strict_custom.should_add(query="q", trajectory="INVALID_PATH", ground_truth=None) is False

    def test_addition_factory(self):
        p1 = create_addition_policy("fixed")
        assert isinstance(p1, FixedAdditionPolicy)

        p2 = create_addition_policy(AdditionPolicyType.ADD_ALL)
        assert isinstance(p2, AddAllAdditionPolicy)

        p3 = create_addition_policy("coarse", threshold=1.4, error_based=True)
        assert isinstance(p3, CoarseAdditionPolicy)
        assert p3.threshold == 1.4

        p4 = create_addition_policy("strict", error_threshold=0.5)
        assert isinstance(p4, StrictAdditionPolicy)
        assert p4.error_threshold == 0.5


# =====================================================================
# 4. Memory Deletion Policy Tests
# =====================================================================

class TestDeletionPolicies:
    """Test forgetting policies: Periodic, History-based, Combined, and Constrained Capacity."""

    def test_periodic_deletion_policy(self):
        bank = BaseMemoryBank()
        # Create 3 records
        # r1: active in window [0, 500] (retrieved at step 250 and 400)
        r1 = ExperienceRecord(id="r1", query_key=[1, 0], trajectory_text="t1", entry_step=0)
        r1.record_retrieval(step=250)
        r1.record_retrieval(step=400)

        # r2: inactive in window [0, 500] (never retrieved)
        r2 = ExperienceRecord(id="r2", query_key=[0, 1], trajectory_text="t2", entry_step=0)

        # r3: stale (retrieved at step 50, but nothing in window [500, 1000])
        r3 = ExperienceRecord(id="r3", query_key=[1, 1], trajectory_text="t3", entry_step=0)
        r3.record_retrieval(step=50)

        bank.add_many([r1, r2, r3])

        # Step 499: not periodic step -> nothing evicted
        policy = PeriodicDeletionPolicy(period=500, alpha=0)
        assert policy.get_eviction_candidates(bank, current_step=499) == []

        # Step 500: period = 500, alpha = 0. Window [0, 500].
        # r1 has 2 retrievals > 0 -> kept
        # r2 has 0 retrievals <= 0 -> candidate
        # r3 has 1 retrieval > 0 -> kept
        evicted = policy.apply(bank, current_step=500)
        assert evicted == ["r2"]
        assert bank.size() == 2
        assert bank.get("r2") is None

        # Advance to step 1000. Window [500, 1000].
        # Neither r1 nor r3 had retrievals between 500 and 1000.
        evicted_1000 = policy.apply(bank, current_step=1000)
        assert set(evicted_1000) == {"r1", "r3"}
        assert bank.size() == 0

    def test_history_based_deletion_policy(self):
        bank = BaseMemoryBank()

        # r_untested: retrieval_count = 2 (< min_retrievals=5), utility = 0.0 -> protected
        r_untested = ExperienceRecord(id="untested", query_key=[1, 0], trajectory_text="t")
        r_untested.record_retrieval(1, utility=0.0)
        r_untested.record_retrieval(2, utility=0.0)

        # r_toxic: retrieval_count = 5, utility = [0, 0, 0, 0, 1] -> mean 0.2 (<= beta 0.5) -> evict
        r_toxic = ExperienceRecord(id="toxic", query_key=[0, 1], trajectory_text="t")
        for _ in range(4):
            r_toxic.record_retrieval(10, utility=0.0)
        r_toxic.record_retrieval(11, utility=1.0)
        assert r_toxic.mean_utility == pytest.approx(0.2)

        # r_helpful: retrieval_count = 5, utility = [1, 1, 1, 1, 0] -> mean 0.8 (> beta 0.5) -> keep
        r_helpful = ExperienceRecord(id="helpful", query_key=[1, 1], trajectory_text="t")
        for _ in range(4):
            r_helpful.record_retrieval(10, utility=1.0)
        r_helpful.record_retrieval(11, utility=0.0)
        assert r_helpful.mean_utility == pytest.approx(0.8)

        bank.add_many([r_untested, r_toxic, r_helpful])

        policy = HistoryBasedDeletionPolicy(min_retrievals=5, utility_threshold=0.5, higher_is_better=True)
        evicted = policy.apply(bank, current_step=100)

        assert evicted == ["toxic"]
        assert bank.size() == 2
        assert bank.get("toxic") is None
        assert bank.get("untested") is not None
        assert bank.get("helpful") is not None

    def test_combined_deletion_policy(self):
        bank = BaseMemoryBank()

        # r1: inactive (periodic candidate at step 500)
        r1 = ExperienceRecord(id="inactive", query_key=[1, 0], trajectory_text="t1", entry_step=0)

        # r2: toxic (history candidate: n=5, mean=0.1 <= 0.5)
        r2 = ExperienceRecord(id="toxic", query_key=[0, 1], trajectory_text="t2", entry_step=0)
        for _ in range(5):
            r2.record_retrieval(200, utility=0.1)

        # r3: star record (active and high utility)
        r3 = ExperienceRecord(id="star", query_key=[1, 1], trajectory_text="t3", entry_step=0)
        for _ in range(5):
            r3.record_retrieval(300, utility=1.0)

        bank.add_many([r1, r2, r3])

        comb_policy = CombinedDeletionPolicy(
            period=500,
            alpha=0,
            min_retrievals=5,
            utility_threshold=0.5,
        )

        evicted = comb_policy.apply(bank, current_step=500)
        assert set(evicted) == {"inactive", "toxic"}
        assert bank.size() == 1
        assert bank.get("star") is not None

    def test_constrained_capacity_deletion_policy(self):
        bank = BaseMemoryBank()

        # Create 5 records with different mean utilities
        records = []
        for i, u in enumerate([0.1, 0.4, 0.6, 0.8, 0.95]):
            rec = ExperienceRecord(id=f"r_{i}", query_key=[float(i), 0.0], trajectory_text=f"traj_{i}")
            rec.update_utility(u)
            records.append(rec)

        bank.add_many(records)
        assert bank.size() == 5

        # Enforce max capacity = 3
        # Lowest utilities (0.1 -> r_0, 0.4 -> r_1) must be evicted
        policy = ConstrainedCapacityDeletionPolicy(max_capacity=3, higher_is_better=True)
        evicted = policy.apply(bank, current_step=10)

        assert set(evicted) == {"r_0", "r_1"}
        assert bank.size() == 3
        remaining_ids = {r.id for r in bank.all_records()}
        assert remaining_ids == {"r_2", "r_3", "r_4"}

        # Running again when size <= max_capacity does nothing
        assert policy.apply(bank, current_step=11) == []
        assert bank.size() == 3


# =====================================================================
# 5. AdaptiveReadFilter (System-1 Read Rejection) Tests
# =====================================================================

class TestAdaptiveReadFilter:
    """Test our engineering extension: dynamic read-time candidate filtering and backoff."""

    def test_adaptive_read_filter_backoff(self):
        bank = BaseMemoryBank(metric=SimilarityMetricType.COSINE)

        # Nearest candidate (toxic record with bad utility history)
        # Query: [1.0, 0.0]
        r_toxic = ExperienceRecord(
            id="toxic_top1",
            query_key=[1.0, 0.0],
            trajectory_text="erroneous demonstration",
            retrieval_count=3,
            utility_history=[0.0, 0.0, 0.0],
            mean_utility=0.0,
        )

        # Second nearest candidate (high-utility demonstration)
        r_good = ExperienceRecord(
            id="good_top2",
            query_key=[0.95, 0.31],
            trajectory_text="correct demonstration",
            retrieval_count=3,
            utility_history=[1.0, 1.0, 1.0],
            mean_utility=1.0,
        )

        # Third candidate (fresh / un-evaluated demonstration)
        r_fresh = ExperienceRecord(
            id="fresh_top3",
            query_key=[0.8, 0.6],
            trajectory_text="fresh demonstration",
            retrieval_count=0,
            utility_history=[],
            mean_utility=0.0,
        )

        bank.add_many([r_toxic, r_good, r_fresh])

        # Setup filter with baseline threshold 0.5
        read_filter = AdaptiveReadFilter(
            min_retrievals=1,
            utility_threshold=0.5,
            use_moving_average=False,
        )

        # Standard retrieve without filter gives toxic_top1
        raw_results = bank.retrieve([1.0, 0.0], top_k=1)
        assert raw_results[0].record.id == "toxic_top1"

        # Read-filtered retrieve rejects toxic_top1 and backs off to good_top2
        filtered_results = read_filter.retrieve_filtered(bank, [1.0, 0.0], top_k=1)
        assert len(filtered_results) == 1
        assert filtered_results[0].record.id == "good_top2"
        assert filtered_results[0].rank == 1

        stats = read_filter.get_stats()
        assert stats["total_rejected_candidates"] == 1
        assert stats["total_accepted_candidates"] == 1
        assert stats["rejection_rate"] == 0.5

    def test_dynamic_threshold_adaptation_with_moving_average(self):
        read_filter = AdaptiveReadFilter(
            min_retrievals=1,
            utility_threshold=0.3,
            use_moving_average=True,
            moving_avg_window=10,
            margin=0.1,
            higher_is_better=True,
        )

        # Initial threshold equals baseline
        assert read_filter.get_current_threshold() == pytest.approx(0.3)

        # Agent experiences strong run (utility = 0.9)
        for _ in range(10):
            read_filter.update_agent_utility(0.9)

        # Moving average = 0.9, margin = 0.1 -> threshold becomes 0.8
        assert read_filter.get_current_threshold() == pytest.approx(0.8)

        # Record with utility 0.6 would pass baseline 0.3 but fails dynamic 0.8 threshold
        r_mediocre = ExperienceRecord(
            id="med",
            query_key=[1, 0],
            trajectory_text="t",
            retrieval_count=2,
            utility_history=[0.6, 0.6],
            mean_utility=0.6,
        )
        assert read_filter.should_reject(r_mediocre) is True

    def test_fallback_behavior_when_all_rejected(self):
        bank = BaseMemoryBank()
        r_bad1 = ExperienceRecord(id="b1", query_key=[1, 0], trajectory_text="t", retrieval_count=2, utility_history=[0.0], mean_utility=0.0)
        r_bad2 = ExperienceRecord(id="b2", query_key=[0, 1], trajectory_text="t", retrieval_count=2, utility_history=[0.1], mean_utility=0.1)
        bank.add_many([r_bad1, r_bad2])

        read_filter = AdaptiveReadFilter(
            min_retrievals=1,
            utility_threshold=0.7,
            use_moving_average=False,
            fallback_to_top_k=True,
        )

        # All records fail threshold 0.7, fallback returns top 1
        results = read_filter.retrieve_filtered(bank, [1, 0], top_k=1)
        assert len(results) == 1
        assert results[0].record.id == "b1"


# =====================================================================
# 6. End-to-End Episodic Memory Lifecycle Simulation Test
# =====================================================================

class TestEndToEndMemoryLifecycle:
    """Simulate a multi-step episodic execution loop: retrieve -> score -> add -> update -> delete."""

    def test_full_episodic_stream_simulation(self):
        np.random.seed(42)
        bank = BaseMemoryBank(metric=SimilarityMetricType.COSINE)
        addition_policy = CoarseAdditionPolicy(threshold=1.4, error_based=True)
        deletion_policy = CombinedDeletionPolicy(
            period=20,
            alpha=0,
            min_retrievals=3,
            utility_threshold=0.5,
        )
        read_filter = AdaptiveReadFilter(
            min_retrievals=1,
            utility_threshold=0.4,
            use_moving_average=True,
        )

        # Seed initial bank with 5 clean records
        for i in range(5):
            vec = np.random.randn(4).tolist()
            bank.add(
                ExperienceRecord(
                    id=f"seed_{i}",
                    query_key=vec,
                    trajectory_text=f"clean_demo_{i}",
                    entry_step=0,
                    mean_utility=1.0,
                )
            )

        assert bank.size() == 5

        # Stream of 30 tasks
        for step in range(1, 31):
            q_vec = np.random.randn(4).tolist()

            # 1. Retrieve top-2 demos with adaptive filter
            demos = read_filter.retrieve_filtered(bank, q_vec, top_k=2)
            assert len(demos) > 0

            # 2. Simulated agent outcome
            # Even steps: agent succeeds (error 0.5 <= 1.4, utility 1.0)
            # Odd steps: agent fails (error 2.0 > 1.4, utility 0.0)
            is_success = (step % 2 == 0)
            simulated_error = 0.5 if is_success else 2.0
            downstream_utility = 1.0 if is_success else 0.0

            # 3. Update retrieved demonstrations with downstream utility
            for demo in demos:
                bank.update_utility(demo.record.id, downstream_utility, step=step)
                demo.record.record_retrieval(step=step)

            # 4. Update read filter moving average
            read_filter.update_agent_utility(downstream_utility)

            # 5. Check memory admission
            if addition_policy.should_add(query=q_vec, trajectory="pred", evaluation_result=simulated_error):
                new_rec = ExperienceRecord(
                    id=f"step_{step}",
                    query_key=q_vec,
                    trajectory_text=f"trajectory_at_step_{step}",
                    entry_step=step,
                )
                bank.add(new_rec)

            # 6. Apply deletion policy at periodic boundary
            deletion_policy.apply(bank, current_step=step)

        # Verify memory remains stable and functional
        assert bank.size() > 0
        all_records = bank.all_records()
        for rec in all_records:
            assert isinstance(rec, ExperienceRecord)

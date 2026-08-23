"""Configuration dataclasses for memory management baselines and benchmark protocols.

Matches specifications defined in research/RESEARCH_SPEC.md and research/evaluation_plan.md:
- Addition Baselines: Fixed, Add-All, Coarse (C1, C2, C3), Strict
- Deletion Baselines: None, Periodic, History-Based, Combined, Bounded Capacity
- Protocols: Protocol A (Growth), Protocol B (KDE), Protocol C (Shift), Protocol D (Bounded), Protocol E (Size-Matched)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, List, Optional, Sequence


@dataclass
class BaselineConfig:
    """Base configuration for agent execution and memory management."""
    name: str = "base"
    addition_policy: str = "fixed"  # 'fixed', 'add_all', 'coarse', 'strict'
    deletion_policy: str = "none"    # 'none', 'periodic', 'history', 'combined', 'bounded'
    top_k: int = 6
    temperature: float = 0.0
    initial_memory_size: int = 100

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FixedMemoryConfig(BaselineConfig):
    """Fixed memory baseline (pi=0, no deletion)."""
    name: str = "fixed_baseline"
    addition_policy: str = "fixed"
    deletion_policy: str = "none"


@dataclass
class AddAllConfig(BaselineConfig):
    """Naive growth baseline (pi=1, no deletion)."""
    name: str = "add_all"
    addition_policy: str = "add_all"
    deletion_policy: str = "none"


@dataclass
class CoarseAdditionConfig(BaselineConfig):
    """Selective coarse addition with C1, C2, or C3 judge level."""
    name: str = "coarse_addition"
    addition_policy: str = "coarse"
    deletion_policy: str = "none"
    coarse_level: str = "C1"  # 'C1', 'C2', 'C3'
    coarse_threshold: Optional[float] = None


@dataclass
class StrictAdditionConfig(BaselineConfig):
    """Selective strict addition with ground-truth oracle."""
    name: str = "strict_addition"
    addition_policy: str = "strict"
    deletion_policy: str = "none"
    strict_threshold: float = 1.0


@dataclass
class NoDeletionConfig:
    """No deletion policy config."""
    deletion_policy: str = "none"


@dataclass
class PeriodicDeletionConfig:
    """Periodic deletion policy config."""
    deletion_policy: str = "periodic"
    period: int = 500
    alpha: int = 0  # Prune if retrievals in window <= alpha


@dataclass
class HistoryDeletionConfig:
    """History-based utility deletion policy config."""
    deletion_policy: str = "history"
    min_retrievals: int = 5  # Minimum retrieval count n before deletion eligibility
    beta: float = 0.5        # Mean utility threshold below which record is pruned


@dataclass
class StrictDeletionConfig(HistoryDeletionConfig):
    """Alias for history-based utility deletion config with strict addition."""
    name: str = "strict_deletion"


@dataclass
class CombinedDeletionConfig:
    """Combined periodic and history-based deletion policy config."""
    deletion_policy: str = "combined"
    period: int = 500
    alpha: int = 0
    min_retrievals: int = 5
    beta: float = 0.5


@dataclass
class BoundedCapacityConfig:
    """Hard capacity bounded memory config."""
    deletion_policy: str = "bounded"
    max_capacity: int = 100
    period: int = 500
    alpha: int = 0


# =====================================================================
# Experimental Protocol Configs
# =====================================================================

@dataclass
class ProtocolAConfig:
    """Protocol A: Long-Term Memory Growth & Evolution.

    Evaluates task performance (SR/ACC), memory bank size M(t), and Pearson r_EF
    across long task streams (T = 500 to 4000).
    """
    protocol_name: str = "Protocol_A_Long_Term_Growth"
    stream_length: int = 1000
    initial_memory_size: int = 100
    seeds: List[int] = field(default_factory=lambda: [42, 128, 256, 512, 1024])
    addition_strategies: List[str] = field(
        default_factory=lambda: ["fixed", "add_all", "coarse_c1", "coarse_c2", "coarse_c3", "strict"]
    )
    evaluate_error_free_twin: bool = True
    output_dir: str = "results/protocol_a"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ProtocolBConfig:
    """Protocol B: Memory Deletion & Utility Eviction.

    Evaluates the quality separation between deleted and retained memories via KDE.
    """
    protocol_name: str = "Protocol_B_Deletion_KDE"
    stream_length: int = 1000
    initial_memory_size: int = 100
    seeds: List[int] = field(default_factory=lambda: [42, 128, 256, 512, 1024])
    deletion_strategies: List[str] = field(
        default_factory=lambda: ["none", "periodic", "history", "combined"]
    )
    history_min_retrievals: int = 5
    history_beta_thresholds: List[float] = field(default_factory=lambda: [0.3, 0.5, 0.7])
    periodic_window: int = 500
    periodic_alpha: int = 0
    output_dir: str = "results/protocol_b"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ProtocolCConfig:
    """Protocol C: Task Distribution Shift Adaptation.

    Sequences streaming queries by GMM clusters (C0 -> C1 -> C2) to test memory adaptation.
    """
    protocol_name: str = "Protocol_C_Distribution_Shift"
    stream_length: int = 1000
    initial_memory_size: int = 100
    n_clusters: int = 3
    seeds: List[int] = field(default_factory=lambda: [42, 128, 256, 512, 1024])
    comparison_strategies: List[str] = field(
        default_factory=lambda: ["fixed", "strict_no_del", "strict_periodic", "strict_history", "strict_combined"]
    )
    output_dir: str = "results/protocol_c"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ProtocolDConfig:
    """Protocol D: Resource-Constrained Bounded Memory.

    Imposes hard capacity limits M_max in {50, 100, 180, 360} with utility-based eviction.
    """
    protocol_name: str = "Protocol_D_Bounded_Memory"
    stream_length: int = 1000
    initial_memory_size: int = 100
    capacity_limits: List[int] = field(default_factory=lambda: [50, 100, 180, 360])
    seeds: List[int] = field(default_factory=lambda: [42, 128, 256, 512, 1024])
    output_dir: str = "results/protocol_d"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ProtocolEConfig:
    """Protocol E: Size-Matched Deletion Quality Ablation.

    Compares subsampled equal-size memory pools on a fresh held-out test split.
    """
    protocol_name: str = "Protocol_E_Size_Matched_Ablation"
    stream_length: int = 1000
    initial_memory_size: int = 100
    subsample_size: int = 500
    test_split_size: int = 500
    seeds: List[int] = field(default_factory=lambda: [42, 128, 256, 512, 1024])
    output_dir: str = "results/protocol_e"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BenchmarkConfig:
    """Master benchmark launch configuration."""
    benchmark_name: str = "agent_memory_management_benchmark"
    environment: str = "reg_agent"  # 'reg_agent', 'ciciot', 'all'
    active_protocols: List[str] = field(default_factory=lambda: ["A", "B", "C", "D"])
    stream_length: int = 1000
    initial_memory_size: int = 100
    seeds: List[int] = field(default_factory=lambda: [42, 128, 256, 512, 1024])
    output_root: str = "results"
    protocol_a: ProtocolAConfig = field(default_factory=ProtocolAConfig)
    protocol_b: ProtocolBConfig = field(default_factory=ProtocolBConfig)
    protocol_c: ProtocolCConfig = field(default_factory=ProtocolCConfig)
    protocol_d: ProtocolDConfig = field(default_factory=ProtocolDConfig)
    protocol_e: ProtocolEConfig = field(default_factory=ProtocolEConfig)
    log_level: str = "INFO"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

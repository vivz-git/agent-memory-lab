"""Agent Orchestrator managing retrieval, execution, evaluation, and memory updates."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Union
import numpy as np
from pydantic import BaseModel, Field, ConfigDict

from src.environments.base import BaseEnvironment, TaskQuery, TaskResult
from src.agent.core import BaseAgent, Demonstration


class StepResult(BaseModel):
    """Encapsulates the complete state and metrics of a single orchestrator step."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    step: int
    query: TaskQuery
    retrieved_demonstrations: List[Demonstration] = Field(default_factory=list)
    filtered_demonstrations: List[Demonstration] = Field(default_factory=list)
    prompt: str = ""
    raw_output: str = ""
    prediction: Any = None
    task_result: TaskResult
    added_to_memory: bool = False
    deleted_memory_ids: List[str] = Field(default_factory=list)
    memory_size_after_step: int = 0
    input_similarity: Optional[float] = None
    output_similarity: Optional[float] = None


class AdaptiveReadFilter:
    """System-1 'Read' Rejection Filter (Section 13 of RESEARCH_SPEC.md).

    Filters out retrieved demonstrations whose historical mean utility is below threshold.
    """

    def __init__(
        self,
        utility_threshold: float = 0.5,
        min_retrievals: int = 1,
    ) -> None:
        self.utility_threshold = utility_threshold
        self.min_retrievals = min_retrievals

    def filter(
        self,
        query: TaskQuery,
        demonstrations: List[Demonstration],
    ) -> List[Demonstration]:
        """Filter demonstration list by mean utility."""
        accepted: List[Demonstration] = []
        for d in demonstrations:
            retrieval_count = d.metadata.get("retrieval_count", 0)
            mean_utility = d.score if d.score is not None else d.metadata.get("mean_utility", 1.0)

            if retrieval_count >= self.min_retrievals and mean_utility < self.utility_threshold:
                continue
            accepted.append(d)
        return accepted


class SimpleEpisodicMemoryBank:
    """Built-in standalone episodic memory bank for testing and decoupled execution."""

    def __init__(self, compute_similarity_fn: Optional[Callable[[Any, Any], float]] = None) -> None:
        self.records: Dict[str, Dict[str, Any]] = {}
        self.compute_similarity_fn = compute_similarity_fn

    def __len__(self) -> int:
        return len(self.records)

    def add(
        self,
        query: TaskQuery,
        execution: str,
        step: int = 0,
        memory_id: Optional[str] = None,
        initial_utility: Optional[float] = None,
    ) -> str:
        mid = memory_id or f"mem_{len(self.records):05d}_{step}"
        self.records[mid] = {
            "id": mid,
            "query": query,
            "execution": execution,
            "query_vector": query.get_numpy_vector(),
            "retrieval_count": 0,
            "utility_history": [initial_utility] if initial_utility is not None else [],
            "mean_utility": initial_utility if initial_utility is not None else 1.0,
            "step_added": step,
            "last_retrieved_step": step,
        }
        return mid

    def retrieve(
        self,
        query: TaskQuery,
        top_k: int = 6,
    ) -> List[Demonstration]:
        if not self.records:
            return []

        q_vec = query.get_numpy_vector()
        scored_records = []

        for mid, rec in self.records.items():
            if self.compute_similarity_fn is not None:
                sim = self.compute_similarity_fn(query, rec["query"])
            else:
                r_vec = rec["query_vector"]
                norm_q = np.linalg.norm(q_vec)
                norm_r = np.linalg.norm(r_vec)
                if norm_q > 1e-12 and norm_r > 1e-12:
                    sim = float(np.dot(q_vec, r_vec) / (norm_q * norm_r))
                else:
                    sim = 0.0

            scored_records.append((sim, rec))

        scored_records.sort(key=lambda x: x[0], reverse=True)
        top_matches = scored_records[:top_k]

        demos: List[Demonstration] = []
        for sim, rec in top_matches:
            demo = Demonstration(
                query=rec["query"],
                execution=rec["execution"],
                score=rec["mean_utility"],
                memory_id=rec["id"],
                metadata={
                    "similarity": sim,
                    "retrieval_count": rec["retrieval_count"],
                    "mean_utility": rec["mean_utility"],
                    "step_added": rec["step_added"],
                },
            )
            demos.append(demo)

        return demos

    def record_retrieval_utility(
        self,
        memory_id: str,
        step: int,
        utility: float,
    ) -> None:
        if memory_id in self.records:
            rec = self.records[memory_id]
            rec["retrieval_count"] += 1
            rec["utility_history"].append(utility)
            rec["mean_utility"] = float(np.mean(rec["utility_history"]))
            rec["last_retrieved_step"] = step

    def delete(self, memory_ids: List[str]) -> None:
        for mid in memory_ids:
            self.records.pop(mid, None)


class AgentOrchestrator:
    """Manages the end-to-end agent step and stream lifecycle."""

    def __init__(
        self,
        agent: BaseAgent,
        env: BaseEnvironment,
        memory_bank: Optional[Any] = None,
        addition_policy: Optional[Callable[[TaskQuery, str, TaskResult], bool]] = None,
        deletion_policy: Optional[Callable[[Any, int], List[str]]] = None,
        read_filter: Optional[AdaptiveReadFilter] = None,
        top_k: int = 6,
    ) -> None:
        self.agent = agent
        self.env = env
        self.memory_bank = (
            memory_bank
            if memory_bank is not None
            else SimpleEpisodicMemoryBank(compute_similarity_fn=env.compute_input_similarity)
        )
        self.addition_policy = addition_policy
        self.deletion_policy = deletion_policy
        self.read_filter = read_filter
        self.top_k = top_k
        self.current_step = 0

        self.input_similarities: List[float] = []
        self.output_similarities: List[float] = []

    def populate_initial_memory(self, initial_queries: List[TaskQuery]) -> None:
        """Seed memory bank with verified initial trajectories."""
        for q in initial_queries:
            if isinstance(self.agent, BaseAgent) and hasattr(self.agent, "format_initial_execution"):
                exec_str = getattr(self.agent, "format_initial_execution")(q)
            elif isinstance(q.ground_truth, (int, float)):
                exec_str = f"boxed{{{q.ground_truth:.4f}}}"
            else:
                exec_str = f"ANALYSIS: Verified baseline demonstration.\nANSWER: {q.ground_truth}"

            if hasattr(self.memory_bank, "add"):
                self.memory_bank.add(
                    query=q,
                    execution=exec_str,
                    step=0,
                    memory_id=q.query_id,
                    initial_utility=1.0,
                )

    def run_step(self, query: TaskQuery) -> StepResult:
        """Execute a single step through the full agent memory loop."""
        self.current_step += 1
        step = self.current_step

        if hasattr(self.memory_bank, "retrieve"):
            retrieved_demos: List[Demonstration] = self.memory_bank.retrieve(
                query, top_k=self.top_k
            )
        else:
            retrieved_demos = []

        if self.read_filter is not None:
            filtered_demos = self.read_filter.filter(query, retrieved_demos)
        else:
            filtered_demos = list(retrieved_demos)

        prompt = self.agent.generate_prompt(query, filtered_demos)
        prediction, raw_output = self.agent.act(query, filtered_demos)

        task_result = self.env.evaluate(query, prediction, raw_output)

        input_sim: Optional[float] = None
        output_sim: Optional[float] = None
        if retrieved_demos:
            top_demo = retrieved_demos[0]
            input_sim = self.env.compute_input_similarity(query, top_demo.query)
            output_sim = self.env.compute_output_similarity(raw_output, top_demo.execution)
            self.input_similarities.append(input_sim)
            self.output_similarities.append(output_sim)

        for demo in filtered_demos:
            if demo.memory_id and hasattr(self.memory_bank, "record_retrieval_utility"):
                self.memory_bank.record_retrieval_utility(
                    demo.memory_id,
                    step=step,
                    utility=task_result.score,
                )

        added_to_memory = False
        if self.addition_policy is not None:
            should_add = self.addition_policy(query, raw_output, task_result)
        else:
            should_add = False

        if should_add and hasattr(self.memory_bank, "add"):
            self.memory_bank.add(
                query=query,
                execution=raw_output,
                step=step,
                initial_utility=task_result.score,
            )
            added_to_memory = True

        deleted_ids: List[str] = []
        if self.deletion_policy is not None:
            deleted_ids = self.deletion_policy(self.memory_bank, step)
            if deleted_ids and hasattr(self.memory_bank, "delete"):
                self.memory_bank.delete(deleted_ids)

        mem_size = len(self.memory_bank) if hasattr(self.memory_bank, "__len__") else 0

        return StepResult(
            step=step,
            query=query,
            retrieved_demonstrations=retrieved_demos,
            filtered_demonstrations=filtered_demos,
            prompt=prompt,
            raw_output=raw_output,
            prediction=prediction,
            task_result=task_result,
            added_to_memory=added_to_memory,
            deleted_memory_ids=deleted_ids,
            memory_size_after_step=mem_size,
            input_similarity=input_sim,
            output_similarity=output_sim,
        )

    def run_stream(self, queries: List[TaskQuery]) -> List[StepResult]:
        """Execute an entire sequence of task queries."""
        results: List[StepResult] = []
        for q in queries:
            results.append(self.run_step(q))
        return results

    def compute_experience_following_pearson_r(self) -> float:
        """Compute Pearson correlation r between input and output similarities."""
        if len(self.input_similarities) < 2:
            return 0.0
        in_arr = np.asarray(self.input_similarities, dtype=np.float64)
        out_arr = np.asarray(self.output_similarities, dtype=np.float64)

        std_in = np.std(in_arr)
        std_out = np.std(out_arr)

        if std_in < 1e-12 or std_out < 1e-12:
            return 0.0

        r = float(np.corrcoef(in_arr, out_arr)[0, 1])
        return 0.0 if np.isnan(r) else r

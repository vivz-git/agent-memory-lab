"""Evaluator modules for agent trajectory assessment.

Implements:
- BaseEvaluator abstract base class & EvaluationResult dataclass
- StrictOracleEvaluator (ground-truth oracles across RegAgent, CIC-IoT, EHRAgent, AgentDriver)
- CoarseLLMJudgeEvaluator (C1, C2, C3 threshold & prompt scoring logic matching Appendix A.4)
- Domain-specific specialized evaluator classes
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import re
from typing import Any, Callable, Optional, Sequence, Union
import numpy as np


@dataclass
class EvaluationResult:
    """Standard container for trajectory evaluation outcomes."""
    passed: bool
    score: float
    reasoning: str = ""
    error_magnitude: Optional[float] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "score": self.score,
            "reasoning": self.reasoning,
            "error_magnitude": self.error_magnitude,
            "metadata": self.metadata,
        }


class BaseEvaluator(ABC):
    """Abstract base class for all trajectory evaluators."""

    @abstractmethod
    def evaluate(
        self,
        query: Any,
        trajectory: Any,
        ground_truth: Any = None,
    ) -> EvaluationResult:
        """Evaluate an executed trajectory against a task query and optional ground truth."""
        raise NotImplementedError


# =====================================================================
# Strict Oracle Evaluators
# =====================================================================

class StrictOracleEvaluator(BaseEvaluator):
    """Strict ground-truth oracle evaluator.

    Decision Rule (pi_strict):
        pi_strict(q, e) = 1 if GroundTruthMatch(q, e) else 0

    Task rules:
        - regression (RegAgent): |pred - gt| <= threshold (default 1.0)
        - classification (CIC-IoT): exact string match of attack label
        - code / sql (EHRAgent): exact execution result / code equivalence
        - trajectory (AgentDriver): UniAD 3-second average L2 error < threshold (default 2.5m)
    """

    def __init__(
        self,
        task_type: str = "regression",
        regression_threshold: float = 1.0,
        trajectory_l2_threshold: float = 2.5,
    ):
        self.task_type = task_type.lower()
        self.regression_threshold = regression_threshold
        self.trajectory_l2_threshold = trajectory_l2_threshold

    def evaluate(
        self,
        query: Any,
        trajectory: Any,
        ground_truth: Any = None,
    ) -> EvaluationResult:
        if ground_truth is None:
            return EvaluationResult(
                passed=False,
                score=0.0,
                reasoning="Ground truth required for StrictOracleEvaluator but None provided.",
            )

        if self.task_type in ("regression", "regagent"):
            return self._evaluate_regression(trajectory, ground_truth)
        elif self.task_type in ("classification", "ciciot", "ciciot_agent"):
            return self._evaluate_classification(trajectory, ground_truth)
        elif self.task_type in ("trajectory", "agentdriver"):
            return self._evaluate_trajectory(trajectory, ground_truth)
        elif self.task_type in ("code", "ehr", "ehragent"):
            return self._evaluate_code(trajectory, ground_truth)
        else:
            # Generic equality check
            matched = str(trajectory).strip() == str(ground_truth).strip()
            return EvaluationResult(
                passed=matched,
                score=1.0 if matched else 0.0,
                reasoning=f"Generic equality match: {'passed' if matched else 'failed'}",
            )

    def _evaluate_regression(self, trajectory: Any, ground_truth: Any) -> EvaluationResult:
        pred_val = parse_regagent_prediction(trajectory)
        gt_val = float(ground_truth)
        if pred_val is None:
            return EvaluationResult(
                passed=False,
                score=0.0,
                reasoning=f"Failed to parse numeric prediction from trajectory: {trajectory}",
                error_magnitude=float("inf"),
            )

        abs_err = abs(pred_val - gt_val)
        passed = abs_err <= self.regression_threshold
        return EvaluationResult(
            passed=passed,
            score=1.0 if passed else 0.0,
            reasoning=f"Absolute error {abs_err:.4f} <= threshold {self.regression_threshold:.4f}: {passed}",
            error_magnitude=abs_err,
            metadata={"pred": pred_val, "gt": gt_val, "error": abs_err},
        )

    def _evaluate_classification(self, trajectory: Any, ground_truth: Any) -> EvaluationResult:
        pred_label = parse_ciciot_prediction(trajectory)
        gt_label = str(ground_truth).strip().lower()
        passed = pred_label.strip().lower() == gt_label
        return EvaluationResult(
            passed=passed,
            score=1.0 if passed else 0.0,
            reasoning=f"Predicted '{pred_label}' vs ground truth '{gt_label}' -> {'CORRECT' if passed else 'INCORRECT'}",
            error_magnitude=0.0 if passed else 1.0,
            metadata={"pred_label": pred_label, "gt_label": str(ground_truth)},
        )

    def _evaluate_trajectory(self, trajectory: Any, ground_truth: Any) -> EvaluationResult:
        pred_arr = np.asarray(trajectory, dtype=np.float64)
        gt_arr = np.asarray(ground_truth, dtype=np.float64)
        l2_err = float(np.mean(np.linalg.norm(pred_arr - gt_arr, axis=-1)))
        passed = l2_err < self.trajectory_l2_threshold
        return EvaluationResult(
            passed=passed,
            score=1.0 if passed else 0.0,
            reasoning=f"Average waypoint L2 error {l2_err:.4f}m < threshold {self.trajectory_l2_threshold:.4f}m: {passed}",
            error_magnitude=l2_err,
            metadata={"l2_error": l2_err},
        )

    def _evaluate_code(self, trajectory: Any, ground_truth: Any) -> EvaluationResult:
        pred_str = str(trajectory).strip()
        gt_str = str(ground_truth).strip()
        passed = pred_str == gt_str
        return EvaluationResult(
            passed=passed,
            score=1.0 if passed else 0.0,
            reasoning=f"Code execution result match: {passed}",
            metadata={"pred": pred_str, "gt": gt_str},
        )


# =====================================================================
# Coarse LLM Judge Evaluators (C1, C2, C3)
# =====================================================================

COARSE_THRESHOLDS = {
    "C1": 1.6,  # Lenient judge
    "C2": 1.4,  # Moderate judge
    "C3": 1.2,  # Tuned / strict judge
}

CICIOT_JUDGE_PROMPT_TEMPLATE = """You are a strict evaluator for IoT traffic classification answers. You should infer the most likely label from the provided flow-level features, then judge whether the model's answer is CORRECT or INCORRECT.
### Judging Criteria: The Model Answer is CORRECT if its predicted label exactly matches your inferred gold label, else INCORRECT.
### Judging Steps:
1) Carefully analyze key fields and reason your inferred gold label.
2) Compare the Model Answer's label to your gold label.
### Problem: Based on the following features, determine the most likely traffic type from the list below:
### Features: {problem}
### Model Answer: {generated_answer}
- Respond with your judgement and explanation as following format:
- First line: Respond with 'CORRECT' or 'INCORRECT' only.
- Following lines: Provide your reasoning or chain-of-thought.
Your judgement:"""

EHR_JUDGE_PROMPT_TEMPLATE = """You are an expert judge for Electronic Health Records (EHR) database queries and analysis. Your task is to evaluate whether the provided code solution and execution result are reasonable and correct for the given medical database question.
Output Format:
- First line of your answer: 'yes' or 'no' ONLY.
- Following lines: briefly provide your reasoning and analysis.
### Execution to be judged: Question:{question} Solution:{code}
### Execution Result: {execution_result}
Your evaluation:"""

AGENTDRIVER_JUDGE_PROMPT_TEMPLATE = """You are a highly knowledgeable and rigorous judge for autonomous driving. You are judging a short-horizon trajectory.
Your output format:
- First line: strictly output 'yes' or 'no'.
- Following lines: provide your reasoning.
### Predicted trajectory: {pred_traj}
### Planning target: {planning_target}
Your evaluation:"""


class CoarseLLMJudgeEvaluator(BaseEvaluator):
    """Coarse / Automatic Evaluator with C1, C2, C3 threshold levels & prompt templates.

    For synthetic numeric tasks (RegAgent):
        - C1: |pred - gt| <= 1.6
        - C2: |pred - gt| <= 1.4
        - C3: |pred - gt| <= 1.2
        (or custom float threshold)

    For symbolic/LLM tasks (CIC-IoT, EHRAgent, AgentDriver):
        - Formats prompt according to Appendix A.4
        - Calls llm_callable(prompt) -> str
        - Parses first-line decision: 'CORRECT'/'INCORRECT' or 'yes'/'no'
    """

    def __init__(
        self,
        level: str = "C1",
        task_type: str = "regression",
        custom_threshold: Optional[float] = None,
        llm_callable: Optional[Callable[[str], str]] = None,
    ):
        self.level = level.upper()
        self.task_type = task_type.lower()
        self.llm_callable = llm_callable

        if custom_threshold is not None:
            self.threshold = float(custom_threshold)
        else:
            self.threshold = COARSE_THRESHOLDS.get(self.level, 1.6)

    def evaluate(
        self,
        query: Any,
        trajectory: Any,
        ground_truth: Any = None,
    ) -> EvaluationResult:
        if self.task_type in ("regression", "regagent"):
            return self._evaluate_regression(trajectory, ground_truth)
        elif self.task_type in ("classification", "ciciot", "ciciot_agent"):
            return self._evaluate_ciciot_llm(query, trajectory, ground_truth)
        elif self.task_type in ("code", "ehr", "ehragent"):
            return self._evaluate_ehr_llm(query, trajectory, ground_truth)
        elif self.task_type in ("trajectory", "agentdriver"):
            return self._evaluate_agentdriver_llm(query, trajectory, ground_truth)
        else:
            # Fallback to threshold or generic
            return self._evaluate_regression(trajectory, ground_truth)

    def _evaluate_regression(self, trajectory: Any, ground_truth: Any) -> EvaluationResult:
        if ground_truth is None:
            return EvaluationResult(
                passed=False,
                score=0.0,
                reasoning="Ground truth required for regression coarse evaluation.",
            )
        pred_val = parse_regagent_prediction(trajectory)
        gt_val = float(ground_truth)
        if pred_val is None:
            return EvaluationResult(
                passed=False,
                score=0.0,
                reasoning=f"Unparseable regression prediction: {trajectory}",
                error_magnitude=float("inf"),
            )

        abs_err = abs(pred_val - gt_val)
        passed = abs_err <= self.threshold
        return EvaluationResult(
            passed=passed,
            score=1.0 if passed else 0.0,
            reasoning=f"Coarse {self.level} absolute error {abs_err:.4f} <= threshold {self.threshold:.4f}: {passed}",
            error_magnitude=abs_err,
            metadata={"level": self.level, "threshold": self.threshold, "pred": pred_val, "gt": gt_val},
        )

    def _evaluate_ciciot_llm(self, query: Any, trajectory: Any, ground_truth: Any) -> EvaluationResult:
        prompt = CICIOT_JUDGE_PROMPT_TEMPLATE.format(
            problem=str(query),
            generated_answer=str(trajectory),
        )

        if self.llm_callable is not None:
            raw_response = self.llm_callable(prompt)
            passed, reasoning = parse_judge_decision(raw_response)
            return EvaluationResult(
                passed=passed,
                score=1.0 if passed else 0.0,
                reasoning=reasoning,
                metadata={"raw_llm_response": raw_response, "level": self.level},
            )

        # Heuristic fallback if no LLM callable is attached
        if ground_truth is not None:
            pred_label = parse_ciciot_prediction(trajectory)
            passed = pred_label.strip().lower() == str(ground_truth).strip().lower()
            return EvaluationResult(
                passed=passed,
                score=1.0 if passed else 0.0,
                reasoning=f"Heuristic judge (no LLM attached): predicted '{pred_label}' vs gold '{ground_truth}' -> {passed}",
                metadata={"pred_label": pred_label, "level": self.level},
            )

        return EvaluationResult(
            passed=False,
            score=0.0,
            reasoning="Cannot evaluate without llm_callable or ground_truth.",
        )

    def _evaluate_ehr_llm(self, query: Any, trajectory: Any, ground_truth: Any) -> EvaluationResult:
        prompt = EHR_JUDGE_PROMPT_TEMPLATE.format(
            question=str(query),
            code=str(trajectory),
            execution_result=str(ground_truth or ""),
        )

        if self.llm_callable is not None:
            raw_response = self.llm_callable(prompt)
            passed, reasoning = parse_judge_decision(raw_response)
            return EvaluationResult(
                passed=passed,
                score=1.0 if passed else 0.0,
                reasoning=reasoning,
                metadata={"raw_llm_response": raw_response, "level": self.level},
            )

        # Fallback
        matched = str(trajectory).strip() == str(ground_truth).strip()
        return EvaluationResult(
            passed=matched,
            score=1.0 if matched else 0.0,
            reasoning=f"Heuristic code evaluation match: {matched}",
        )

    def _evaluate_agentdriver_llm(self, query: Any, trajectory: Any, ground_truth: Any) -> EvaluationResult:
        prompt = AGENTDRIVER_JUDGE_PROMPT_TEMPLATE.format(
            pred_traj=str(trajectory),
            planning_target=str(query),
        )

        if self.llm_callable is not None:
            raw_response = self.llm_callable(prompt)
            passed, reasoning = parse_judge_decision(raw_response)
            return EvaluationResult(
                passed=passed,
                score=1.0 if passed else 0.0,
                reasoning=reasoning,
                metadata={"raw_llm_response": raw_response, "level": self.level},
            )

        # Fallback
        return EvaluationResult(
            passed=True,
            score=1.0,
            reasoning="Default heuristic trajectory pass.",
        )


# =====================================================================
# Domain Specialized Evaluators
# =====================================================================

class RegAgentStrictEvaluator(StrictOracleEvaluator):
    """Strict oracle evaluator for RegAgent (|pred - gt| <= 1.0)."""

    def __init__(self, threshold: float = 1.0):
        super().__init__(task_type="regression", regression_threshold=threshold)


class RegAgentCoarseEvaluator(CoarseLLMJudgeEvaluator):
    """Coarse evaluator for RegAgent with C1 (1.6), C2 (1.4), or C3 (1.2) threshold."""

    def __init__(self, level: str = "C1", custom_threshold: Optional[float] = None):
        super().__init__(level=level, task_type="regression", custom_threshold=custom_threshold)


class CICIOTStrictEvaluator(StrictOracleEvaluator):
    """Strict oracle evaluator for CIC-IoT Agent (exact traffic class match)."""

    def __init__(self):
        super().__init__(task_type="classification")


class CICIOTCoarseEvaluator(CoarseLLMJudgeEvaluator):
    """Coarse LLM judge evaluator for CIC-IoT Agent."""

    def __init__(self, level: str = "C1", llm_callable: Optional[Callable[[str], str]] = None):
        super().__init__(level=level, task_type="classification", llm_callable=llm_callable)


# =====================================================================
# String & Output Parsing Utilities
# =====================================================================

def parse_regagent_prediction(trajectory: Any) -> Optional[float]:
    """Extract numeric guess from RegAgent output.

    Handles formats:
        - raw float/int
        - "Guess: boxed{12.34}"
        - "\\boxed{12.34}"
        - "boxed{-0.5}"
        - "Guess: 12.34"
    """
    if isinstance(trajectory, (int, float)):
        return float(trajectory)

    text = str(trajectory).strip()

    # Try boxed patterns: boxed{...} or \boxed{...}
    boxed_match = re.search(r"\\?boxed\{([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)\}", text)
    if boxed_match:
        try:
            return float(boxed_match.group(1))
        except ValueError:
            pass

    # Try Guess: <number>
    guess_match = re.search(r"Guess:\s*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)", text, re.IGNORECASE)
    if guess_match:
        try:
            return float(guess_match.group(1))
        except ValueError:
            pass

    # Try lone number in string
    num_match = re.search(r"([+-]?\d+\.?\d*(?:[eE][+-]?\d+)?)", text)
    if num_match:
        try:
            return float(num_match.group(1))
        except ValueError:
            pass

    return None


def parse_ciciot_prediction(trajectory: Any) -> str:
    """Extract predicted attack class from CIC-IoT output.

    Handles format:
        ANALYSIS: ...
        ANSWER: <traffic_type>
    """
    text = str(trajectory).strip()
    ans_match = re.search(r"ANSWER:\s*([^\n\r]+)", text, re.IGNORECASE)
    if ans_match:
        return ans_match.group(1).strip()

    # Fallback to direct string if short
    return text


def parse_judge_decision(raw_response: str) -> tuple[bool, str]:
    """Parse binary decision from LLM judge response according to Appendix A.4.

    Valid positive signals: 'CORRECT', 'yes', 'true', '1' on first line or prefix.
    Valid negative signals: 'INCORRECT', 'no', 'false', '0' on first line or prefix.
    """
    lines = [l.strip() for l in raw_response.strip().splitlines() if l.strip()]
    if not lines:
        return False, "Empty judge response"

    first_line = lines[0].upper()
    reasoning = "\n".join(lines[1:]) if len(lines) > 1 else lines[0]

    # Check first line tokens
    if "INCORRECT" in first_line or first_line.startswith("NO") or first_line.startswith("FALSE"):
        return False, reasoning
    if "CORRECT" in first_line or first_line.startswith("YES") or first_line.startswith("TRUE"):
        return True, reasoning

    # Secondary check in entire text if first line is ambiguous
    first_token = first_line.split()[0].rstrip(".:,;") if first_line.split() else ""
    if first_token in ("YES", "CORRECT", "TRUE", "PASS"):
        return True, reasoning
    if first_token in ("NO", "INCORRECT", "FALSE", "FAIL"):
        return False, reasoning

    return False, f"Ambiguous judge response: {raw_response[:100]}"

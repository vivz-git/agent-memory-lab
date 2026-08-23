"""
Memory Record and Output Validation Module.

Provides:
- MemoryRecordValidator: Validates episodic memory record schemas, vector dimensions,
  numeric float bounds, and detects NaN/Inf/None anomalies.
- OutputValidator: Validates and parses LLM outputs for RegAgent, CIC-IoT, and Judge
  evaluators with safe fallbacks for malformed or unparseable text.
"""

from __future__ import annotations

import math
import re
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np


@dataclass
class ValidationResult:
    """Encapsulates the result of a validation operation."""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    sanitized_data: Optional[Dict[str, Any]] = None

    def __bool__(self) -> bool:
        return self.is_valid


class MemoryRecordValidator:
    """
    Validates episodic memory records for structural schema integrity,
    vector dimensional consistency, numeric float bounds, and NaN/Inf corruption.
    """

    DEFAULT_ALLOWED_VECTOR_BOUNDS: Tuple[float, float] = (-1e6, 1e6)
    DEFAULT_UTILITY_BOUNDS: Tuple[float, float] = (0.0, 1.0)

    def __init__(
        self,
        expected_vector_dim: Optional[int] = None,
        vector_bounds: Tuple[float, float] = DEFAULT_ALLOWED_VECTOR_BOUNDS,
        utility_bounds: Tuple[float, float] = DEFAULT_UTILITY_BOUNDS,
        strict_utility_mean_check: bool = True,
    ) -> None:
        """
        Initialize the MemoryRecordValidator.

        Args:
            expected_vector_dim: Expected dimensionality of query/key vectors (e.g. 6 for RegAgent).
            vector_bounds: Tuple of (min_val, max_val) allowed for vector components.
            utility_bounds: Tuple of (min_utility, max_utility) allowed for utility scores.
            strict_utility_mean_check: Whether to strictly check mean_utility against utility_history.
        """
        self.expected_vector_dim = expected_vector_dim
        self.vector_bounds = vector_bounds
        self.utility_bounds = utility_bounds
        self.strict_utility_mean_check = strict_utility_mean_check

    @staticmethod
    def _is_finite_number(val: Any) -> bool:
        """Check if a value is a valid, finite real number (not NaN or Inf)."""
        if isinstance(val, (int, float, np.number)):
            if math.isnan(float(val)) or math.isinf(float(val)):
                return False
            return True
        return False

    def validate_vector(
        self,
        vector: Any,
        expected_dim: Optional[int] = None,
        vector_bounds: Optional[Tuple[float, float]] = None,
    ) -> Tuple[bool, List[str], List[float]]:
        """
        Validate a numerical vector for dimensionality, finite values, and bounds.

        Args:
            vector: Sequence or numpy array of numbers.
            expected_dim: Optional dimensionality constraint.
            vector_bounds: Optional (min_val, max_val) bounds constraint.

        Returns:
            Tuple of (is_valid, list_of_errors, sanitized_float_list)
        """
        errors: List[str] = []
        sanitized: List[float] = []
        exp_dim = expected_dim if expected_dim is not None else self.expected_vector_dim
        v_bounds = vector_bounds if vector_bounds is not None else self.vector_bounds

        if vector is None:
            return False, ["Vector cannot be None"], []

        # Convert numpy array or sequence to list
        if isinstance(vector, np.ndarray):
            raw_elements = vector.flatten().tolist()
        elif isinstance(vector, (list, tuple)):
            raw_elements = list(vector)
        else:
            return False, [f"Vector must be a sequence or numpy array, got {type(vector).__name__}"], []

        # Check dimension
        if exp_dim is not None and len(raw_elements) != exp_dim:
            errors.append(
                f"Vector dimension mismatch: expected {exp_dim}, got {len(raw_elements)}"
            )

        if len(raw_elements) == 0:
            errors.append("Vector cannot be empty")
            return False, errors, []

        # Check each element
        for idx, elem in enumerate(raw_elements):
            if elem is None:
                errors.append(f"Vector component at index {idx} is None")
                continue
            if not self._is_finite_number(elem):
                errors.append(f"Vector component at index {idx} is non-finite or NaN/Inf: {elem}")
                continue
            
            f_val = float(elem)
            if not (v_bounds[0] <= f_val <= v_bounds[1]):
                errors.append(
                    f"Vector component at index {idx} ({f_val}) exceeds allowed bounds [{v_bounds[0]}, {v_bounds[1]}]"
                )
            sanitized.append(f_val)

        is_valid = len(errors) == 0
        return is_valid, errors, sanitized if is_valid else []

    def validate_record(
        self,
        record: Any,
        expected_dim: Optional[int] = None,
    ) -> ValidationResult:
        """
        Validate an episodic memory record dict or object.

        Expected standard fields:
            - id: int or str
            - query_vector / key / query: numerical vector or structured query
            - trajectory / execution / output: execution representation
            - retrieval_count / fr_t: non-negative integer
            - utility_history: list of float utility scores
            - mean_utility: float average utility

        Returns:
            ValidationResult with sanitized record dict if valid.
        """
        errors: List[str] = []
        warnings: List[str] = []
        rec_dict: Dict[str, Any] = {}

        # Handle dict, Pydantic model, or generic object
        if isinstance(record, dict):
            rec_dict = dict(record)
        elif hasattr(record, "model_dump"):  # Pydantic v2
            rec_dict = record.model_dump()
        elif hasattr(record, "dict"):  # Pydantic v1
            rec_dict = record.dict()
        elif hasattr(record, "__dict__"):
            rec_dict = dict(record.__dict__)
        else:
            return ValidationResult(
                is_valid=False,
                errors=[f"Invalid record type: expected dict or object, got {type(record).__name__}"],
            )

        sanitized: Dict[str, Any] = {}

        # 1. Validate ID
        rec_id = rec_dict.get("id")
        if rec_id is None:
            errors.append("Record is missing required field 'id'")
        elif not isinstance(rec_id, (int, str)):
            errors.append(f"Record id must be int or str, got {type(rec_id).__name__}")
        else:
            sanitized["id"] = rec_id

        # 2. Validate Query Vector / Key
        vec_key = None
        for candidate in ["query_vector", "key", "query"]:
            if candidate in rec_dict and rec_dict[candidate] is not None:
                vec_key = candidate
                break

        if vec_key is None:
            errors.append("Record missing query vector key ('query_vector', 'key', or 'query')")
        else:
            raw_vec = rec_dict[vec_key]
            # If query is a vector, validate vector
            if isinstance(raw_vec, (list, tuple, np.ndarray)):
                v_valid, v_errors, clean_vec = self.validate_vector(raw_vec, expected_dim=expected_dim)
                if not v_valid:
                    errors.extend([f"In '{vec_key}': {err}" for err in v_errors])
                else:
                    sanitized["query_vector"] = clean_vec
            elif isinstance(raw_vec, (str, dict)):
                # Non-vector semantic query
                sanitized["query"] = raw_vec
            else:
                errors.append(f"Field '{vec_key}' has unsupported type {type(raw_vec).__name__}")

        # 3. Validate Trajectory / Execution
        traj_key = None
        for candidate in ["trajectory", "execution", "output", "action"]:
            if candidate in rec_dict and rec_dict[candidate] is not None:
                traj_key = candidate
                break

        if traj_key is None:
            errors.append("Record missing execution trajectory ('trajectory', 'execution', or 'output')")
        else:
            traj_val = rec_dict[traj_key]
            if isinstance(traj_val, str):
                if len(traj_val.strip()) == 0:
                    errors.append(f"Execution trajectory '{traj_key}' cannot be empty whitespace")
                else:
                    sanitized["trajectory"] = traj_val
            elif isinstance(traj_val, (dict, list, int, float)):
                sanitized["trajectory"] = traj_val
            else:
                errors.append(f"Field '{traj_key}' has unsupported type {type(traj_val).__name__}")

        # 4. Validate Retrieval Count (fr_t)
        retrieval_count = rec_dict.get("retrieval_count", rec_dict.get("fr_t", 0))
        if retrieval_count is None:
            retrieval_count = 0

        if not isinstance(retrieval_count, (int, np.integer)):
            errors.append(f"retrieval_count must be an integer, got {type(retrieval_count).__name__}")
        elif retrieval_count < 0:
            errors.append(f"retrieval_count cannot be negative ({retrieval_count})")
        else:
            sanitized["retrieval_count"] = int(retrieval_count)

        # 5. Validate Utility History
        util_hist = rec_dict.get("utility_history", rec_dict.get("utility_scores", []))
        if util_hist is None:
            util_hist = []

        if not isinstance(util_hist, (list, tuple, np.ndarray)):
            errors.append(f"utility_history must be a sequence, got {type(util_hist).__name__}")
            sanitized_hist: List[float] = []
        else:
            sanitized_hist = []
            for idx, u_val in enumerate(util_hist):
                if not self._is_finite_number(u_val):
                    errors.append(f"utility_history[{idx}] is non-finite or NaN/Inf: {u_val}")
                    continue
                f_u = float(u_val)
                if not (self.utility_bounds[0] <= f_u <= self.utility_bounds[1]):
                    warnings.append(
                        f"utility_history[{idx}] ({f_u}) is outside typical utility bounds {self.utility_bounds}"
                    )
                sanitized_hist.append(f_u)
            sanitized["utility_history"] = sanitized_hist

        # 6. Validate Mean Utility
        mean_u = rec_dict.get("mean_utility", rec_dict.get("mean_score", None))
        if mean_u is not None:
            if not self._is_finite_number(mean_u):
                errors.append(f"mean_utility is non-finite or NaN/Inf: {mean_u}")
            else:
                f_mean = float(mean_u)
                if sanitized_hist and self.strict_utility_mean_check:
                    expected_mean = sum(sanitized_hist) / len(sanitized_hist)
                    if abs(f_mean - expected_mean) > 1e-3:
                        warnings.append(
                            f"mean_utility ({f_mean:.4f}) does not match utility_history average ({expected_mean:.4f}). Auto-correcting."
                        )
                        f_mean = expected_mean
                sanitized["mean_utility"] = f_mean
        else:
            if sanitized_hist:
                sanitized["mean_utility"] = sum(sanitized_hist) / len(sanitized_hist)
            else:
                sanitized["mean_utility"] = 0.0

        # Preserve metadata fields
        for meta_k in ["entry_step", "t_entry", "t_last_retrieved", "metadata", "cluster_id"]:
            if meta_k in rec_dict:
                sanitized[meta_k] = rec_dict[meta_k]

        is_valid = len(errors) == 0
        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            sanitized_data=sanitized if is_valid else None,
        )

    def validate_batch(
        self,
        records: Sequence[Any],
        expected_dim: Optional[int] = None,
    ) -> Tuple[List[Dict[str, Any]], List[Tuple[Any, List[str]]]]:
        """
        Validate a batch of memory records.

        Returns:
            Tuple of (valid_records, rejected_records_with_reasons)
        """
        valid_records: List[Dict[str, Any]] = []
        rejected_records: List[Tuple[Any, List[str]]] = []

        for rec in records:
            res = self.validate_record(rec, expected_dim=expected_dim)
            if res.is_valid and res.sanitized_data is not None:
                valid_records.append(res.sanitized_data)
            else:
                rejected_records.append((rec, res.errors))

        return valid_records, rejected_records


class OutputValidator:
    """
    Validates and extracts structured predictions from raw LLM text outputs
    across all supported benchmarks (RegAgent, CIC-IoT, Coarse Judges).
    Provides safe fallbacks when outputs are malformed, unparseable, or contain NaN/Infs.
    """

    CICIOT_ALLOWED_CLASSES = [
        "DDoS-ICMP_Flood",
        "DDoS-UDP_Flood",
        "DDoS-TCP_Flood",
        "DDoS-SYN_Flood",
        "DDoS-PSHACK_Flood",
        "DDoS-RSTFINFlood",
        "DDoS-HTTP_Flood",
        "BenignTraffic",
    ]

    @staticmethod
    def validate_regagent_output(
        raw_output: str,
        fallback: float = 0.0,
        bounds: Tuple[float, float] = (-1e5, 1e5),
    ) -> Tuple[bool, float, str]:
        """
        Validate and parse RegAgent LLM output.
        Expected format: `Guess: boxed{<number>}` or `boxed{<number>}` or `\\boxed{<number>}`.

        Returns:
            Tuple of (is_valid, parsed_float_value, error_message)
        """
        if not isinstance(raw_output, str) or not raw_output.strip():
            return False, fallback, "Raw output is empty or not a string"

        text = raw_output.strip()

        # Regular expression patterns to find boxed numbers
        patterns = [
            r"(?:Guess:\s*)?(?:\\)?boxed\{([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\}",
            r"Guess:\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)",
            r"([+-]?\d+\.\d+(?:[eE][+-]?\d+)?)",
            r"^([+-]?\d+)$",
        ]

        extracted_val: Optional[float] = None
        for pat in patterns:
            match = re.search(pat, text, re.IGNORECASE)
            if match:
                try:
                    candidate = float(match.group(1))
                    if not math.isnan(candidate) and not math.isinf(candidate):
                        extracted_val = candidate
                        break
                except (ValueError, IndexError):
                    continue

        if extracted_val is None:
            return False, fallback, f"Failed to extract numerical value from output: '{text[:80]}...'"

        if not (bounds[0] <= extracted_val <= bounds[1]):
            return (
                False,
                fallback,
                f"Extracted value {extracted_val} is out of safe bounds [{bounds[0]}, {bounds[1]}]",
            )

        return True, extracted_val, ""

    @classmethod
    def validate_ciciot_output(
        cls,
        raw_output: str,
        allowed_classes: Optional[List[str]] = None,
        fallback: str = "BenignTraffic",
    ) -> Tuple[bool, str, str, str]:
        """
        Validate and parse CIC-IoT agent LLM output.
        Expected format:
            ANALYSIS: <reasoning>
            ANSWER: <traffic_type>

        Returns:
            Tuple of (is_valid, parsed_analysis, parsed_label, error_message)
        """
        valid_classes = allowed_classes or cls.CICIOT_ALLOWED_CLASSES
        if not isinstance(raw_output, str) or not raw_output.strip():
            return False, "", fallback, "Raw output is empty or not a string"

        text = raw_output.strip()

        # Extract Analysis
        analysis_match = re.search(
            r"ANALYSIS:\s*(.*?)(?=ANSWER:|$)", text, re.IGNORECASE | re.DOTALL
        )
        analysis = analysis_match.group(1).strip() if analysis_match else ""

        # Extract Answer
        answer_match = re.search(r"ANSWER:\s*([^\n\r]+)", text, re.IGNORECASE)
        if not answer_match:
            # Fallback scan for class names directly in text
            found_classes = [c for c in valid_classes if re.search(r"\b" + re.escape(c) + r"\b", text, re.IGNORECASE)]
            if len(found_classes) == 1:
                return True, analysis, found_classes[0], ""
            return False, analysis, fallback, "Missing 'ANSWER:' header in output"

        raw_answer = answer_match.group(1).strip().strip("[]'\"`.")

        # Exact match or case-insensitive match against allowed classes
        matched_class = None
        for cls_name in valid_classes:
            if raw_answer.lower() == cls_name.lower():
                matched_class = cls_name
                break

        if matched_class is None:
            # Substring match check
            matching_sub = [c for c in valid_classes if c.lower() in raw_answer.lower()]
            if len(matching_sub) == 1:
                matched_class = matching_sub[0]
            elif len(matching_sub) > 1:
                return (
                    False,
                    analysis,
                    fallback,
                    f"Ambiguous answer: multiple traffic types matched in '{raw_answer}'",
                )
            else:
                return (
                    False,
                    analysis,
                    fallback,
                    f"Invalid traffic type '{raw_answer}' not in allowed classes",
                )

        return True, analysis, matched_class, ""

    @staticmethod
    def validate_judge_output(
        raw_output: str,
        fallback: bool = False,
    ) -> Tuple[bool, bool, str]:
        """
        Validate and parse binary LLM Judge evaluator outputs.
        Expected format:
            First line: 'CORRECT'/'INCORRECT' or 'yes'/'no' (case-insensitive).

        Returns:
            Tuple of (is_valid, is_correct_bool, raw_verdict_str)
        """
        if not isinstance(raw_output, str) or not raw_output.strip():
            return False, fallback, "Empty judge output"

        lines = [line.strip() for line in raw_output.strip().splitlines() if line.strip()]
        if not lines:
            return False, fallback, "No non-empty lines in judge output"

        first_line = lines[0].upper().strip("`'\"*.:;,")

        # Check positive verdicts
        if first_line in ["CORRECT", "YES", "TRUE", "PASS", "SUCCESS"]:
            return True, True, first_line
        if first_line.startswith("CORRECT") or first_line.startswith("YES"):
            return True, True, first_line

        # Check negative verdicts
        if first_line in ["INCORRECT", "NO", "FALSE", "FAIL", "FAILURE"]:
            return True, False, first_line
        if first_line.startswith("INCORRECT") or first_line.startswith("NO"):
            return True, False, first_line

        # Look in the rest of the text if first line is ambiguous
        text_upper = raw_output.upper()
        if "JUDGEMENT: CORRECT" in text_upper or "EVALUATION: YES" in text_upper:
            return True, True, "CORRECT"
        if "JUDGEMENT: INCORRECT" in text_upper or "EVALUATION: NO" in text_upper:
            return True, False, "INCORRECT"

        return False, fallback, f"Unrecognized judge verdict: '{first_line}'"

    @staticmethod
    def validate_json_output(
        raw_output: str,
        required_keys: Optional[List[str]] = None,
        fallback: Optional[Dict[str, Any]] = None,
    ) -> Tuple[bool, Dict[str, Any], str]:
        """
        Safely parse JSON from LLM output, extracting from markdown code fences if present.
        """
        default_fb = fallback if fallback is not None else {}
        if not isinstance(raw_output, str) or not raw_output.strip():
            return False, default_fb, "Empty output"

        text = raw_output.strip()

        # Extract markdown json block if present
        json_fence_match = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text, re.IGNORECASE)
        if json_fence_match:
            json_str = json_fence_match.group(1)
        else:
            json_obj_match = re.search(r"\{[\s\S]*\}", text)
            json_str = json_obj_match.group(0) if json_obj_match else text

        try:
            data = json.loads(json_str)
            if not isinstance(data, dict):
                return False, default_fb, f"Parsed JSON is not an object (got {type(data).__name__})"

            if required_keys:
                missing = [k for k in required_keys if k not in data]
                if missing:
                    return False, data, f"Missing required keys: {missing}"

            return True, data, ""
        except json.JSONDecodeError as exc:
            return False, default_fb, f"JSON decode error: {str(exc)}"

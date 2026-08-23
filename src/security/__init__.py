"""
Security & QA Module for Agent Memory Management.

Provides:
- MemoryRecordValidator: Schema validation, vector dimension checks, numeric float bounds, NaN/Inf checks.
- OutputValidator: LLM output format validation, malformed syntax catching, safe fallbacks.
- PromptSanitizer: Prompt injection defense, control token stripping, delimiter escaping.
- ExecutionGuardrails: Execution sandbox (no eval/exec), timeouts, secret redaction, safe logging.
"""

from .validator import (
    MemoryRecordValidator,
    OutputValidator,
    ValidationResult,
)
from .sanitizer import (
    PromptSanitizer,
    SanitizationResult,
)
from .guardrails import (
    ExecutionGuardrails,
    ExecutionTimeoutError,
    SecurityViolationError,
    SafeLogger,
)

__all__ = [
    "MemoryRecordValidator",
    "OutputValidator",
    "ValidationResult",
    "PromptSanitizer",
    "SanitizationResult",
    "ExecutionGuardrails",
    "ExecutionTimeoutError",
    "SecurityViolationError",
    "SafeLogger",
]

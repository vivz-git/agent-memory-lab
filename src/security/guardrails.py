"""
Execution Guardrails and Safety Module.

Provides:
- ExecutionGuardrails: AST-based static safety analysis (blocks eval/exec/dangerous imports),
  safe math evaluation, execution timeout enforcement, and secret redaction.
- SafeLogger: Secure logging utility that automatically filters sensitive secrets and API keys.
"""

from __future__ import annotations

import ast
import concurrent.futures
import functools
import logging
import operator
import re
import sys
import threading
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, Union


class SecurityViolationError(Exception):
    """Raised when an operation violates security boundaries (e.g. eval/exec, malicious imports)."""
    pass


class ExecutionTimeoutError(TimeoutError):
    """Raised when an execution exceeds its configured timeout duration."""
    pass


class ExecutionGuardrails:
    """
    Ensures safe agent execution by enforcing sandboxing, strict timeouts,
    and secret leakage prevention across logging and outputs.
    """

    # Modules strictly prohibited from execution / import
    PROHIBITED_MODULES: Set[str] = {
        "os",
        "sys",
        "subprocess",
        "shutil",
        "socket",
        "pty",
        "ctypes",
        "posix",
        "nt",
        "threading",
        "multiprocessing",
        "inspect",
        "importlib",
        "pickle",
        "marshal",
        "tempfile",
        "webbrowser",
    }

    # Builtin functions strictly prohibited in evaluated scripts/expressions
    PROHIBITED_CALLS: Set[str] = {
        "eval",
        "exec",
        "compile",
        "__import__",
        "open",
        "input",
        "globals",
        "locals",
        "getattr",
        "setattr",
        "delattr",
        "vars",
        "dir",
        "breakpoint",
    }

    # Prohibited dunder attributes
    PROHIBITED_ATTRS: Set[str] = {
        "__subclasses__",
        "__bases__",
        "__mro__",
        "__globals__",
        "__code__",
        "__builtins__",
        "__dict__",
        "__class__",
        "__reduce__",
        "__reduce_ex__",
    }

    # Secret and API key regex patterns for automated redaction
    SECRET_PATTERNS: List[Tuple[str, str]] = [
        # Anthropic API Keys (must be before generic sk- pattern)
        (r"sk-ant-[A-Za-z0-9_\-]{16,}", "[REDACTED_ANTHROPIC_KEY]"),
        # OpenAI API Keys (including project keys)
        (r"sk-(?:proj-)?[A-Za-z0-9_\-]{16,}", "[REDACTED_OPENAI_KEY]"),
        # HuggingFace tokens
        (r"hf_[A-Za-z0-9]{20,}", "[REDACTED_HF_TOKEN]"),
        # AWS Access Key ID
        (r"\bAKIA[0-9A-Z]{16}\b", "[REDACTED_AWS_KEY]"),
        # Bearer Tokens
        (r"(?i)bearer\s+[a-zA-Z0-9_\-\.]{24,}", "Bearer [REDACTED_BEARER_TOKEN]"),
        # Key=Value secret assignments (e.g. password=..., api_key="...")
        (
            r"(?i)\b(password|passwd|api[_-]?key|apikey|auth[_-]?token|secret_key)\s*[:=]\s*[\"']?[A-Za-z0-9_\-!@#$%^&*]{4,}[\"']?",
            r"\1=[REDACTED_SECRET]",
        ),
        # RSA/EC/Private Keys
        (
            r"-----BEGIN\s+[A-Z\s]+PRIVATE\s+KEY-----[\s\S]*?-----END\s+[A-Z\s]+PRIVATE\s+KEY-----",
            "[REDACTED_PRIVATE_KEY]",
        ),
    ]

    # Safe AST arithmetic operators map
    _SAFE_OPERATORS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }

    @classmethod
    def is_code_safe(cls, code_str: str) -> Tuple[bool, List[str]]:
        """
        Statically inspect Python code using AST to detect prohibited calls (eval, exec),
        dangerous imports (os, subprocess), or dunder reflection attacks.

        Returns:
            Tuple of (is_safe_bool, list_of_violations)
        """
        if not isinstance(code_str, str) or not code_str.strip():
            return True, []

        violations: List[str] = []

        try:
            tree = ast.parse(code_str)
        except SyntaxError as exc:
            return False, [f"Syntax error during AST parse: {exc}"]

        for node in ast.walk(tree):
            # Check for prohibited function calls
            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Name) and func.id in cls.PROHIBITED_CALLS:
                    violations.append(f"Prohibited function call: '{func.id}()'")
                elif isinstance(func, ast.Attribute) and func.attr in cls.PROHIBITED_CALLS:
                    violations.append(f"Prohibited method call: '{func.attr}()'")

            # Check for prohibited imports
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    root_mod = alias.name.split(".")[0]
                    if root_mod in cls.PROHIBITED_MODULES:
                        violations.append(f"Prohibited module import: '{alias.name}'")

            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    root_mod = node.module.split(".")[0]
                    if root_mod in cls.PROHIBITED_MODULES:
                        violations.append(f"Prohibited module import from: '{node.module}'")

            # Check for prohibited attribute access (dunders)
            elif isinstance(node, ast.Attribute):
                if node.attr in cls.PROHIBITED_ATTRS:
                    violations.append(f"Prohibited dunder attribute access: '{node.attr}'")

        return len(violations) == 0, violations

    @classmethod
    def safe_eval_math(cls, expr: str) -> float:
        """
        Safely evaluate a mathematical expression containing only numbers and standard
        arithmetic operators without using eval().

        Args:
            expr: Mathematical expression string, e.g. "3.5 * (2 + 4.1) / 2"

        Returns:
            Result as float.

        Raises:
            SecurityViolationError: If expression contains non-mathematical AST nodes.
            ValueError: If division by zero or invalid operation occurs.
        """
        if not isinstance(expr, str) or not expr.strip():
            raise ValueError("Expression cannot be empty")

        try:
            tree = ast.parse(expr.strip(), mode="eval")
        except SyntaxError as e:
            raise SecurityViolationError(f"Invalid mathematical syntax: {e}")

        def _eval_node(node: ast.AST) -> float:
            if isinstance(node, ast.Expression):
                return _eval_node(node.body)
            elif isinstance(node, ast.Constant):  # Python 3.8+ for numbers
                if isinstance(node.value, (int, float)):
                    return float(node.value)
                raise SecurityViolationError(f"Unsupported constant type: {type(node.value)}")
            elif isinstance(node, ast.UnaryOp):
                op_type = type(node.op)
                if op_type in cls._SAFE_OPERATORS:
                    val = _eval_node(node.operand)
                    return float(cls._SAFE_OPERATORS[op_type](val))
                raise SecurityViolationError(f"Unsupported unary operator: {op_type.__name__}")
            elif isinstance(node, ast.BinOp):
                op_type = type(node.op)
                if op_type in cls._SAFE_OPERATORS:
                    left = _eval_node(node.left)
                    right = _eval_node(node.right)
                    # Protect against excessive exponentiation (DoS)
                    if op_type is ast.Pow and right > 100:
                        raise SecurityViolationError(f"Exponent {right} exceeds safety limit of 100")
                    return float(cls._SAFE_OPERATORS[op_type](left, right))
                raise SecurityViolationError(f"Unsupported binary operator: {op_type.__name__}")
            else:
                raise SecurityViolationError(f"Prohibited node type in expression: {type(node).__name__}")

        return _eval_node(tree)

    @classmethod
    def run_with_timeout(
        cls,
        func: Callable[..., Any],
        args: Tuple[Any, ...] = (),
        kwargs: Optional[Dict[str, Any]] = None,
        timeout_seconds: float = 5.0,
    ) -> Any:
        """
        Execute a callable with an enforced timeout using a ThreadPoolExecutor.
        Works consistently across Windows, Linux, and macOS.

        Args:
            func: The function to execute.
            args: Positional arguments tuple.
            kwargs: Keyword arguments dictionary.
            timeout_seconds: Maximum allowed time in seconds.

        Returns:
            The return value of func(*args, **kwargs).

        Raises:
            ExecutionTimeoutError: If execution exceeds timeout_seconds.
            Exception: Re-raises any exception thrown by func.
        """
        kw = kwargs or {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(func, *args, **kw)
            try:
                return future.result(timeout=timeout_seconds)
            except concurrent.futures.TimeoutError:
                raise ExecutionTimeoutError(
                    f"Execution exceeded timeout limit of {timeout_seconds:.2f} seconds"
                )

    @classmethod
    def redact_secrets(
        cls,
        text: str,
        additional_secrets: Optional[List[str]] = None,
    ) -> str:
        """
        Redact API keys, bearer tokens, passwords, and private keys from text.

        Args:
            text: Text to sanitize.
            additional_secrets: Optional explicit secret strings to redact.

        Returns:
            Redacted string safe for logging.
        """
        if not isinstance(text, str):
            return str(text)

        redacted = text

        # 1. Apply regex pattern redactions
        for pattern, replacement in cls.SECRET_PATTERNS:
            redacted = re.sub(pattern, replacement, redacted)

        # 2. Redact specific known secret strings if provided
        if additional_secrets:
            for secret in additional_secrets:
                if secret and len(secret.strip()) >= 4:
                    redacted = redacted.replace(secret.strip(), "[REDACTED_CUSTOM_SECRET]")

        return redacted


class SafeLogger:
    """
    A security-hardened logger wrapper that automatically redacts API keys
    and sensitive tokens from all log messages.
    """

    def __init__(
        self,
        name: str = "agent_memory_safe_logger",
        level: int = logging.INFO,
    ) -> None:
        self.name = name
        self.level = level
        self._logs: List[str] = []
        self._lock = threading.Lock()

    def _record(self, level_name: str, msg: str, *args: Any) -> str:
        formatted = msg % args if args else msg
        clean_msg = ExecutionGuardrails.redact_secrets(str(formatted))
        entry = f"[{level_name}] {clean_msg}"
        with self._lock:
            self._logs.append(entry)
        return clean_msg

    def info(self, msg: str, *args: Any) -> str:
        return self._record("INFO", msg, *args)

    def warning(self, msg: str, *args: Any) -> str:
        return self._record("WARNING", msg, *args)

    def error(self, msg: str, *args: Any) -> str:
        return self._record("ERROR", msg, *args)

    def debug(self, msg: str, *args: Any) -> str:
        return self._record("DEBUG", msg, *args)

    def get_logs(self) -> List[str]:
        with self._lock:
            return list(self._logs)

    def clear(self) -> None:
        with self._lock:
            self._logs.clear()

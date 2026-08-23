"""
Prompt Sanitization and Injection Defense Module.

Provides:
- PromptSanitizer: Neutralizes prompt injection attempts within retrieved memories,
  escapes template formatting delimiters, and strips special control tokens.
"""

from __future__ import annotations

import re
import html
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union


@dataclass
class SanitizationResult:
    """Encapsulates the result of a text sanitization operation."""
    sanitized_text: str
    detected_injections: List[str] = field(default_factory=list)
    stripped_tokens: List[str] = field(default_factory=list)
    was_modified: bool = False

    @property
    def has_threats(self) -> bool:
        return len(self.detected_injections) > 0 or len(self.stripped_tokens) > 0


class PromptSanitizer:
    """
    Sanitizes retrieved memory items, input queries, and formatted prompts to prevent:
    1. Direct & Indirect Prompt Injection in retrieved demonstrations.
    2. Model control token hijacking (<|im_start|>, [INST], etc.).
    3. Template format string injection and delimiter corruption.
    """

    # Special LLM control tokens across OpenAI, Llama, Anthropic, Qwen, Mistral
    CONTROL_TOKENS: List[str] = [
        "<|endoftext|>",
        "<|im_start|>",
        "<|im_end|>",
        "<|system|>",
        "<|user|>",
        "<|assistant|>",
        "<|begin_of_text|>",
        "<|end_of_text|>",
        "<|start_header_id|>",
        "<|end_header_id|>",
        "<|eot_id|>",
        "<s>",
        "</s>",
        "[INST]",
        "[/INST]",
        "[SYS]",
        "[/SYS]",
        "<<SYS>>",
        "<</SYS>>",
        "<|pad|>",
        "<|sep|>",
    ]

    # Regex patterns indicating prompt injection or instruction hijacking
    INJECTION_PATTERNS: List[Tuple[str, str]] = [
        (
            r"(?i)\b(?:ignore|disregard|forget|override)\s+(?:all\s+)?(?:previous|prior|above|existing|system)?\s*(?:instructions|prompts|rules|commands|constraints)\b",
            "INSTRUCTION_OVERRIDE",
        ),
        (
            r"(?i)\b(?:you\s+are\s+now|act\s+as|pretend\s+to\s+be)\s+(?:a|an)?\s+(?:evil|unaligned|unrestricted|jailbroken|developer|admin|DAN|root)\b",
            "ROLEPLAY_JAILBREAK",
        ),
        (
            r"(?i)\b(?:system\s*:\s*|assistant\s*:\s*|human\s*:\s*|user\s*:\s*)(?:you\s+must|always|ignore|output)\b",
            "ROLE_INJECTION",
        ),
        (
            r"(?i)\b(?:output|respond|reply)\s+only\s+(?:with|as)?\s*(?:boxed\{[^}]*\}|[\"'][^\"']*[\"'])",
            "FORCED_OUTPUT_HIJACK",
        ),
        (
            r"(?i)\b(?:reveal|leak|print|show|display|tell\s+me)\s+(?:the\s+)?(?:[a-zA-Z0-9_\-]+\s+)*(?:secret|api[\s_-]?key|password|system\s+prompt|hidden\s+instructions)\b",
            "SECRET_EXFILTRATION",
        ),
        (
            r"(?i)\b(?:new\s+system\s+prompt|enable\s+developer\s+mode|unrestricted\s+mode)\b",
            "MODE_SWITCH_ATTACK",
        ),
        (
            r"(?i)<script[\s\S]*?>[\s\S]*?<\/script>",
            "HTML_SCRIPT_INJECTION",
        ),
    ]

    def __init__(
        self,
        strict_mode: bool = False,
        defang_tag: str = "[DEFANGED_INJECTION]",
    ) -> None:
        """
        Initialize PromptSanitizer.

        Args:
            strict_mode: If True, blocks/drops content with severe injections instead of defanging.
            defang_tag: Replacement tag for neutralizing matched injection clauses.
        """
        self.strict_mode = strict_mode
        self.defang_tag = defang_tag
        # Compile control tokens regex
        escaped_tokens = [re.escape(tok) for tok in self.CONTROL_TOKENS]
        self._control_token_regex = re.compile("|".join(escaped_tokens), re.IGNORECASE)

    def strip_control_tokens(self, text: str) -> Tuple[str, List[str]]:
        """
        Strip known LLM control tokens from text.

        Returns:
            Tuple of (cleaned_text, list_of_stripped_tokens)
        """
        if not isinstance(text, str):
            return str(text), []

        stripped: List[str] = []

        def _repl(match: re.Match) -> str:
            tok = match.group(0)
            stripped.append(tok)
            return ""

        cleaned = self._control_token_regex.sub(_repl, text)
        return cleaned, stripped

    def detect_injection(self, text: str) -> Tuple[bool, List[str]]:
        """
        Detect prompt injection signatures in text.

        Returns:
            Tuple of (has_injection_bool, list_of_detected_pattern_tags)
        """
        if not isinstance(text, str):
            return False, []

        detected: List[str] = []
        for pat, tag in self.INJECTION_PATTERNS:
            if re.search(pat, text):
                detected.append(tag)

        return len(detected) > 0, detected

    def defang_injection(self, text: str) -> Tuple[str, List[str]]:
        """
        Neutralize prompt injection phrases by replacing active imperative phrases
        with harmless defanged tokens.

        Returns:
            Tuple of (defanged_text, list_of_neutralized_tags)
        """
        if not isinstance(text, str):
            return str(text), []

        defanged = text
        detected: List[str] = []

        for pat, tag in self.INJECTION_PATTERNS:
            if re.search(pat, defanged):
                detected.append(tag)
                defanged = re.sub(pat, f" {self.defang_tag}: ({tag}) ", defanged)

        return defanged, detected

    def escape_delimiters(self, text: str, escape_braces_for_format: bool = True) -> str:
        """
        Escape template delimiters to prevent format string vulnerabilities
        and markdown block injection.

        Args:
            text: Input string.
            escape_braces_for_format: If True, doubles single braces '{' -> '{{' and '}' -> '}}'.

        Returns:
            Escaped string.
        """
        if not isinstance(text, str):
            return str(text)

        result = text

        # Prevent markdown fence breakout (e.g. ```` or raw backtick attacks)
        # Normalize excessive backtick sequences in content
        result = re.sub(r"`{4,}", "```", result)

        if escape_braces_for_format:
            # Safely double unescaped single braces (avoiding quadruple braces)
            # Replace single '{' not preceded or followed by '{' with '{{'
            result = re.sub(r"(?<!\{)\{(?!\{)", "{{", result)
            result = re.sub(r"(?<!\})\}(?!\})", "}}", result)

        return result

    def sanitize_demonstration(
        self,
        query: Any,
        trajectory: Any,
        escape_braces: bool = False,
    ) -> Tuple[str, str, SanitizationResult]:
        """
        Sanitize a demonstration (query, trajectory) retrieved from episodic memory
        before injecting into an in-context prompt.

        Returns:
            Tuple of (clean_query_str, clean_trajectory_str, SanitizationResult)
        """
        q_str = str(query)
        t_str = str(trajectory)

        # 1. Strip control tokens
        clean_q, tokens_q = self.strip_control_tokens(q_str)
        clean_t, tokens_t = self.strip_control_tokens(t_str)
        all_stripped = tokens_q + tokens_t

        # 2. Defang injection attempts
        clean_q, inj_q = self.defang_injection(clean_q)
        clean_t, inj_t = self.defang_injection(clean_t)
        all_injections = inj_q + inj_t

        # 3. Optional brace escaping
        if escape_braces:
            clean_q = self.escape_delimiters(clean_q, escape_braces_for_format=True)
            clean_t = self.escape_delimiters(clean_t, escape_braces_for_format=True)

        was_modified = (q_str != clean_q) or (t_str != clean_t)

        result = SanitizationResult(
            sanitized_text=f"Query: {clean_q} | Trajectory: {clean_t}",
            detected_injections=all_injections,
            stripped_tokens=all_stripped,
            was_modified=was_modified,
        )

        return clean_q, clean_t, result

    def sanitize_prompt(
        self,
        prompt_text: str,
        strip_control: bool = True,
        defang_injections: bool = True,
    ) -> SanitizationResult:
        """
        Sanitize a full rendered prompt string before LLM dispatch.
        """
        if not isinstance(prompt_text, str):
            return SanitizationResult(sanitized_text="", was_modified=False)

        current = prompt_text
        stripped_tokens: List[str] = []
        detected_injections: List[str] = []

        if strip_control:
            current, stripped_tokens = self.strip_control_tokens(current)

        if defang_injections:
            current, detected_injections = self.defang_injection(current)

        was_mod = current != prompt_text

        return SanitizationResult(
            sanitized_text=current,
            detected_injections=detected_injections,
            stripped_tokens=stripped_tokens,
            was_modified=was_mod,
        )

    def sanitize_tabular_features(
        self,
        features: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Sanitize tabular IoT / cyber features dictionary.
        Strips hidden prompt injections or control tokens embedded in feature names or values.
        """
        cleaned: Dict[str, Any] = {}
        for key, val in features.items():
            # Sanitize key
            clean_key, _ = self.strip_control_tokens(str(key))
            clean_key = re.sub(r"[^\w\-\.]", "_", clean_key.strip())

            # Sanitize value
            if isinstance(val, str):
                clean_val, _ = self.strip_control_tokens(val)
                clean_val, _ = self.defang_injection(clean_val)
                cleaned[clean_key] = clean_val.strip()
            elif isinstance(val, (int, float, bool)):
                cleaned[clean_key] = val
            elif isinstance(val, (list, tuple)):
                cleaned[clean_key] = [
                    self.strip_control_tokens(str(v))[0] if isinstance(v, str) else v for v in val
                ]
            else:
                cleaned[clean_key] = str(val)

        return cleaned

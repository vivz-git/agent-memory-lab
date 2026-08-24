"""Agent implementations, demonstration structures, and LLM interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
import os
import re
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
from pydantic import BaseModel, Field, ConfigDict

from src.environments.base import TaskQuery
from src.agent.prompts import (
    format_regagent_prompt,
    format_ciciot_prompt,
)
from src.environments.ciciot_env import CICIOT_CLASSES, canonical_label


class Demonstration(BaseModel):
    """Represents a retrieved demonstration record."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    query: TaskQuery
    execution: str = Field(
        description="The agent execution trajectory (e.g. boxed{1.23} or CoT reasoning).",
    )
    score: Optional[float] = Field(
        default=None,
        description="Historical or current utility score Phi(q, e).",
    )
    memory_id: Optional[str] = Field(
        default=None,
        description="Unique identifier of this demonstration in the memory bank.",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional context such as retrieval frequency, step added, etc.",
    )


class BaseLLMClient(ABC):
    """Abstract interface for LLM completions."""

    @abstractmethod
    def complete(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 256,
    ) -> str:
        """Generate completion text given a prompt string."""
        pass


class MockLLMClient(BaseLLMClient):
    """Deterministic local rule-based / mock execution LLM client for offline testability.

    Modes:
        - 'rule_based': Uses closed-form math/heuristics to emulate a competent agent.
        - 'demonstration_mimic': Strongly imitates the closest retrieved demonstration.
        - 'fixed': Returns a fixed predefined string.
        - 'custom': Dispatches to a custom callable.
    """

    def __init__(
        self,
        mode: str = "rule_based",
        fixed_response: str = "Guess: boxed{0.0}",
        custom_responder: Optional[Callable[[str], str]] = None,
        noise_std: float = 0.0,
        seed: Optional[int] = 42,
    ) -> None:
        self.mode = mode
        self.fixed_response = fixed_response
        self.custom_responder = custom_responder
        self.noise_std = noise_std
        self.rng = np.random.RandomState(seed)

    def complete(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 256,
    ) -> str:
        """Produce deterministic mock LLM completion."""
        if self.mode == "custom" and self.custom_responder is not None:
            return self.custom_responder(prompt)

        if self.mode == "fixed":
            return self.fixed_response

        # Check domain based on prompt headers
        if "6-dimensional input vector" in prompt or "Guess:" in prompt:
            return self._handle_regagent_prompt(prompt)
        elif "traffic type" in prompt or "ANALYSIS:" in prompt:
            return self._handle_ciciot_prompt(prompt)

        return "UNKNOWN_RESPONSE"

    def _handle_regagent_prompt(self, prompt: str) -> str:
        """Solve or interpolate RegAgent query from prompt text."""
        demo_matches = re.findall(
            r"Input:\s*\[([^\]]+)\]\s*Guess:\s*(?:\\)?boxed\{([^}]+)\}", prompt
        )
        demos: List[Tuple[np.ndarray, float]] = []
        for in_str, out_str in demo_matches:
            try:
                x_vec = np.fromstring(in_str, sep=",")
                y_val = float(out_str)
                demos.append((x_vec, y_val))
            except Exception:
                continue

        curr_match = re.search(r"Now solve for the new input\.\s*Input:\s*\[([^\]]+)\]", prompt)
        if not curr_match:
            all_inputs = re.findall(r"Input:\s*\[([^\]]+)\]", prompt)
            in_str = all_inputs[-1] if all_inputs else "0,0,0,0,0,0"
        else:
            in_str = curr_match.group(1)

        try:
            curr_x = np.fromstring(in_str, sep=",")
        except Exception:
            curr_x = np.zeros(6, dtype=np.float64)

        if self.mode == "demonstration_mimic" and demos:
            best_sim = -float("inf")
            best_guess = demos[0][1]
            curr_norm = np.linalg.norm(curr_x)
            for d_x, d_y in demos:
                d_norm = np.linalg.norm(d_x)
                if curr_norm > 1e-7 and d_norm > 1e-7:
                    sim = float(np.dot(curr_x, d_x) / (curr_norm * d_norm))
                else:
                    sim = 0.0
                if sim > best_sim:
                    best_sim = sim
                    best_guess = d_y
            return f"Guess: boxed{{{best_guess:.4f}}}"

        if len(demos) >= 3:
            X_mat = np.array([d[0] for d in demos])
            Y_vec = np.array([d[1] for d in demos])
            try:
                ridge_lambda = 0.1
                w_est = np.linalg.solve(
                    X_mat.T @ X_mat + ridge_lambda * np.eye(curr_x.shape[0]),
                    X_mat.T @ Y_vec,
                )
                pred = float(np.dot(w_est, curr_x))
            except Exception:
                pred = float(np.mean([d[1] for d in demos]))
        elif len(demos) > 0:
            weights = []
            values = []
            for d_x, d_y in demos:
                dist = np.linalg.norm(curr_x - d_x)
                w = 1.0 / (dist + 1e-3)
                weights.append(w)
                values.append(d_y)
            pred = float(np.average(values, weights=weights))
        else:
            default_w = np.array([0.8, -0.5, 1.2, -1.0, 0.4, -0.7])
            pred = float(np.dot(default_w, curr_x))

        if self.noise_std > 0.0:
            pred += self.rng.normal(0.0, self.noise_std)

        return f"Guess: boxed{{{pred:.4f}}}"

    def _handle_ciciot_prompt(self, prompt: str) -> str:
        """Classify CIC-IoT traffic based on feature signatures in prompt."""
        is_icmp = "ICMP traffic flag: 1" in prompt
        is_udp = "UDP traffic flag: 1" in prompt
        is_tcp = "TCP traffic flag: 1" in prompt
        is_http = "HTTP traffic flag: 1" in prompt
        is_syn = "Number of SYN flags [description: SYN flag value]: 1" in prompt
        is_psh = "Number of PSH flags [description: PSH flag value]: 1" in prompt
        is_rst = "Number of RST flags [description: RST flag value]: 1" in prompt
        is_fin = "Number of FIN flags [description: FIN flag value]: 1" in prompt

        rate_match = re.search(r"Rate \[description: Rate of packet transmission in a flow\]:\s*([0-9.]+)", prompt)
        rate = float(rate_match.group(1)) if rate_match else 10.0

        if is_icmp:
            pred_class = "DDoS-ICMP_Flood"
            reason = "High ICMP traffic rate with ICMP protocol flag set."
        elif is_udp and rate > 100.0:
            pred_class = "DDoS-UDP_Flood"
            reason = "High packet transmission rate over UDP protocol."
        elif is_tcp and is_syn and rate > 100.0:
            pred_class = "DDoS-SYN_Flood"
            reason = "High volume of SYN packets indicating SYN flood."
        elif is_tcp and is_psh:
            pred_class = "DDoS-PSHACK_Flood"
            reason = "PSH and ACK flags set in high-rate flow."
        elif is_tcp and (is_rst or is_fin) and rate > 100.0:
            pred_class = "DDoS-RSTFINFlood"
            reason = "RST and FIN flags observed in flood scenario."
        elif is_tcp and is_http:
            pred_class = "DDoS-HTTP_Flood"
            reason = "HTTP layer flood attack over TCP."
        elif is_tcp and rate > 200.0:
            pred_class = "DDoS-TCP_Flood"
            reason = "Generic high-rate TCP flood without specific handshake flags."
        else:
            pred_class = "BenignTraffic"
            reason = "Low packet rate and standard bidirectional flow features."

        return f"ANALYSIS: {reason}\nANSWER: {pred_class}"


class OpenAILLMClient(BaseLLMClient):
    """OpenAI API LLM client with automatic graceful fallback to mock."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        fallback_to_mock: bool = True,
    ) -> None:
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.fallback_to_mock = fallback_to_mock
        self._mock_client = MockLLMClient(mode="rule_based")

    def complete(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 256,
    ) -> str:
        """Call OpenAI API or fallback to mock."""
        if not self.api_key:
            if self.fallback_to_mock:
                return self._mock_client.complete(
                    prompt, temperature=temperature, max_tokens=max_tokens
                )
            raise ValueError("OPENAI_API_KEY not configured and fallback_to_mock is False.")

        try:
            import openai
            client = openai.OpenAI(api_key=self.api_key)
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content or ""
        except Exception:
            if self.fallback_to_mock:
                return self._mock_client.complete(
                    prompt, temperature=temperature, max_tokens=max_tokens
                )
            raise

class GroqLLMClient(BaseLLMClient):
    """Groq API LLM client with automatic graceful fallback to mock."""

    def __init__(
        self,
        model: str = "openai/gpt-oss-120b",
        api_key: Optional[str] = None,
        fallback_to_mock: bool = True,
    ) -> None:
        self.model = model
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.fallback_to_mock = fallback_to_mock
        self._mock_client = MockLLMClient(mode="rule_based")

    def complete(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> str:
        """Call Groq API or fallback to mock."""
        if not self.api_key:
            if self.fallback_to_mock:
                return self._mock_client.complete(
                    prompt, temperature=temperature, max_tokens=max_tokens
                )
            raise ValueError("GROQ_API_KEY not configured and fallback_to_mock is False.")

        try:
            import groq
            client = groq.Groq(api_key=self.api_key)
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            content = response.choices[0].message.content or ""
            if not content:
                content = getattr(response.choices[0].message, "reasoning", "") or ""
            return content
        except Exception:
            if self.fallback_to_mock:
                return self._mock_client.complete(
                    prompt, temperature=temperature, max_tokens=max_tokens
                )
            raise

def get_llm_client() -> BaseLLMClient:
    """Factory to return the configured LLM Client."""
    provider = os.getenv("LLM_PROVIDER", "openai").lower()
    
    if provider == "groq":
        return GroqLLMClient(
            model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
            api_key=os.getenv("GROQ_API_KEY")
        )
    else:
        return OpenAILLMClient(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            api_key=os.getenv("OPENAI_API_KEY")
        )


class BaseAgent(ABC):
    """Abstract Base Agent executing queries with retrieved demonstrations."""

    def __init__(self, llm_client: Optional[BaseLLMClient] = None) -> None:
        self.llm_client = llm_client or MockLLMClient()

    @abstractmethod
    def generate_prompt(
        self, query: TaskQuery, demonstrations: List[Demonstration]
    ) -> str:
        """Construct full in-context prompt."""
        pass

    @abstractmethod
    def parse_output(self, raw_output: str) -> Any:
        """Extract structured prediction from raw output trajectory."""
        pass

    def act(
        self,
        query: TaskQuery,
        demonstrations: List[Demonstration],
        temperature: float = 0.0,
    ) -> Tuple[Any, str]:
        """Execute one step: generate prompt, call LLM, and parse output."""
        prompt = self.generate_prompt(query, demonstrations)
        raw_output = self.llm_client.complete(prompt, temperature=temperature)
        prediction = self.parse_output(raw_output)
        return prediction, raw_output


class RegAgent(BaseAgent):
    """Synthetic 6D Gaussian Linear Regression Agent."""

    def generate_prompt(
        self, query: TaskQuery, demonstrations: List[Demonstration]
    ) -> str:
        return format_regagent_prompt(query, demonstrations)

    def parse_output(self, raw_output: str) -> Optional[float]:
        """Extract numerical prediction from boxed{...} or Guess: ..."""
        if not raw_output:
            return None

        boxed_match = re.search(r"\\?boxed\{([^}]+)\}", raw_output)
        if boxed_match:
            content = boxed_match.group(1).strip()
            num_match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", content)
            if num_match:
                try:
                    return float(num_match.group(0))
                except ValueError:
                    pass

        guess_match = re.search(
            r"Guess:\s*[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", raw_output, re.IGNORECASE
        )
        if guess_match:
            num_match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", guess_match.group(0))
            if num_match:
                try:
                    return float(num_match.group(0))
                except ValueError:
                    pass

        generic_match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", raw_output)
        if generic_match:
            try:
                return float(generic_match.group(0))
            except ValueError:
                pass

        return None


class CICIOTAgent(BaseAgent):
    """8-class IoT Network Traffic Classification Agent."""

    def __init__(
        self,
        llm_client: Optional[BaseLLMClient] = None,
        valid_classes: Optional[List[str]] = None,
    ) -> None:
        super().__init__(llm_client=llm_client)
        self.valid_classes = valid_classes or list(CICIOT_CLASSES)

    def generate_prompt(
        self, query: TaskQuery, demonstrations: List[Demonstration]
    ) -> str:
        return format_ciciot_prompt(query, demonstrations)

    def parse_output(self, raw_output: str) -> str:
        """Extract ANSWER: <traffic_type> and normalize to valid class."""
        if not raw_output:
            return "Unknown"

        ans_match = re.search(r"ANSWER:\s*([^\n\r]+)", raw_output, re.IGNORECASE)
        extracted = ans_match.group(1).strip() if ans_match else raw_output.strip()

        extracted_canon = canonical_label(extracted)
        for vc in self.valid_classes:
            if canonical_label(vc) == extracted_canon:
                return vc

        for vc in self.valid_classes:
            if canonical_label(vc) in extracted_canon:
                return vc

        raw_canon = canonical_label(raw_output)
        for vc in self.valid_classes:
            if canonical_label(vc) in raw_canon:
                return vc

        return extracted if extracted else "Unknown"

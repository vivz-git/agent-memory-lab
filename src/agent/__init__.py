"""Agent module implementing core architectures, prompts, and orchestration."""

from src.agent.core import (
    BaseAgent,
    RegAgent,
    CICIOTAgent,
    BaseLLMClient,
    MockLLMClient,
    OpenAILLMClient,
    Demonstration,
)
from src.agent.orchestrator import (
    AgentOrchestrator,
    StepResult,
    AdaptiveReadFilter,
    SimpleEpisodicMemoryBank,
)
from src.agent.prompts import (
    format_regagent_prompt,
    format_ciciot_prompt,
    format_ciciot_features_block,
    REGAGENT_PROMPT_TEMPLATE,
    CICIOT_PROMPT_TEMPLATE,
)

__all__ = [
    "BaseAgent",
    "RegAgent",
    "CICIOTAgent",
    "BaseLLMClient",
    "MockLLMClient",
    "OpenAILLMClient",
    "Demonstration",
    "AgentOrchestrator",
    "StepResult",
    "AdaptiveReadFilter",
    "SimpleEpisodicMemoryBank",
    "format_regagent_prompt",
    "format_ciciot_prompt",
    "format_ciciot_features_block",
    "REGAGENT_PROMPT_TEMPLATE",
    "CICIOT_PROMPT_TEMPLATE",
]

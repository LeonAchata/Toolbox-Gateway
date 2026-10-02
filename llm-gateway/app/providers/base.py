from __future__ import annotations

import asyncio
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from ..config import Settings
from ..schemas import FinishReason, Message, ToolCall, ToolSpec, Usage


@dataclass
class ProviderResult:
    content: str
    finish_reason: FinishReason
    usage: Usage
    model_id: str
    tool_calls: list[ToolCall] = field(default_factory=list)


@dataclass(frozen=True)
class Price:
    """USD per million tokens."""

    input: float
    output: float


class Provider(ABC):
    #: Canonical alias clients use to select this provider.
    name: str
    #: Vendor label shown in listings.
    vendor: str
    #: Extra names accepted for backwards compatibility.
    aliases: tuple[str, ...] = ()
    #: Known list prices keyed by model id prefix. Longest prefix wins.
    prices: dict[str, Price] = {}

    def __init__(self, settings: Settings):
        self.settings = settings
        self._semaphore = asyncio.Semaphore(settings.max_concurrent_requests_per_provider)

    @property
    @abstractmethod
    def model_id(self) -> str:
        """Exact upstream model identifier."""

    @abstractmethod
    def unavailable_reason(self) -> str | None:
        """Return why the provider cannot be used, or None if it is ready."""

    @abstractmethod
    async def _generate(
        self,
        messages: list[Message],
        tools: list[ToolSpec],
        temperature: float,
        max_tokens: int,
    ) -> ProviderResult: ...

    async def generate(
        self,
        messages: list[Message],
        tools: list[ToolSpec],
        temperature: float,
        max_tokens: int,
    ) -> ProviderResult:
        async with self._semaphore:
            return await self._generate(messages, tools, temperature, max_tokens)

    def estimate_cost(self, usage: Usage, model_id: str | None = None) -> float:
        model_id = model_id or self.model_id
        matches = [prefix for prefix in self.prices if prefix in model_id]
        if not matches:
            return 0.0
        price = self.prices[max(matches, key=len)]
        return (usage.input_tokens * price.input + usage.output_tokens * price.output) / 1e6


def new_call_id() -> str:
    return f"call_{uuid.uuid4().hex[:24]}"


def system_prompt(messages: list[Message]) -> str | None:
    parts = [m.content for m in messages if m.role == "system" and m.content.strip()]
    return "\n\n".join(parts) or None


def tool_names_by_call_id(messages: list[Message]) -> dict[str, str]:
    return {call.id: call.name for m in messages for call in m.tool_calls}

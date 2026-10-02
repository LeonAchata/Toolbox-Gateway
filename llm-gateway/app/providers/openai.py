from __future__ import annotations

import json
import logging
from typing import Any

import openai
from langsmith import traceable

from ..errors import ProviderError, status_for_upstream
from ..schemas import FinishReason, Message, ToolCall, ToolSpec, Usage
from .base import Price, Provider, ProviderResult, new_call_id

logger = logging.getLogger(__name__)

_FINISH_REASONS: dict[str, FinishReason] = {
    "stop": "stop",
    "tool_calls": "tool_calls",
    "function_call": "tool_calls",
    "length": "length",
    "content_filter": "content_filter",
}
# Reasoning models reject a custom temperature.
_FIXED_TEMPERATURE_PREFIXES = ("o1", "o3", "o4", "gpt-5")


def to_openai_messages(messages: list[Message]) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    for message in messages:
        if message.role == "tool":
            converted.append(
                {
                    "role": "tool",
                    "tool_call_id": message.tool_call_id,
                    "content": message.content,
                }
            )
        elif message.role == "assistant" and message.tool_calls:
            converted.append(
                {
                    "role": "assistant",
                    "content": message.content or None,
                    "tool_calls": [
                        {
                            "id": call.id,
                            "type": "function",
                            "function": {
                                "name": call.name,
                                "arguments": json.dumps(call.arguments),
                            },
                        }
                        for call in message.tool_calls
                    ],
                }
            )
        else:
            converted.append({"role": message.role, "content": message.content})
    return converted


def to_openai_tools(tools: list[ToolSpec]) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.input_schema,
            },
        }
        for tool in tools
    ]


def parse_arguments(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("Model produced invalid JSON arguments: %s", raw[:200])
        return {"_invalid_json": raw}
    return parsed if isinstance(parsed, dict) else {"value": parsed}


class OpenAIProvider(Provider):
    name = "openai"
    vendor = "OpenAI"
    aliases = ("gpt-4o", "gpt", "chatgpt")
    prices = {
        "gpt-4o": Price(2.50, 10.00),
        "gpt-4o-mini": Price(0.15, 0.60),
        "gpt-4.1": Price(2.00, 8.00),
        "gpt-4.1-mini": Price(0.40, 1.60),
        "gpt-4.1-nano": Price(0.10, 0.40),
    }

    def __init__(self, settings):
        super().__init__(settings)
        self._client: openai.AsyncOpenAI | None = None

    @property
    def model_id(self) -> str:
        return self.settings.openai_model

    def unavailable_reason(self) -> str | None:
        return None if self.settings.openai_api_key else "OPENAI_API_KEY is not set"

    @property
    def client(self) -> openai.AsyncOpenAI:
        if self._client is None:
            self._client = openai.AsyncOpenAI(
                api_key=self.settings.openai_api_key,
                organization=self.settings.openai_org_id,
                base_url=self.settings.openai_base_url,
                timeout=self.settings.request_timeout_seconds,
                max_retries=2,
            )
        return self._client

    @traceable(run_type="llm", name="openai.chat.completions")
    async def _generate(
        self,
        messages: list[Message],
        tools: list[ToolSpec],
        temperature: float,
        max_tokens: int,
    ) -> ProviderResult:
        params: dict[str, Any] = {
            "model": self.model_id,
            "messages": to_openai_messages(messages),
            "max_completion_tokens": max_tokens,
        }
        if not self.model_id.startswith(_FIXED_TEMPERATURE_PREFIXES):
            params["temperature"] = temperature
        if tools:
            params["tools"] = to_openai_tools(tools)

        try:
            response = await self.client.chat.completions.create(**params)
        except openai.APITimeoutError as exc:
            raise ProviderError("OpenAI request timed out", 504) from exc
        except openai.APIConnectionError as exc:
            raise ProviderError(f"Could not reach OpenAI: {exc}") from exc
        except openai.APIStatusError as exc:
            raise ProviderError(
                f"OpenAI returned {exc.status_code}: {exc.message}",
                status_for_upstream(exc.status_code),
            ) from exc

        if not response.choices:
            raise ProviderError("OpenAI returned no choices")
        choice = response.choices[0]
        message = choice.message
        tool_calls = [
            ToolCall(
                id=call.id or new_call_id(),
                name=call.function.name,
                arguments=parse_arguments(call.function.arguments),
            )
            for call in (message.tool_calls or [])
            if getattr(call, "type", "function") == "function"
        ]
        usage = response.usage
        return ProviderResult(
            content=message.content or message.refusal or "",
            tool_calls=tool_calls,
            finish_reason=_FINISH_REASONS.get(choice.finish_reason or "stop", "stop"),
            usage=Usage(
                input_tokens=usage.prompt_tokens if usage else 0,
                output_tokens=usage.completion_tokens if usage else 0,
                total_tokens=usage.total_tokens if usage else 0,
            ),
            model_id=response.model or self.model_id,
        )

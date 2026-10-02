from __future__ import annotations

import asyncio
import logging
from typing import Any

import boto3
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError, NoCredentialsError, ReadTimeoutError
from langsmith import traceable

from ..errors import ProviderError
from ..schemas import FinishReason, Message, ToolCall, ToolSpec, Usage
from .base import Price, Provider, ProviderResult, new_call_id, system_prompt

logger = logging.getLogger(__name__)

_FINISH_REASONS: dict[str, FinishReason] = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "tool_use": "tool_calls",
    "max_tokens": "length",
    "guardrail_intervened": "content_filter",
    "content_filtered": "content_filter",
}
_ERROR_STATUS = {
    "ThrottlingException": 429,
    "ServiceQuotaExceededException": 429,
    "ModelTimeoutException": 504,
    "ValidationException": 400,
    "ModelNotReadyException": 503,
    "ServiceUnavailableException": 503,
}


def _append(conversation: list[dict[str, Any]], role: str, blocks: list[dict[str, Any]]) -> None:
    # Converse requires user and assistant turns to alternate, so consecutive
    # messages with the same role (e.g. several tool results) are merged.
    if conversation and conversation[-1]["role"] == role:
        conversation[-1]["content"].extend(blocks)
    else:
        conversation.append({"role": role, "content": blocks})


def to_converse_messages(messages: list[Message]) -> list[dict[str, Any]]:
    conversation: list[dict[str, Any]] = []
    for message in messages:
        if message.role == "system":
            continue
        if message.role == "user":
            _append(conversation, "user", [{"text": message.content or " "}])
        elif message.role == "tool":
            _append(
                conversation,
                "user",
                [
                    {
                        "toolResult": {
                            "toolUseId": message.tool_call_id,
                            "content": [{"text": message.content or "(empty)"}],
                        }
                    }
                ],
            )
        else:
            blocks: list[dict[str, Any]] = []
            if message.content.strip():
                blocks.append({"text": message.content})
            blocks.extend(
                {"toolUse": {"toolUseId": call.id, "name": call.name, "input": call.arguments}}
                for call in message.tool_calls
            )
            _append(conversation, "assistant", blocks or [{"text": " "}])

    if conversation and conversation[0]["role"] != "user":
        conversation.insert(0, {"role": "user", "content": [{"text": "(conversation start)"}]})
    return conversation


def to_tool_config(tools: list[ToolSpec]) -> dict[str, Any]:
    return {
        "tools": [
            {
                "toolSpec": {
                    "name": tool.name,
                    "description": tool.description or tool.name,
                    "inputSchema": {"json": tool.input_schema},
                }
            }
            for tool in tools
        ]
    }


class BedrockProvider(Provider):
    name = "bedrock"
    vendor = "AWS Bedrock"
    aliases = ("bedrock-nova-pro", "nova", "aws")
    prices = {
        "nova-pro": Price(0.80, 3.20),
        "nova-lite": Price(0.06, 0.24),
        "nova-micro": Price(0.035, 0.14),
        "nova-premier": Price(2.50, 12.50),
    }

    def __init__(self, settings):
        super().__init__(settings)
        self._client = None
        self._has_credentials: bool | None = None
        self._session = boto3.Session(
            aws_access_key_id=settings.aws_access_key_id,
            aws_secret_access_key=settings.aws_secret_access_key,
            aws_session_token=settings.aws_session_token,
            region_name=settings.aws_region,
        )

    @property
    def model_id(self) -> str:
        return self.settings.bedrock_model_id

    def unavailable_reason(self) -> str | None:
        # Resolved once: the default chain may probe the instance metadata
        # service, which is slow outside AWS.
        if self._has_credentials is None:
            try:
                self._has_credentials = self._session.get_credentials() is not None
            except BotoCoreError:
                self._has_credentials = False
        return None if self._has_credentials else "No AWS credentials found"

    @property
    def client(self):
        if self._client is None:
            self._client = self._session.client(
                "bedrock-runtime",
                config=Config(
                    read_timeout=self.settings.request_timeout_seconds,
                    connect_timeout=10,
                    retries={"max_attempts": 3, "mode": "standard"},
                ),
            )
        return self._client

    @traceable(run_type="llm", name="bedrock.converse")
    async def _generate(
        self,
        messages: list[Message],
        tools: list[ToolSpec],
        temperature: float,
        max_tokens: int,
    ) -> ProviderResult:
        params: dict[str, Any] = {
            "modelId": self.model_id,
            "messages": to_converse_messages(messages),
            "inferenceConfig": {
                "maxTokens": max_tokens,
                # Bedrock models accept 0..1.
                "temperature": min(temperature, 1.0),
            },
        }
        system = system_prompt(messages)
        if system:
            params["system"] = [{"text": system}]
        if tools:
            params["toolConfig"] = to_tool_config(tools)

        try:
            # boto3 is synchronous; keep it off the event loop.
            response = await asyncio.to_thread(self.client.converse, **params)
        except ClientError as exc:
            error = exc.response.get("Error", {})
            code = error.get("Code", "Unknown")
            raise ProviderError(
                f"Bedrock {code}: {error.get('Message', str(exc))}",
                _ERROR_STATUS.get(code, 502),
            ) from exc
        except NoCredentialsError as exc:
            raise ProviderError("No AWS credentials found", 503) from exc
        except ReadTimeoutError as exc:
            raise ProviderError("Bedrock request timed out", 504) from exc
        except BotoCoreError as exc:
            raise ProviderError(f"Bedrock client error: {exc}") from exc

        blocks = response.get("output", {}).get("message", {}).get("content", [])
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        for block in blocks:
            if "text" in block:
                text_parts.append(block["text"])
            elif "toolUse" in block:
                use = block["toolUse"]
                tool_calls.append(
                    ToolCall(
                        id=use.get("toolUseId") or new_call_id(),
                        name=use["name"],
                        arguments=use.get("input") or {},
                    )
                )

        usage = response.get("usage", {})
        input_tokens = usage.get("inputTokens", 0)
        output_tokens = usage.get("outputTokens", 0)
        return ProviderResult(
            content="".join(text_parts).strip(),
            tool_calls=tool_calls,
            finish_reason=_FINISH_REASONS.get(response.get("stopReason", ""), "stop"),
            usage=Usage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=usage.get("totalTokens", input_tokens + output_tokens),
            ),
            model_id=self.model_id,
        )

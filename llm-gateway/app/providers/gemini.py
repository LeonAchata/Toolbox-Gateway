from __future__ import annotations

import base64
import logging

import httpx
from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from langsmith import traceable

from ..errors import ProviderError, status_for_upstream
from ..schemas import FinishReason, Message, ToolCall, ToolSpec, Usage
from .base import (
    Price,
    Provider,
    ProviderResult,
    new_call_id,
    system_prompt,
    tool_names_by_call_id,
)

logger = logging.getLogger(__name__)

_FILTERED = {
    "SAFETY",
    "RECITATION",
    "BLOCKLIST",
    "PROHIBITED_CONTENT",
    "SPII",
    "IMAGE_SAFETY",
}


def _append(contents: list[types.Content], role: str, parts: list[types.Part]) -> None:
    # Parallel function responses must travel in a single turn, and Gemini
    # expects user and model turns to alternate.
    if contents and contents[-1].role == role:
        contents[-1].parts = [*(contents[-1].parts or []), *parts]
    else:
        contents.append(types.Content(role=role, parts=parts))


def to_gemini_contents(messages: list[Message]) -> list[types.Content]:
    names = tool_names_by_call_id(messages)
    contents: list[types.Content] = []
    for message in messages:
        if message.role == "system":
            continue
        if message.role == "user":
            _append(contents, "user", [types.Part(text=message.content or " ")])
        elif message.role == "tool":
            name = message.name or names.get(message.tool_call_id or "", "tool")
            _append(
                contents,
                "user",
                [
                    types.Part(
                        function_response=types.FunctionResponse(
                            name=name, response={"result": message.content}
                        )
                    )
                ],
            )
        else:
            parts: list[types.Part] = []
            if message.content.strip():
                parts.append(types.Part(text=message.content))
            for call in message.tool_calls:
                parts.append(
                    types.Part(
                        function_call=types.FunctionCall(name=call.name, args=call.arguments),
                        thought_signature=base64.b64decode(call.signature)
                        if call.signature
                        else None,
                    )
                )
            _append(contents, "model", parts or [types.Part(text=" ")])
    return contents


def to_gemini_tools(tools: list[ToolSpec]) -> list[types.Tool]:
    if not tools:
        return []
    return [
        types.Tool(
            function_declarations=[
                types.FunctionDeclaration(
                    name=tool.name,
                    description=tool.description,
                    parameters_json_schema=tool.input_schema,
                )
                for tool in tools
            ]
        )
    ]


def _finish_reason(name: str | None, has_tool_calls: bool) -> FinishReason:
    if has_tool_calls:
        return "tool_calls"
    if name == "MAX_TOKENS":
        return "length"
    if name in _FILTERED:
        return "content_filter"
    return "stop"


class GeminiProvider(Provider):
    name = "gemini"
    vendor = "Google Gemini"
    aliases = ("gemini-pro", "google")
    prices = {
        "gemini-2.5-pro": Price(1.25, 10.00),
        "gemini-2.5-flash": Price(0.30, 2.50),
        "gemini-2.5-flash-lite": Price(0.10, 0.40),
        "gemini-2.0-flash": Price(0.10, 0.40),
    }

    def __init__(self, settings):
        super().__init__(settings)
        self._client: genai.Client | None = None

    @property
    def model_id(self) -> str:
        return self.settings.gemini_model

    def unavailable_reason(self) -> str | None:
        return None if self.settings.google_api_key else "GOOGLE_API_KEY is not set"

    @property
    def client(self) -> genai.Client:
        if self._client is None:
            self._client = genai.Client(
                api_key=self.settings.google_api_key,
                http_options=types.HttpOptions(
                    timeout=int(self.settings.request_timeout_seconds * 1000),
                    retry_options=types.HttpRetryOptions(attempts=3),
                ),
            )
        return self._client

    @traceable(run_type="llm", name="gemini.generate_content")
    async def _generate(
        self,
        messages: list[Message],
        tools: list[ToolSpec],
        temperature: float,
        max_tokens: int,
    ) -> ProviderResult:
        config = types.GenerateContentConfig(
            system_instruction=system_prompt(messages),
            temperature=temperature,
            max_output_tokens=max_tokens,
            tools=to_gemini_tools(tools) or None,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )
        try:
            response = await self.client.aio.models.generate_content(
                model=self.model_id,
                contents=to_gemini_contents(messages),
                config=config,
            )
        except genai_errors.APIError as exc:
            raise ProviderError(
                f"Gemini returned {exc.code}: {exc.message}", status_for_upstream(exc.code)
            ) from exc
        except httpx.TimeoutException as exc:
            raise ProviderError("Gemini request timed out", 504) from exc
        except httpx.HTTPError as exc:
            raise ProviderError(f"Could not reach Gemini: {exc}") from exc

        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        finish_name: str | None = None
        candidates = response.candidates or []
        if candidates:
            candidate = candidates[0]
            finish_name = candidate.finish_reason.name if candidate.finish_reason else None
            for part in (candidate.content.parts if candidate.content else None) or []:
                if part.function_call:
                    call = part.function_call
                    tool_calls.append(
                        ToolCall(
                            id=call.id or new_call_id(),
                            name=call.name or "",
                            arguments=dict(call.args or {}),
                            signature=base64.b64encode(part.thought_signature).decode()
                            if part.thought_signature
                            else None,
                        )
                    )
                elif part.text and not part.thought:
                    text_parts.append(part.text)
        elif response.prompt_feedback and response.prompt_feedback.block_reason:
            finish_name = "SAFETY"

        meta = response.usage_metadata
        input_tokens = (meta.prompt_token_count or 0) if meta else 0
        # Thinking tokens are billed as output.
        output_tokens = (
            ((meta.candidates_token_count or 0) + (meta.thoughts_token_count or 0)) if meta else 0
        )
        return ProviderResult(
            content="".join(text_parts).strip(),
            tool_calls=tool_calls,
            finish_reason=_finish_reason(finish_name, bool(tool_calls)),
            usage=Usage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=(meta.total_token_count if meta else None)
                or input_tokens + output_tokens,
            ),
            model_id=response.model_version or self.model_id,
        )

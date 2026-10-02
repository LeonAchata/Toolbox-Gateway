"""Wire format of the gateway.

The message format is provider neutral and close to the OpenAI chat format,
with tool calls carried as structured objects instead of JSON strings. Each
provider adapter translates to and from its native representation.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

FinishReason = Literal["stop", "tool_calls", "length", "content_filter", "error"]


class ToolCall(BaseModel):
    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    # Opaque provider data that must be sent back unchanged with the call in
    # later turns (Gemini thought signatures). Clients should not inspect it.
    signature: str | None = None


class Message(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    # Only for role="tool": which call this is the result of, and the tool name.
    tool_call_id: str | None = None
    name: str | None = None

    @model_validator(mode="after")
    def _check_role_fields(self) -> Message:
        if self.role == "tool" and not self.tool_call_id:
            raise ValueError("tool messages need a tool_call_id")
        if self.tool_calls and self.role != "assistant":
            raise ValueError("only assistant messages can carry tool_calls")
        return self


class ToolSpec(BaseModel):
    name: str = Field(..., pattern=r"^[a-zA-Z0-9_-]{1,64}$")
    description: str = ""
    input_schema: dict[str, Any] = Field(
        default_factory=lambda: {"type": "object", "properties": {}}
    )


class GenerateRequest(BaseModel):
    model: str | None = Field(None, description="Model alias; omit to use the default")
    messages: list[Message] = Field(..., min_length=1)
    tools: list[ToolSpec] = Field(default_factory=list)
    temperature: float = Field(0.3, ge=0.0, le=2.0)
    max_tokens: int = Field(1024, gt=0, le=32_000)
    use_cache: bool = True


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


class GenerateResponse(BaseModel):
    id: str
    model: str = Field(..., description="Gateway alias that served the request")
    provider: str
    provider_model: str = Field(..., description="Exact upstream model id")
    content: str
    tool_calls: list[ToolCall] = Field(default_factory=list)
    finish_reason: FinishReason
    usage: Usage
    cached: bool
    latency_ms: float
    estimated_cost_usd: float


class ModelInfo(BaseModel):
    name: str
    provider: str
    provider_model: str
    available: bool
    reason: str | None = None
    default: bool = False
    aliases: list[str] = Field(default_factory=list)
    supports_tools: bool = True

"""HTTP agent: one request in, one complete answer (plus its trace) out."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import HTTPException, Request
from pydantic import AliasChoices, BaseModel, Field

from agent_core import AgentSettings, GatewayError, ToolboxUnavailableError, create_base_app

logger = logging.getLogger("agent-http")

settings = AgentSettings(service_name="agent-http")
app = create_base_app(
    title="Agent (HTTP)",
    description="LangGraph tool-calling agent behind a request/response API.",
    settings=settings,
)


class ChatRequest(BaseModel):
    # "input" is accepted for compatibility with the original /process API.
    message: str = Field(
        ..., min_length=1, max_length=8000, validation_alias=AliasChoices("message", "input")
    )
    model: str | None = Field(None, description="Model alias, e.g. openai, gemini or bedrock")
    session_id: str | None = Field(
        None, max_length=128, description="Reuse to continue a conversation"
    )


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    model: str | None
    provider_model: str | None
    usage: dict[str, Any]
    trace: list[dict[str, Any]]


@app.post("/chat", response_model=ChatResponse)
@app.post("/process", response_model=ChatResponse, include_in_schema=False)
async def chat(body: ChatRequest, request: Request) -> dict[str, Any]:
    runtime = request.app.state.runtime
    try:
        return await runtime.run(body.message, model=body.model, session_id=body.session_id)
    except GatewayError as exc:
        raise HTTPException(exc.status_code, exc.message) from exc
    except ToolboxUnavailableError as exc:
        raise HTTPException(503, str(exc)) from exc

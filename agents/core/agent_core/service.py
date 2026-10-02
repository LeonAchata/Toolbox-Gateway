"""Pieces shared by the HTTP and WebSocket agents."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import uuid
from collections import OrderedDict
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from .config import AgentSettings
from .gateway import GatewayClient, GatewayError
from .graph import build_agent
from .models import detect_model
from .observability import setup_logging, setup_tracing
from .toolbox import ToolboxClient, ToolboxUnavailableError

logger = logging.getLogger(__name__)


class AgentRuntime:
    """Owns the clients, the compiled graph and the conversation memory."""

    def __init__(
        self,
        settings: AgentSettings,
        gateway: GatewayClient | None = None,
        toolbox: ToolboxClient | None = None,
    ):
        self.settings = settings
        self.gateway = gateway or GatewayClient(
            settings.llm_gateway_url, api_key=settings.gateway_api_key
        )
        self.toolbox = toolbox or ToolboxClient(settings.toolbox_url)
        self.checkpointer = InMemorySaver()
        self.graph = build_agent(settings, self.gateway, self.toolbox, self.checkpointer)
        self._sessions: OrderedDict[str, None] = OrderedDict()

    async def close(self) -> None:
        await self.gateway.close()
        await self.toolbox.close()

    def _touch_session(self, session_id: str) -> None:
        self._sessions[session_id] = None
        self._sessions.move_to_end(session_id)
        while len(self._sessions) > self.settings.max_sessions:
            evicted, _ = self._sessions.popitem(last=False)
            self.checkpointer.delete_thread(evicted)

    def forget(self, session_id: str) -> bool:
        if session_id not in self._sessions:
            return False
        del self._sessions[session_id]
        self.checkpointer.delete_thread(session_id)
        return True

    def resolve_model(self, message: str, requested: str | None) -> str | None:
        return requested or detect_model(message) or self.settings.default_model or None

    def _prepare(self, message: str, model: str | None, session_id: str | None):
        session_id = session_id or uuid.uuid4().hex
        self._touch_session(session_id)
        state = {
            "messages": [HumanMessage(content=message)],
            "model": self.resolve_model(message, model),
            "rounds": 0,
            "trace": [],
            "answer": None,
        }
        config = {"configurable": {"thread_id": session_id}, "recursion_limit": 50}
        return session_id, state, config

    async def run(
        self, message: str, *, model: str | None = None, session_id: str | None = None
    ) -> dict[str, Any]:
        session_id, state, config = self._prepare(message, model, session_id)
        result = await self.graph.ainvoke(state, config)
        return {"session_id": session_id, **summarize(result["trace"]), "answer": result["answer"]}

    async def stream(
        self, message: str, *, model: str | None = None, session_id: str | None = None
    ) -> AsyncIterator[dict[str, Any]]:
        session_id, state, config = self._prepare(message, model, session_id)
        async for event in self.graph.astream(state, config, stream_mode="custom"):
            yield event

    async def status(self) -> dict[str, Any]:
        toolbox_ok, gateway_ok = await asyncio.gather(self.toolbox.ping(), self.gateway.ping())
        if toolbox_ok and not self.toolbox.loaded:
            # Best effort: a failure here already shows up as tools_loaded == 0.
            with contextlib.suppress(Exception):
                await self.toolbox.refresh()
        return {
            "status": "ok" if toolbox_ok and gateway_ok else "degraded",
            "service": self.settings.service_name,
            "dependencies": {
                "toolbox": "ok" if toolbox_ok else "unreachable",
                "llm_gateway": "ok" if gateway_ok else "unreachable",
            },
            "tools_loaded": self.toolbox.tool_count,
            "active_sessions": len(self._sessions),
        }


def summarize(trace: list[dict[str, Any]]) -> dict[str, Any]:
    llm_events = [e for e in trace if e["type"] == "llm"]
    tool_events = [e for e in trace if e["type"] == "tool"]
    return {
        "model": llm_events[-1]["model"] if llm_events else None,
        "provider_model": llm_events[-1]["provider_model"] if llm_events else None,
        "usage": {
            "llm_calls": len(llm_events),
            "tool_calls": len(tool_events),
            "input_tokens": sum(e["usage"]["input_tokens"] for e in llm_events),
            "output_tokens": sum(e["usage"]["output_tokens"] for e in llm_events),
            "cost_usd": round(sum(e["cost_usd"] for e in llm_events), 8),
            "latency_ms": round(
                sum(e["latency_ms"] for e in llm_events)
                + sum(e["duration_ms"] for e in tool_events),
                2,
            ),
        },
        "trace": trace,
    }


def create_base_app(
    *, title: str, description: str, settings: AgentSettings, runtime: AgentRuntime | None = None
) -> FastAPI:
    setup_logging(settings.log_level)
    tracing = setup_tracing(settings.service_name)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.runtime = runtime or AgentRuntime(settings)
        logger.info("LangSmith tracing %s", "enabled" if tracing else "disabled")
        # Load the tool catalog in the background so a slow toolbox does not
        # block startup; requests load it lazily if this has not finished.
        connect = asyncio.create_task(app.state.runtime.toolbox.connect())
        yield
        connect.cancel()
        await app.state.runtime.close()

    app = FastAPI(title=title, description=description, version="3.0.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return await app.state.runtime.status()

    @app.get("/models")
    async def models() -> dict[str, Any]:
        try:
            return {"models": await app.state.runtime.gateway.models()}
        except GatewayError as exc:
            raise HTTPException(exc.status_code, exc.message) from exc

    @app.get("/tools")
    async def tools() -> dict[str, Any]:
        try:
            return {"tools": await app.state.runtime.toolbox.tools()}
        except ToolboxUnavailableError as exc:
            raise HTTPException(503, str(exc)) from exc

    @app.get("/metrics")
    async def metrics() -> dict[str, Any]:
        try:
            return await app.state.runtime.gateway.metrics()
        except GatewayError as exc:
            raise HTTPException(exc.status_code, exc.message) from exc

    @app.delete("/sessions/{session_id}")
    async def forget_session(session_id: str) -> dict[str, bool]:
        return {"deleted": app.state.runtime.forget(session_id)}

    return app

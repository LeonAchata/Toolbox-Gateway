"""WebSocket agent: streams every step of the agent loop as it happens.

Client -> server
    {"type": "message", "content": "...", "model": "gemini"}   model is optional
    {"type": "reset"}                                          forget the conversation
    {"type": "ping"}

Server -> client
    connected, start, llm_start, llm, tool, answer, done, error, reset, pong
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Any

from fastapi import WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field, ValidationError

from agent_core import AgentSettings, GatewayError, ToolboxUnavailableError, create_base_app
from agent_core.service import summarize

logger = logging.getLogger("agent-websocket")

settings = AgentSettings(service_name="agent-websocket")
app = create_base_app(
    title="Agent (WebSocket)",
    description="LangGraph tool-calling agent that streams its steps over a WebSocket.",
    settings=settings,
)
active_connections: set[str] = set()


class ClientMessage(BaseModel):
    type: str = "message"
    content: str = Field("", max_length=8000)
    model: str | None = None


async def run_turn(websocket: WebSocket, session_id: str, message: ClientMessage) -> None:
    runtime = websocket.app.state.runtime
    started = time.perf_counter()
    trace: list[dict[str, Any]] = []
    await websocket.send_json(
        {"type": "start", "model": runtime.resolve_model(message.content, message.model)}
    )
    try:
        async for event in runtime.stream(
            message.content, model=message.model, session_id=session_id
        ):
            if event["type"] != "llm_start":
                trace.append(event)
            await websocket.send_json(event)
    except GatewayError as exc:
        await websocket.send_json(
            {"type": "error", "message": exc.message, "status": exc.status_code}
        )
        return
    except ToolboxUnavailableError as exc:
        await websocket.send_json({"type": "error", "message": str(exc), "status": 503})
        return
    except Exception:
        logger.exception("Turn failed for session %s", session_id)
        await websocket.send_json(
            {"type": "error", "message": "Internal error while running the agent", "status": 500}
        )
        return

    summary = summarize(trace)
    await websocket.send_json(
        {
            "type": "done",
            "model": summary["model"],
            "provider_model": summary["provider_model"],
            "usage": summary["usage"],
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
        }
    )


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    connection_id = uuid.uuid4().hex[:12]
    session_id = f"ws-{connection_id}"
    runtime = websocket.app.state.runtime
    active_connections.add(connection_id)
    logger.info("Connection %s opened (%d active)", connection_id, len(active_connections))

    busy: asyncio.Task | None = None
    try:
        await websocket.send_json(
            {
                "type": "connected",
                "connection_id": connection_id,
                "tools": runtime.toolbox.tool_count,
                "default_model": settings.default_model or None,
            }
        )
        while True:
            raw = await websocket.receive_text()
            try:
                message = ClientMessage.model_validate_json(raw)
            except ValidationError as exc:
                await websocket.send_json({"type": "error", "message": exc.errors()[0]["msg"]})
                continue

            if message.type == "ping":
                await websocket.send_json({"type": "pong"})
            elif message.type == "reset":
                runtime.forget(session_id)
                session_id = f"ws-{connection_id}-{uuid.uuid4().hex[:6]}"
                await websocket.send_json({"type": "reset"})
            elif message.type == "message":
                if not message.content.strip():
                    await websocket.send_json({"type": "error", "message": "Message is empty"})
                elif busy and not busy.done():
                    await websocket.send_json(
                        {"type": "error", "message": "Still working on the previous message"}
                    )
                else:
                    # Run in a task so pings and resets are still served meanwhile.
                    busy = asyncio.create_task(run_turn(websocket, session_id, message))
            else:
                await websocket.send_json(
                    {"type": "error", "message": f"Unknown message type '{message.type}'"}
                )
    except WebSocketDisconnect:
        pass
    except Exception:
        logger.exception("Connection %s failed", connection_id)
    finally:
        if busy and not busy.done():
            busy.cancel()
        runtime.forget(session_id)
        active_connections.discard(connection_id)
        logger.info("Connection %s closed (%d active)", connection_id, len(active_connections))

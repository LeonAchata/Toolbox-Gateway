"""Toolbox service.

Exposes the tool registry over two transports:

* ``GET /tools`` and ``POST /tools/call``: a small JSON bridge that mirrors the
  MCP ``tools/list`` and ``tools/call`` payloads. The agents use this.
* ``/mcp``: a native MCP endpoint (streamable HTTP, stateless) so any MCP
  client, such as Claude Desktop or the MCP Inspector, can use the same tools.
"""

from __future__ import annotations

import logging
import time
from contextlib import AsyncExitStack, asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from mcp.server.mcpserver import MCPServer
from mcp.server.transport_security import TransportSecuritySettings
from pydantic import BaseModel, Field

from . import tools  # noqa: F401  (registers the built-in tools)
from .config import settings
from .registry import ToolArgumentError, ToolNotFoundError, registry

VERSION = "3.0.0"

logging.basicConfig(
    level=settings.log_level.upper(),
    format="%(asctime)s %(levelname)-7s %(name)s  %(message)s",
)
logger = logging.getLogger("toolbox")


def build_mcp_server() -> MCPServer:
    server = MCPServer(name="toolbox", version=VERSION)
    for tool in registry.all():
        server.add_tool(tool.fn, name=tool.name, description=tool.description)
    return server


mcp_server = build_mcp_server() if settings.mcp_endpoint_enabled else None
mcp_app = (
    mcp_server.streamable_http_app(
        streamable_http_path="/mcp",
        stateless_http=True,
        json_response=True,
        # The service sits on a private Docker network and is reached by
        # container name, so Host-header pinning would only get in the way.
        transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False),
    )
    if mcp_server
    else None
)


@asynccontextmanager
async def lifespan(_: FastAPI):
    async with AsyncExitStack() as stack:
        if mcp_server is not None:
            await stack.enter_async_context(mcp_server.session_manager.run())
        logger.info(
            "Toolbox %s ready with %d tools: %s",
            VERSION,
            len(registry),
            ", ".join(t.name for t in registry.all()),
        )
        yield


app = FastAPI(
    title="Toolbox",
    description="Tool server exposing a JSON bridge and a native MCP endpoint.",
    version=VERSION,
    lifespan=lifespan,
)


class ToolCallRequest(BaseModel):
    name: str = Field(..., min_length=1)
    arguments: dict[str, Any] = Field(default_factory=dict)


class TextContent(BaseModel):
    type: str = "text"
    text: str


class ToolCallResult(BaseModel):
    content: list[TextContent]
    isError: bool = False
    durationMs: float


@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "service": "toolbox",
        "version": VERSION,
        "tools": len(registry),
        "mcp_endpoint": "/mcp" if mcp_server else None,
    }


@app.get("/tools")
async def list_tools() -> dict[str, Any]:
    return {"tools": [tool.describe() for tool in registry.all()]}


@app.post("/tools/call", response_model=ToolCallResult)
async def call_tool(request: ToolCallRequest) -> ToolCallResult:
    try:
        tool = registry.get(request.name)
    except ToolNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    started = time.perf_counter()
    is_error = False
    try:
        text = str(await tool.run(request.arguments))
    except ToolArgumentError as exc:
        # Reported as a tool error rather than an HTTP error, following MCP:
        # the model gets to read the message and fix its own arguments.
        is_error, text = True, f"Invalid arguments: {exc}"
    except ValueError as exc:
        is_error, text = True, str(exc)
    except Exception:
        logger.exception("Tool %s crashed", request.name)
        is_error, text = True, "Internal error while running the tool"

    duration_ms = round((time.perf_counter() - started) * 1000, 2)
    log = logger.warning if is_error else logger.info
    log("%s(%s) -> %s [%.2f ms]", request.name, request.arguments, text[:200], duration_ms)
    return ToolCallResult(
        content=[TextContent(text=text)], isError=is_error, durationMs=duration_ms
    )


if mcp_app is not None:
    # Mounted last so the JSON routes above take precedence.
    app.mount("/", mcp_app)

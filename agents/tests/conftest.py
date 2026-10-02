import json
import sys
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[1]
for sub in ("core", "agent-http", "agent-websocket"):
    sys.path.insert(0, str(ROOT / sub))

from agent_core import AgentRuntime, AgentSettings, GatewayClient, ToolboxClient  # noqa: E402

TOOLS = [
    {
        "name": "add",
        "description": "Add two numbers",
        "inputSchema": {
            "type": "object",
            "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
            "required": ["a", "b"],
        },
    },
    {
        "name": "uppercase",
        "description": "Uppercase text",
        "inputSchema": {"type": "object", "properties": {"text": {"type": "string"}}},
    },
]


def llm_reply(content="", tool_calls=(), model="fake", finish_reason=None, **extra):
    return {
        "id": "gen_1",
        "model": model,
        "provider": "Fake",
        "provider_model": f"{model}-1",
        "content": content,
        "tool_calls": [
            {"id": c[0], "name": c[1], "arguments": c[2], "signature": c[3] if len(c) > 3 else None}
            for c in tool_calls
        ],
        "finish_reason": finish_reason or ("tool_calls" if tool_calls else "stop"),
        "usage": {"input_tokens": 100, "output_tokens": 20, "total_tokens": 120},
        "cached": False,
        "latency_ms": 12.5,
        "estimated_cost_usd": 0.0001,
        **extra,
    }


class FakeServices:
    """Scripted gateway plus an in-memory toolbox, wired through httpx transports."""

    def __init__(self):
        self.replies: list = []
        self.gateway_requests: list[dict] = []
        self.tool_calls: list[tuple[str, dict]] = []

    def gateway_handler(self, request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        if request.url.path == "/models":
            return httpx.Response(200, json={"models": [{"name": "fake", "available": True}]})
        if request.url.path == "/generate":
            self.gateway_requests.append(json.loads(request.content))
            reply = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
            if isinstance(reply, httpx.Response):
                return reply
            return httpx.Response(200, json=reply)
        return httpx.Response(404)

    def toolbox_handler(self, request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        if request.url.path == "/tools":
            return httpx.Response(200, json={"tools": TOOLS})
        if request.url.path == "/tools/call":
            body = json.loads(request.content)
            self.tool_calls.append((body["name"], body["arguments"]))
            args = body["arguments"]
            if body["name"] == "add":
                text, error = str(args["a"] + args["b"]), False
            elif body["name"] == "uppercase":
                text, error = args["text"].upper(), False
            else:
                return httpx.Response(404, json={"detail": f"Unknown tool '{body['name']}'"})
            return httpx.Response(
                200,
                json={"content": [{"type": "text", "text": text}], "isError": error},
            )
        return httpx.Response(404)

    def runtime(self, **overrides) -> AgentRuntime:
        settings = AgentSettings(_env_file=None, **overrides)
        return AgentRuntime(
            settings,
            gateway=GatewayClient(
                "http://gateway", transport=httpx.MockTransport(self.gateway_handler)
            ),
            toolbox=ToolboxClient(
                "http://toolbox", transport=httpx.MockTransport(self.toolbox_handler)
            ),
        )


@pytest.fixture
def services():
    return FakeServices()

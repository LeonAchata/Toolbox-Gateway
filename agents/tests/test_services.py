"""The HTTP and WebSocket front doors."""

import httpx
import pytest
from fastapi.testclient import TestClient

from agent_core import AgentSettings, create_base_app
from conftest import llm_reply


def build(module_name, services):
    module = __import__(module_name)
    runtime = services.runtime()
    module.app.state.runtime = runtime
    # Swap the runtime the lifespan would create for the fake one.
    module.app.router.lifespan_context = _lifespan_with(runtime)
    return module.app


def _lifespan_with(runtime):
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def lifespan(app):
        app.state.runtime = runtime
        await runtime.toolbox.connect(attempts=1)
        yield

    return lifespan


@pytest.fixture
def http_client(services):
    with TestClient(build("http_app", services)) as client:
        yield client


@pytest.fixture
def ws_client(services):
    with TestClient(build("ws_app", services)) as client:
        yield client


def test_http_chat(http_client, services):
    services.replies = [
        llm_reply(tool_calls=[("c1", "add", {"a": 5, "b": 3})]),
        llm_reply(content="5 + 3 = 8"),
    ]
    body = http_client.post("/chat", json={"message": "5 + 3?"}).json()
    assert body["answer"] == "5 + 3 = 8"
    assert body["model"] == "fake"
    assert body["usage"]["tool_calls"] == 1
    assert body["session_id"]

    legacy = http_client.post("/process", json={"input": "again"})
    assert legacy.status_code == 200


def test_http_maps_gateway_errors(http_client, services):
    services.replies = [
        httpx.Response(503, json={"error": {"message": "Model 'openai' is not available"}})
    ]
    response = http_client.post("/chat", json={"message": "hi", "model": "openai"})
    assert response.status_code == 503
    assert "not available" in response.json()["detail"]


def test_http_validates_input(http_client):
    assert http_client.post("/chat", json={"message": ""}).status_code == 422


def test_health_models_and_tools(http_client):
    health = http_client.get("/health").json()
    assert health["status"] == "ok"
    assert health["tools_loaded"] == 2
    assert http_client.get("/models").json()["models"][0]["name"] == "fake"
    assert len(http_client.get("/tools").json()["tools"]) == 2


def test_websocket_streams_a_turn(ws_client, services):
    services.replies = [
        llm_reply(tool_calls=[("c1", "uppercase", {"text": "mcp"})]),
        llm_reply(content="MCP"),
    ]
    with ws_client.websocket_connect("/ws") as ws:
        assert ws.receive_json()["type"] == "connected"
        ws.send_json({"type": "ping"})
        assert ws.receive_json() == {"type": "pong"}

        ws.send_json({"type": "message", "content": "uppercase mcp"})
        events = []
        while True:
            event = ws.receive_json()
            events.append(event)
            if event["type"] in ("done", "error"):
                break

    types = [e["type"] for e in events]
    assert types == ["start", "llm_start", "llm", "tool", "llm_start", "llm", "answer", "done"]
    assert events[3]["result"] == "MCP"
    assert events[-1]["usage"]["llm_calls"] == 2


def test_websocket_rejects_bad_messages(ws_client):
    with ws_client.websocket_connect("/ws") as ws:
        ws.receive_json()
        ws.send_text("not json")
        assert ws.receive_json()["type"] == "error"
        ws.send_json({"type": "message", "content": "   "})
        assert ws.receive_json()["message"] == "Message is empty"
        ws.send_json({"type": "teleport"})
        assert "Unknown message type" in ws.receive_json()["message"]


def test_cors_origins_from_comma_separated_env():
    settings = AgentSettings(_env_file=None, cors_origins="http://a.test, http://b.test")
    assert settings.cors_origins == ["http://a.test", "http://b.test"]
    assert create_base_app(title="t", description="d", settings=settings)

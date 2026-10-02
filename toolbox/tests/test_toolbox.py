import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.registry import registry


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def call(client, name, **arguments):
    response = client.post("/tools/call", json={"name": name, "arguments": arguments})
    assert response.status_code == 200, response.text
    body = response.json()
    return body["content"][0]["text"], body["isError"]


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["tools"] == len(registry)


def test_list_tools_exposes_json_schema(client):
    tools = {t["name"]: t for t in client.get("/tools").json()["tools"]}
    assert {"add", "divide", "calculate", "uppercase", "current_time"} <= set(tools)
    schema = tools["add"]["inputSchema"]
    assert schema["type"] == "object"
    assert schema["required"] == ["a", "b"]
    assert schema["properties"]["a"]["type"] == "number"
    assert schema["properties"]["a"]["description"]


@pytest.mark.parametrize(
    ("name", "args", "expected"),
    [
        ("add", {"a": 5, "b": 3}, "8"),
        ("subtract", {"a": 10, "b": 4.5}, "5.5"),
        ("multiply", {"a": 25, "b": 8}, "200"),
        ("divide", {"a": 1, "b": 4}, "0.25"),
        ("power", {"base": 2, "exponent": 10}, "1024"),
        ("calculate", {"expression": "(25 * 8) + 3"}, "203"),
        ("calculate", {"expression": "2 ^ 3"}, "8"),
        ("calculate", {"expression": "0.1 + 0.2"}, "0.3"),
        ("uppercase", {"text": "hello"}, "HELLO"),
        ("lowercase", {"text": "HeLLo"}, "hello"),
        ("reverse_text", {"text": "abc"}, "cba"),
        ("count_words", {"text": "  the sky   is blue "}, "4"),
        ("count_characters", {"text": "a b c", "include_spaces": False}, "3"),
    ],
)
def test_tools(client, name, args, expected):
    assert call(client, name, **args) == (expected, False)


def test_numeric_strings_are_coerced(client):
    assert call(client, "add", a="2", b="3") == ("5", False)


@pytest.mark.parametrize(
    ("name", "args", "fragment"),
    [
        ("divide", {"a": 1, "b": 0}, "divide by zero"),
        ("add", {"a": 1}, "b: Field required"),
        ("add", {"a": 1, "b": 2, "c": 3}, "Extra inputs"),
        ("calculate", {"expression": "__import__('os')"}, "Unsupported syntax"),
        ("calculate", {"expression": "9 ** 99999"}, "Exponent too large"),
        ("current_time", {"timezone": "Mars/Olympus"}, "Unknown time zone"),
    ],
)
def test_errors_are_reported_to_the_model(client, name, args, fragment):
    text, is_error = call(client, name, **args)
    assert is_error
    assert fragment in text


def test_unknown_tool_is_404(client):
    response = client.post("/tools/call", json={"name": "nope", "arguments": {}})
    assert response.status_code == 404
    assert "Available" in response.json()["detail"]


def test_native_mcp_endpoint_lists_tools(client):
    headers = {"Accept": "application/json, text/event-stream"}
    init = client.post(
        "/mcp",
        headers=headers,
        json={
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "pytest", "version": "1"},
            },
        },
    )
    assert init.status_code == 200, init.text
    listed = client.post(
        "/mcp",
        headers=headers,
        json={"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
    )
    assert listed.status_code == 200, listed.text
    names = {t["name"] for t in listed.json()["result"]["tools"]}
    assert names == {t.name for t in registry.all()}

    called = client.post(
        "/mcp",
        headers=headers,
        json={
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {"name": "multiply", "arguments": {"a": 6, "b": 7}},
        },
    )
    assert called.status_code == 200, called.text
    assert called.json()["result"]["content"][0]["text"] == "42"

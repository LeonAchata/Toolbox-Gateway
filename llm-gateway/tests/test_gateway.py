import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.errors import ProviderError
from app.main import create_app
from app.providers.base import Price, Provider, ProviderResult
from app.registry import ProviderRegistry
from app.schemas import ToolCall, Usage


class FakeProvider(Provider):
    name = "fake"
    vendor = "Fake"
    aliases = ("fake-alias",)
    prices = {"fake-1": Price(1.0, 2.0)}
    calls: list = []
    fail_with: Exception | None = None

    @property
    def model_id(self):
        return "fake-1"

    def unavailable_reason(self):
        return None

    async def _generate(self, messages, tools, temperature, max_tokens):
        FakeProvider.calls.append((messages, tools))
        if FakeProvider.fail_with:
            raise FakeProvider.fail_with
        if tools and messages[-1].role == "user":
            return ProviderResult(
                content="",
                tool_calls=[ToolCall(id="c1", name=tools[0].name, arguments={"a": 1})],
                finish_reason="tool_calls",
                usage=Usage(input_tokens=1000, output_tokens=500, total_tokens=1500),
                model_id="fake-1",
            )
        return ProviderResult(
            content=f"echo: {messages[-1].content}",
            finish_reason="stop",
            usage=Usage(input_tokens=10, output_tokens=5, total_tokens=15),
            model_id="fake-1",
        )


class OfflineProvider(FakeProvider):
    name = "offline"
    aliases = ()

    def unavailable_reason(self):
        return "OFFLINE_KEY is not set"


@pytest.fixture
def client():
    FakeProvider.calls = []
    FakeProvider.fail_with = None
    settings = Settings(_env_file=None, default_model="fake")
    registry = ProviderRegistry(settings, provider_classes=(FakeProvider, OfflineProvider))
    with TestClient(create_app(settings, registry)) as c:
        yield c


def generate(client, **body):
    body.setdefault("messages", [{"role": "user", "content": "hi"}])
    return client.post("/generate", json=body)


def test_models_report_availability(client):
    models = {m["name"]: m for m in client.get("/models").json()["models"]}
    assert models["fake"]["available"] and models["fake"]["default"]
    assert not models["offline"]["available"]
    assert models["offline"]["reason"] == "OFFLINE_KEY is not set"


def test_generate_uses_default_model_and_estimates_cost(client):
    body = generate(client).json()
    assert body["model"] == "fake"
    assert body["provider_model"] == "fake-1"
    assert body["content"] == "echo: hi"
    assert body["finish_reason"] == "stop"
    assert body["usage"]["total_tokens"] == 15
    assert body["estimated_cost_usd"] == pytest.approx((10 * 1 + 5 * 2) / 1e6)
    assert body["cached"] is False


def test_aliases_resolve(client):
    assert generate(client, model="FAKE-ALIAS").json()["model"] == "fake"


def test_tool_calls_round_trip(client):
    tools = [{"name": "add", "description": "Add", "input_schema": {"type": "object"}}]
    first = generate(client, tools=tools).json()
    assert first["finish_reason"] == "tool_calls"
    assert first["tool_calls"] == [
        {"id": "c1", "name": "add", "arguments": {"a": 1}, "signature": None}
    ]

    messages = [
        {"role": "user", "content": "add"},
        {"role": "assistant", "content": "", "tool_calls": first["tool_calls"]},
        {"role": "tool", "tool_call_id": "c1", "name": "add", "content": "1"},
    ]
    second = generate(client, tools=tools, messages=messages).json()
    assert second["content"] == "echo: 1"
    sent_messages, _ = FakeProvider.calls[-1]
    assert sent_messages[1].tool_calls[0].name == "add"


def test_cache_hit_skips_provider_and_costs_nothing(client):
    generate(client)
    second = generate(client).json()
    assert second["cached"] is True
    assert second["estimated_cost_usd"] == 0.0
    assert len(FakeProvider.calls) == 1

    generate(client, use_cache=False)
    assert len(FakeProvider.calls) == 2

    assert client.post("/cache/clear").json() == {"cleared": 1}
    generate(client)
    assert len(FakeProvider.calls) == 3


def test_metrics(client):
    generate(client)
    generate(client)
    metrics = client.get("/metrics").json()
    assert metrics["total"]["requests"] == 2
    assert metrics["total"]["cache_hits"] == 1
    assert metrics["by_model"]["fake"]["total_tokens"] == 15
    assert metrics["cache"]["hit_rate"] == 0.5
    client.post("/metrics/reset")
    assert client.get("/metrics").json()["total"]["requests"] == 0


def test_unknown_model_is_400(client):
    response = generate(client, model="gpt-17")
    assert response.status_code == 400
    assert "Unknown model" in response.json()["error"]["message"]


def test_unconfigured_model_is_503(client):
    response = generate(client, model="offline")
    assert response.status_code == 503
    assert "OFFLINE_KEY" in response.json()["error"]["message"]


def test_provider_errors_keep_their_status(client):
    FakeProvider.fail_with = ProviderError("slow down", 429)
    response = generate(client)
    assert response.status_code == 429
    assert response.json()["error"]["message"] == "slow down"
    assert client.get("/metrics").json()["total"]["errors"] == 1


def test_unexpected_errors_become_502(client):
    FakeProvider.fail_with = RuntimeError("boom")
    response = generate(client)
    assert response.status_code == 502
    assert "boom" in response.json()["error"]["message"]


def test_rejects_invalid_messages(client):
    assert generate(client, messages=[]).status_code == 422
    bad_tool = [{"role": "tool", "content": "x"}]
    assert generate(client, messages=bad_tool).status_code == 422


def test_api_key_is_enforced_when_configured():
    settings = Settings(_env_file=None, default_model="fake", gateway_api_key="s3cret")
    registry = ProviderRegistry(settings, provider_classes=(FakeProvider,))
    with TestClient(create_app(settings, registry)) as c:
        assert c.get("/health").status_code == 200
        assert c.get("/models").status_code == 401
        ok = c.get("/models", headers={"Authorization": "Bearer s3cret"})
        assert ok.status_code == 200


def test_blank_env_values_are_treated_as_missing():
    settings = Settings(_env_file=None, openai_api_key="", google_api_key="your_google_api_key")
    assert settings.openai_api_key is None
    assert settings.google_api_key is None

import httpx
import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agent_core.graph import recent_history
from agent_core.models import detect_model
from conftest import llm_reply


async def test_parallel_tool_calls_then_answer(services):
    services.replies = [
        llm_reply(
            tool_calls=[
                ("c1", "add", {"a": 2, "b": 3}, "c2ln"),
                ("c2", "uppercase", {"text": "hi"}),
            ]
        ),
        llm_reply(content="2 + 3 = 5 and HI"),
    ]
    runtime = services.runtime()
    result = await runtime.run("add 2 and 3, and uppercase hi")

    assert result["answer"] == "2 + 3 = 5 and HI"
    assert sorted(services.tool_calls) == [("add", {"a": 2, "b": 3}), ("uppercase", {"text": "hi"})]
    assert [e["type"] for e in result["trace"]] == ["llm", "tool", "tool", "llm", "answer"]
    assert result["usage"]["llm_calls"] == 2
    assert result["usage"]["tool_calls"] == 2
    assert result["usage"]["input_tokens"] == 200

    # Second request carries the assistant tool calls (with the provider
    # signature echoed back) followed by one result per call.
    second = services.gateway_requests[1]
    roles = [m["role"] for m in second["messages"]]
    assert roles == ["system", "user", "assistant", "tool", "tool"]
    assistant = second["messages"][2]
    assert assistant["tool_calls"][0]["signature"] == "c2ln"
    results = {m["tool_call_id"]: m["content"] for m in second["messages"][3:]}
    assert results == {"c1": "5", "c2": "HI"}
    assert {t["name"] for t in second["tools"]} == {"add", "uppercase"}


async def test_conversation_memory_per_session(services):
    services.replies = [llm_reply(content="Nice to meet you, Ana.")]
    runtime = services.runtime()
    first = await runtime.run("My name is Ana")
    await runtime.run("What is my name?", session_id=first["session_id"])

    contents = [m["content"] for m in services.gateway_requests[-1]["messages"]]
    assert contents[1:] == ["My name is Ana", "Nice to meet you, Ana.", "What is my name?"]

    await runtime.run("What is my name?")  # new session
    assert len(services.gateway_requests[-1]["messages"]) == 2


async def test_tool_budget_stops_the_loop_and_cleans_memory(services):
    services.replies = [llm_reply(tool_calls=[("c", "add", {"a": 1, "b": 1})])]
    runtime = services.runtime(max_tool_rounds=3)
    result = await runtime.run("loop forever")

    assert len(services.gateway_requests) == 3
    assert len(services.tool_calls) == 2
    assert "stopped after 3 rounds" in result["answer"]
    assert (
        "maximum number of tool rounds" in services.gateway_requests[-1]["messages"][0]["content"]
    )

    # The unanswered tool call must not leak into the next turn.
    services.replies = [llm_reply(content="ok")]
    await runtime.run("next", session_id=result["session_id"])
    history = services.gateway_requests[-1]["messages"]
    pending = {c["id"] for m in history if m["role"] == "assistant" for c in m["tool_calls"]}
    answered = {m["tool_call_id"] for m in history if m["role"] == "tool"}
    assert pending <= answered


async def test_unknown_tool_is_reported_back_to_the_model(services):
    services.replies = [
        llm_reply(tool_calls=[("c1", "teleport", {})]),
        llm_reply(content="I cannot teleport."),
    ]
    result = await services.runtime().run("teleport me")
    tool_event = result["trace"][1]
    assert tool_event["is_error"] is True
    assert "Unknown tool" in tool_event["result"]
    assert result["answer"] == "I cannot teleport."


async def test_model_selection(services):
    services.replies = [llm_reply(content="hi")]
    runtime = services.runtime(default_model="bedrock")
    await runtime.run("hello")
    assert services.gateway_requests[-1]["model"] == "bedrock"
    await runtime.run("use gemini, say hi")
    assert services.gateway_requests[-1]["model"] == "gemini"
    await runtime.run("hello", model="openai")
    assert services.gateway_requests[-1]["model"] == "openai"


async def test_empty_answers_get_an_explanation(services):
    services.replies = [llm_reply(content="", finish_reason="length")]
    result = await services.runtime().run("write a novel")
    assert "ran out of output tokens" in result["answer"]


async def test_streaming_emits_events_in_order(services):
    services.replies = [
        llm_reply(tool_calls=[("c1", "add", {"a": 1, "b": 2})]),
        llm_reply(content="3"),
    ]
    events = [e async for e in services.runtime().stream("1+2")]
    assert [e["type"] for e in events] == [
        "llm_start",
        "llm",
        "tool",
        "llm_start",
        "llm",
        "answer",
    ]


async def test_gateway_errors_propagate(services):
    from agent_core import GatewayError

    services.replies = [
        httpx.Response(429, json={"error": {"type": "ProviderError", "message": "Rate limited"}})
    ]
    with pytest.raises(GatewayError) as excinfo:
        await services.runtime().run("hi")
    assert excinfo.value.status_code == 429
    assert excinfo.value.message == "Rate limited"


def test_recent_history_cuts_at_user_turns():
    messages = [
        HumanMessage("q1"),
        AIMessage("", tool_calls=[{"id": "a", "name": "add", "args": {}}]),
        ToolMessage("1", tool_call_id="a"),
        AIMessage("a1"),
        HumanMessage("q2"),
        AIMessage("a2"),
        HumanMessage("q3"),
    ]
    assert recent_history(messages, 2) == messages[4:]
    assert recent_history(messages, 5) == messages


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("use gemini, add 2 and 3", "gemini"),
        ("Usa GPT-4o para esto", "openai"),
        ("con bedrock multiplica 3 por 9", "bedrock"),
        ("with nova please", "bedrock"),
        ("use gemini or use gpt", "gemini"),
        ("what is a gemini?", None),
        ("add 2 and 3", None),
    ],
)
def test_detect_model(text, expected):
    assert detect_model(text) == expected

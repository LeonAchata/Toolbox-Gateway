"""Provider adapters, tested against stubbed SDK clients."""

import base64
import json
from types import SimpleNamespace

import pytest
from google.genai import types

from app.config import Settings
from app.errors import ProviderError
from app.providers.bedrock import BedrockProvider, to_converse_messages
from app.providers.gemini import GeminiProvider, to_gemini_contents
from app.providers.openai import OpenAIProvider, to_openai_messages
from app.schemas import Message, ToolCall, ToolSpec

SETTINGS = Settings(
    _env_file=None,
    openai_api_key="sk-test",
    google_api_key="g-test",
    aws_access_key_id="AKIA",
    aws_secret_access_key="secret",
)
TOOLS = [
    ToolSpec(
        name="add",
        description="Add two numbers",
        input_schema={
            "type": "object",
            "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
            "required": ["a", "b"],
        },
    )
]
CONVERSATION = [
    Message(role="system", content="Be brief."),
    Message(role="user", content="What is 2+3 and 4+5?"),
    Message(
        role="assistant",
        content="",
        tool_calls=[
            ToolCall(id="t1", name="add", arguments={"a": 2, "b": 3}),
            ToolCall(id="t2", name="add", arguments={"a": 4, "b": 5}),
        ],
    ),
    Message(role="tool", tool_call_id="t1", name="add", content="5"),
    Message(role="tool", tool_call_id="t2", name="add", content="9"),
]


# OpenAI ---------------------------------------------------------------------


def test_openai_message_conversion():
    converted = to_openai_messages(CONVERSATION)
    assert converted[0] == {"role": "system", "content": "Be brief."}
    assert converted[2]["content"] is None
    assert converted[2]["tool_calls"][1]["function"] == {
        "name": "add",
        "arguments": json.dumps({"a": 4, "b": 5}),
    }
    assert converted[4] == {"role": "tool", "tool_call_id": "t2", "content": "9"}


async def test_openai_generate_parses_tool_calls():
    provider = OpenAIProvider(SETTINGS)
    captured = {}

    async def create(**params):
        captured.update(params)
        call = SimpleNamespace(
            id="call_x",
            type="function",
            function=SimpleNamespace(name="add", arguments='{"a": 1, "b": 2}'),
        )
        return SimpleNamespace(
            model="gpt-4o-mini-2024-07-18",
            choices=[
                SimpleNamespace(
                    finish_reason="tool_calls",
                    message=SimpleNamespace(content=None, refusal=None, tool_calls=[call]),
                )
            ],
            usage=SimpleNamespace(prompt_tokens=50, completion_tokens=10, total_tokens=60),
        )

    provider._client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    result = await provider.generate(CONVERSATION[:2], TOOLS, 0.2, 256)

    assert captured["tools"][0]["function"]["parameters"] == TOOLS[0].input_schema
    assert captured["max_completion_tokens"] == 256
    assert captured["temperature"] == 0.2
    assert result.finish_reason == "tool_calls"
    assert result.tool_calls == [ToolCall(id="call_x", name="add", arguments={"a": 1, "b": 2})]
    assert result.usage.total_tokens == 60
    # Longest matching price prefix wins: gpt-4o-mini, not gpt-4o.
    assert provider.estimate_cost(result.usage, result.model_id) == pytest.approx(
        (50 * 0.15 + 10 * 0.60) / 1e6
    )


# Bedrock --------------------------------------------------------------------


def test_bedrock_merges_tool_results_into_one_user_turn():
    converted = to_converse_messages(CONVERSATION)
    assert [m["role"] for m in converted] == ["user", "assistant", "user"]
    assert converted[1]["content"][0]["toolUse"] == {
        "toolUseId": "t1",
        "name": "add",
        "input": {"a": 2, "b": 3},
    }
    results = [block["toolResult"] for block in converted[2]["content"]]
    assert [r["toolUseId"] for r in results] == ["t1", "t2"]
    assert results[1]["content"] == [{"text": "9"}]


async def test_bedrock_generate():
    provider = BedrockProvider(SETTINGS)
    captured = {}

    def converse(**params):
        captured.update(params)
        return {
            "output": {
                "message": {
                    "role": "assistant",
                    "content": [
                        {"text": "Let me add those."},
                        {"toolUse": {"toolUseId": "tu1", "name": "add", "input": {"a": 1, "b": 1}}},
                    ],
                }
            },
            "stopReason": "tool_use",
            "usage": {"inputTokens": 100, "outputTokens": 20, "totalTokens": 120},
        }

    provider._client = SimpleNamespace(converse=converse)
    result = await provider.generate(CONVERSATION[:2], TOOLS, 1.5, 128)

    assert captured["system"] == [{"text": "Be brief."}]
    assert captured["inferenceConfig"] == {"maxTokens": 128, "temperature": 1.0}
    assert captured["toolConfig"]["tools"][0]["toolSpec"]["inputSchema"]["json"]["required"] == [
        "a",
        "b",
    ]
    assert result.content == "Let me add those."
    assert result.finish_reason == "tool_calls"
    assert result.tool_calls[0].id == "tu1"
    assert provider.estimate_cost(result.usage) == pytest.approx((100 * 0.8 + 20 * 3.2) / 1e6)


async def test_bedrock_client_errors_map_to_status():
    from botocore.exceptions import ClientError

    provider = BedrockProvider(SETTINGS)

    def converse(**_):
        raise ClientError(
            {"Error": {"Code": "ThrottlingException", "Message": "Too many requests"}},
            "Converse",
        )

    provider._client = SimpleNamespace(converse=converse)
    with pytest.raises(ProviderError) as excinfo:
        await provider.generate(CONVERSATION[:2], [], 0.5, 64)
    assert excinfo.value.status_code == 429


# Gemini ---------------------------------------------------------------------


def test_gemini_contents_echo_signatures_and_name_function_responses():
    signature = base64.b64encode(b"sig-bytes").decode()
    conversation = [
        *CONVERSATION[:2],
        Message(
            role="assistant",
            tool_calls=[ToolCall(id="g1", name="add", arguments={"a": 1}, signature=signature)],
        ),
        Message(role="tool", tool_call_id="g1", content="1"),
    ]
    contents = to_gemini_contents(conversation)
    assert [c.role for c in contents] == ["user", "model", "user"]
    call_part = contents[1].parts[0]
    assert call_part.function_call.name == "add"
    assert call_part.thought_signature == b"sig-bytes"
    response_part = contents[2].parts[0]
    assert response_part.function_response.name == "add"
    assert response_part.function_response.response == {"result": "1"}


async def test_gemini_generate():
    provider = GeminiProvider(SETTINGS)
    captured = {}

    async def generate_content(model, contents, config):
        captured.update(model=model, contents=contents, config=config)
        return types.GenerateContentResponse(
            candidates=[
                types.Candidate(
                    finish_reason=types.FinishReason.STOP,
                    content=types.Content(
                        role="model",
                        parts=[
                            types.Part(text="thinking...", thought=True),
                            types.Part(
                                function_call=types.FunctionCall(name="add", args={"a": 2, "b": 2}),
                                thought_signature=b"abc",
                            ),
                        ],
                    ),
                )
            ],
            usage_metadata=types.GenerateContentResponseUsageMetadata(
                prompt_token_count=40,
                candidates_token_count=8,
                thoughts_token_count=12,
                total_token_count=60,
            ),
            model_version="gemini-2.5-flash",
        )

    provider._client = SimpleNamespace(
        aio=SimpleNamespace(models=SimpleNamespace(generate_content=generate_content))
    )
    result = await provider.generate(CONVERSATION[:2], TOOLS, 0.3, 512)

    config = captured["config"]
    assert config.system_instruction == "Be brief."
    assert config.tools[0].function_declarations[0].parameters_json_schema == TOOLS[0].input_schema
    assert result.content == ""
    assert result.finish_reason == "tool_calls"
    assert result.tool_calls[0].arguments == {"a": 2, "b": 2}
    assert base64.b64decode(result.tool_calls[0].signature) == b"abc"
    assert result.usage.output_tokens == 20
    assert result.usage.total_tokens == 60

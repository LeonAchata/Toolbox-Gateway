"""The agent loop as a LangGraph state machine.

    START -> agent --(tool calls?)--> tools -> agent ... -> finalize -> END

Each node reports what it did twice: as an entry in ``state["trace"]`` (the
HTTP agent returns this) and as a custom stream event (the WebSocket agent
forwards these to the browser as they happen).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from .config import AgentSettings
from .gateway import GatewayClient
from .toolbox import ToolboxClient

logger = logging.getLogger(__name__)

_LIMIT_NOTE = (
    "You have used the maximum number of tool rounds for this request. "
    "Do not call any more tools; answer with what you already know."
)


class AgentState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    # Per-turn fields: the caller resets them on every new user message.
    model: str | None
    rounds: int
    trace: list[dict[str, Any]]
    answer: str | None


def recent_history(messages: list[AnyMessage], turns: int) -> list[AnyMessage]:
    """Keep the last ``turns`` user turns.

    Cutting at a user message guarantees we never send a tool result whose
    originating tool call was trimmed away, which every provider rejects.
    """
    human_indexes = [i for i, m in enumerate(messages) if isinstance(m, HumanMessage)]
    if len(human_indexes) <= turns:
        return messages
    return messages[human_indexes[-turns] :]


def to_gateway_messages(system_prompt: str, messages: list[AnyMessage]) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
    for message in messages:
        if isinstance(message, HumanMessage):
            converted.append({"role": "user", "content": message.text})
        elif isinstance(message, AIMessage):
            signatures = message.additional_kwargs.get("signatures", {})
            converted.append(
                {
                    "role": "assistant",
                    "content": message.text,
                    "tool_calls": [
                        {
                            "id": call["id"],
                            "name": call["name"],
                            "arguments": call["args"],
                            "signature": signatures.get(call["id"]),
                        }
                        for call in message.tool_calls
                    ],
                }
            )
        elif isinstance(message, ToolMessage):
            converted.append(
                {
                    "role": "tool",
                    "tool_call_id": message.tool_call_id,
                    "name": message.name,
                    "content": message.text,
                }
            )
    return converted


def build_agent(
    settings: AgentSettings,
    gateway: GatewayClient,
    toolbox: ToolboxClient,
    checkpointer: InMemorySaver | None = None,
):
    async def agent(state: AgentState) -> dict[str, Any]:
        emit = get_stream_writer()
        rounds = state.get("rounds", 0) + 1
        model = state.get("model")
        system_prompt = settings.system_prompt
        if rounds >= settings.max_tool_rounds:
            system_prompt = f"{system_prompt}\n\n{_LIMIT_NOTE}"
        # Tools stay declared even on the last round: Bedrock rejects a
        # conversation that contains tool calls but no tool configuration.
        tools = await toolbox.specs()
        history = recent_history(state["messages"], settings.history_turns)

        emit({"type": "llm_start", "round": rounds, "model": model or "default"})
        response = await gateway.generate(
            messages=to_gateway_messages(system_prompt, history),
            tools=tools,
            model=model,
            temperature=settings.temperature,
            max_tokens=settings.max_tokens,
        )

        tool_calls = response.get("tool_calls") or []
        event = {
            "type": "llm",
            "round": rounds,
            "model": response["model"],
            "provider_model": response["provider_model"],
            "finish_reason": response["finish_reason"],
            "cached": response["cached"],
            "latency_ms": response["latency_ms"],
            "usage": response["usage"],
            "cost_usd": response["estimated_cost_usd"],
            "tool_calls": [
                {"id": c["id"], "name": c["name"], "arguments": c["arguments"]} for c in tool_calls
            ],
        }
        emit(event)
        logger.info(
            "round %d: %s answered with %d tool call(s) in %.0f ms",
            rounds,
            response["model"],
            len(tool_calls),
            response["latency_ms"],
        )

        message = AIMessage(
            content=response["content"],
            tool_calls=[
                {"id": c["id"], "name": c["name"], "args": c["arguments"]} for c in tool_calls
            ],
            additional_kwargs={
                "signatures": {c["id"]: c["signature"] for c in tool_calls if c.get("signature")}
            },
            response_metadata={
                "model": response["model"],
                "provider_model": response["provider_model"],
                "finish_reason": response["finish_reason"],
            },
        )
        return {
            "messages": [message],
            "rounds": rounds,
            "trace": [*state.get("trace", []), event],
        }

    async def tools(state: AgentState) -> dict[str, Any]:
        emit = get_stream_writer()
        calls = state["messages"][-1].tool_calls
        # Independent calls run concurrently.
        results = await asyncio.gather(
            *(toolbox.call(call["name"], call["args"]) for call in calls)
        )

        messages, events = [], []
        for call, result in zip(calls, results, strict=True):
            messages.append(
                ToolMessage(
                    content=result.text,
                    tool_call_id=call["id"],
                    name=call["name"],
                    status="error" if result.is_error else "success",
                )
            )
            event = {
                "type": "tool",
                "id": call["id"],
                "name": call["name"],
                "arguments": call["args"],
                "result": result.text,
                "is_error": result.is_error,
                "duration_ms": result.duration_ms,
            }
            events.append(event)
            emit(event)
        return {"messages": messages, "trace": [*state.get("trace", []), *events]}

    def finalize(state: AgentState) -> dict[str, Any]:
        emit = get_stream_writer()
        last = state["messages"][-1]
        update: dict[str, Any] = {}
        answer = last.text.strip() if isinstance(last, AIMessage) else ""
        if isinstance(last, AIMessage) and last.tool_calls:
            # Tool budget exhausted. Drop the unanswered calls from memory,
            # otherwise every provider rejects the next turn.
            update["messages"] = [last.model_copy(update={"tool_calls": []})]
            answer = answer or (
                f"I stopped after {settings.max_tool_rounds} rounds of tool calls "
                "without reaching a final answer."
            )
        if not answer:
            reason = last.response_metadata.get("finish_reason")
            answer = {
                "length": "The model ran out of output tokens before finishing its answer.",
                "content_filter": "The provider's safety filter blocked this answer.",
            }.get(reason, "The model returned an empty answer.")
        event = {"type": "answer", "content": answer}
        emit(event)
        return {**update, "answer": answer, "trace": [*state.get("trace", []), event]}

    def route(state: AgentState) -> str:
        last = state["messages"][-1]
        if (
            isinstance(last, AIMessage)
            and last.tool_calls
            and state.get("rounds", 0) < settings.max_tool_rounds
        ):
            return "tools"
        return "finalize"

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent)
    graph.add_node("tools", tools)
    graph.add_node("finalize", finalize)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", route, {"tools": "tools", "finalize": "finalize"})
    graph.add_edge("tools", "agent")
    graph.add_edge("finalize", END)
    return graph.compile(checkpointer=checkpointer or InMemorySaver())

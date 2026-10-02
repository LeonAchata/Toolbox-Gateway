"""Tool registry.

A tool is a plain Python function with type-annotated parameters. The registry
derives a Pydantic model from the signature, which gives us argument
validation and a JSON Schema for free. The same function is also registered
with the MCP server, so both transports always describe the tool identically.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ConfigDict, ValidationError, create_model


class ToolNotFoundError(LookupError):
    """Raised when a tool name is not registered."""


class ToolArgumentError(ValueError):
    """Raised when the arguments do not match the tool's schema."""


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    category: str
    fn: Callable[..., Any]
    args_model: type[BaseModel] = field(repr=False)

    @property
    def input_schema(self) -> dict[str, Any]:
        schema = self.args_model.model_json_schema()
        schema.pop("title", None)
        for prop in schema.get("properties", {}).values():
            prop.pop("title", None)
        schema.setdefault("required", [])
        return schema

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "inputSchema": self.input_schema,
        }

    async def run(self, arguments: dict[str, Any]) -> Any:
        try:
            parsed = self.args_model.model_validate(arguments or {})
        except ValidationError as exc:
            problems = "; ".join(
                f"{'.'.join(str(p) for p in err['loc']) or 'arguments'}: {err['msg']}"
                for err in exc.errors()
            )
            raise ToolArgumentError(problems) from exc

        result = self.fn(**parsed.model_dump())
        if inspect.isawaitable(result):
            result = await result
        return result


def _args_model_for(name: str, fn: Callable[..., Any]) -> type[BaseModel]:
    fields: dict[str, Any] = {}
    for param in inspect.signature(fn, eval_str=True).parameters.values():
        if param.annotation is inspect.Parameter.empty:
            raise TypeError(f"Tool '{name}': parameter '{param.name}' needs a type annotation")
        default = ... if param.default is inspect.Parameter.empty else param.default
        fields[param.name] = (param.annotation, default)
    model_name = "".join(part.capitalize() for part in name.split("_")) + "Args"
    return create_model(model_name, __config__=ConfigDict(extra="forbid"), **fields)


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, category: str, name: str | None = None):
        """Decorator that registers a function as a tool.

        The first paragraph of the docstring becomes the tool description.
        """

        def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
            tool_name = name or fn.__name__
            if tool_name in self._tools:
                raise ValueError(f"Tool '{tool_name}' is already registered")
            doc = inspect.getdoc(fn) or ""
            description = doc.split("\n\n", 1)[0].replace("\n", " ").strip()
            if not description:
                raise ValueError(f"Tool '{tool_name}' needs a docstring")
            self._tools[tool_name] = Tool(
                name=tool_name,
                description=description,
                category=category,
                fn=fn,
                args_model=_args_model_for(tool_name, fn),
            )
            return fn

        return decorator

    def get(self, name: str) -> Tool:
        try:
            return self._tools[name]
        except KeyError:
            available = ", ".join(sorted(self._tools))
            raise ToolNotFoundError(f"Unknown tool '{name}'. Available: {available}") from None

    def all(self) -> list[Tool]:
        return list(self._tools.values())

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: object) -> bool:
        return name in self._tools


registry = ToolRegistry()

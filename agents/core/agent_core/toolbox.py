"""Client for the toolbox service."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any

import httpx

logger = logging.getLogger(__name__)


@dataclass
class ToolResult:
    text: str
    is_error: bool
    duration_ms: float


class ToolboxUnavailableError(RuntimeError):
    pass


class ToolboxClient:
    def __init__(
        self,
        base_url: str,
        *,
        timeout: float = 30.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self._http = httpx.AsyncClient(base_url=self.base_url, timeout=timeout, transport=transport)
        self._tools: list[dict[str, Any]] = []
        self._lock = asyncio.Lock()

    async def close(self) -> None:
        await self._http.aclose()

    @property
    def loaded(self) -> bool:
        return bool(self._tools)

    @property
    def tool_count(self) -> int:
        return len(self._tools)

    async def connect(self, attempts: int = 10, base_delay: float = 0.5) -> bool:
        """Load the tool catalog, retrying with backoff while the toolbox boots.

        Returns False instead of raising so the agent can still start and
        report a degraded health status; tools are loaded lazily later.
        """
        for attempt in range(1, attempts + 1):
            try:
                await self.refresh()
                logger.info("Loaded %d tools from %s", len(self._tools), self.base_url)
                return True
            except (httpx.HTTPError, ToolboxUnavailableError) as exc:
                delay = min(base_delay * 2 ** (attempt - 1), 8.0)
                logger.warning(
                    "Toolbox not reachable (attempt %d/%d): %s. Retrying in %.1fs",
                    attempt,
                    attempts,
                    exc,
                    delay,
                )
                if attempt < attempts:
                    await asyncio.sleep(delay)
        logger.error("Giving up on the toolbox for now; will retry on first use")
        return False

    async def refresh(self) -> list[dict[str, Any]]:
        async with self._lock:
            response = await self._http.get("/tools")
            if response.status_code != 200:
                raise ToolboxUnavailableError(f"GET /tools returned {response.status_code}")
            self._tools = response.json().get("tools", [])
            return self._tools

    async def tools(self) -> list[dict[str, Any]]:
        if not self._tools:
            try:
                await self.refresh()
            except httpx.HTTPError as exc:
                raise ToolboxUnavailableError(f"Toolbox unreachable: {exc}") from exc
        return self._tools

    async def specs(self) -> list[dict[str, Any]]:
        """Tool definitions in the gateway's ToolSpec format."""
        return [
            {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "input_schema": tool.get("inputSchema") or {"type": "object", "properties": {}},
            }
            for tool in await self.tools()
        ]

    async def call(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        started = time.perf_counter()
        try:
            response = await self._http.post(
                "/tools/call", json={"name": name, "arguments": arguments}
            )
        except httpx.HTTPError as exc:
            return ToolResult(f"Toolbox unreachable: {exc}", True, _elapsed(started))

        if response.status_code == 404:
            detail = response.json().get("detail", f"Unknown tool '{name}'")
            return ToolResult(detail, True, _elapsed(started))
        if response.status_code != 200:
            return ToolResult(
                f"Toolbox returned HTTP {response.status_code}", True, _elapsed(started)
            )

        body = response.json()
        text = "\n".join(
            item.get("text", "") for item in body.get("content", []) if item.get("type") == "text"
        )
        return ToolResult(text, bool(body.get("isError")), _elapsed(started))

    async def ping(self) -> bool:
        try:
            response = await self._http.get("/health", timeout=3.0)
            return response.status_code == 200
        except httpx.HTTPError:
            return False


def _elapsed(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 2)

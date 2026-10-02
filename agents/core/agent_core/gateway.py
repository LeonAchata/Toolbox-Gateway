"""Client for the LLM gateway."""

from __future__ import annotations

from typing import Any

import httpx


class GatewayError(RuntimeError):
    def __init__(self, message: str, status_code: int = 502):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


class GatewayClient:
    def __init__(
        self,
        base_url: str,
        *,
        api_key: str | None = None,
        timeout: float = 180.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._http = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            headers=headers,
            timeout=httpx.Timeout(timeout, connect=5.0),
            transport=transport,
        )

    async def close(self) -> None:
        await self._http.aclose()

    async def _request(self, method: str, path: str, **kwargs) -> Any:
        try:
            response = await self._http.request(method, path, **kwargs)
        except httpx.TimeoutException as exc:
            raise GatewayError("LLM gateway timed out", 504) from exc
        except httpx.HTTPError as exc:
            raise GatewayError(f"LLM gateway unreachable: {exc}", 503) from exc

        if response.status_code >= 400:
            try:
                body = response.json()
                message = (
                    body.get("error", {}).get("message") or body.get("detail") or response.text
                )
            except ValueError:
                message = response.text
            raise GatewayError(str(message), response.status_code)
        return response.json()

    async def generate(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        model: str | None,
        temperature: float,
        max_tokens: int,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "messages": messages,
            "tools": tools,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if model:
            payload["model"] = model
        return await self._request("POST", "/generate", json=payload)

    async def models(self) -> list[dict[str, Any]]:
        return (await self._request("GET", "/models"))["models"]

    async def metrics(self) -> dict[str, Any]:
        return await self._request("GET", "/metrics")

    async def ping(self) -> bool:
        try:
            response = await self._http.get("/health", timeout=3.0)
            return response.status_code == 200
        except httpx.HTTPError:
            return False

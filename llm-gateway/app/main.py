"""LLM gateway.

One HTTP API in front of several model providers. Clients send a provider
neutral conversation (including tool definitions and tool results) and get a
normalized response back, with usage, latency and an estimated cost.
"""

from __future__ import annotations

import logging
import secrets
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from langsmith import traceable

from .cache import ResponseCache
from .config import Settings
from .config import settings as default_settings
from .errors import GatewayError
from .metrics import Metrics
from .registry import ProviderRegistry
from .schemas import GenerateRequest, GenerateResponse, ModelInfo

VERSION = "3.0.0"

logger = logging.getLogger("llm-gateway")
_bearer = HTTPBearer(auto_error=False)


def create_app(
    settings: Settings | None = None, registry: ProviderRegistry | None = None
) -> FastAPI:
    settings = settings or default_settings
    registry = registry or ProviderRegistry(settings)
    cache = ResponseCache(
        settings.cache_enabled, settings.cache_max_entries, settings.cache_ttl_seconds
    )
    metrics = Metrics()

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        for info in registry.describe():
            state = "ready" if info.available else f"unavailable ({info.reason})"
            logger.info("Model %-8s -> %s: %s", info.name, info.provider_model, state)
        logger.info("Default model: %s", registry.default_name() or "none configured")
        yield

    app = FastAPI(
        title="LLM Gateway",
        description="Provider-neutral chat completions with tool calling, caching and metrics.",
        version=VERSION,
        lifespan=lifespan,
    )
    app.state.registry = registry
    app.state.cache = cache
    app.state.metrics = metrics
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    async def require_api_key(
        credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
    ) -> None:
        expected = settings.gateway_api_key
        if not expected:
            return
        supplied = credentials.credentials if credentials else ""
        if not secrets.compare_digest(supplied.encode(), expected.encode()):
            raise GatewayError("Missing or invalid API key", 401)

    @app.exception_handler(GatewayError)
    async def gateway_error_handler(_: Request, exc: GatewayError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": {"type": type(exc).__name__, "message": exc.message}},
        )

    @app.get("/health")
    async def health() -> dict[str, Any]:
        models = registry.describe()
        return {
            "status": "ok",
            "service": "llm-gateway",
            "version": VERSION,
            "available_models": [m.name for m in models if m.available],
            "default_model": registry.default_name(),
        }

    @app.get("/models", dependencies=[Depends(require_api_key)])
    async def list_models() -> dict[str, list[ModelInfo]]:
        return {"models": registry.describe()}

    @traceable(run_type="chain", name="gateway.generate")
    async def _generate(request: GenerateRequest) -> GenerateResponse:
        started = time.perf_counter()
        provider = registry.resolve(request.model)
        messages = request.messages

        cache_key = ResponseCache.key(
            {
                "provider": provider.name,
                "model": provider.model_id,
                "messages": [m.model_dump() for m in messages],
                "tools": [t.model_dump() for t in request.tools],
                "temperature": request.temperature,
                "max_tokens": request.max_tokens,
            }
        )
        cached = cache.get(cache_key) if request.use_cache else None
        if cached is not None:
            response = GenerateResponse.model_validate(
                {
                    **cached,
                    "id": f"gen_{uuid.uuid4().hex[:20]}",
                    "cached": True,
                    "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                    # A cache hit costs nothing upstream.
                    "estimated_cost_usd": 0.0,
                }
            )
            metrics.record(provider.name, latency_ms=response.latency_ms, cached=True)
            return response

        try:
            result = await provider.generate(
                messages, request.tools, request.temperature, request.max_tokens
            )
        except GatewayError:
            metrics.record(
                provider.name,
                latency_ms=(time.perf_counter() - started) * 1000,
                error=True,
            )
            raise
        except Exception as exc:
            metrics.record(
                provider.name,
                latency_ms=(time.perf_counter() - started) * 1000,
                error=True,
            )
            logger.exception("Unexpected error from %s", provider.name)
            raise GatewayError(f"{provider.vendor} call failed: {exc}", 502) from exc

        latency_ms = round((time.perf_counter() - started) * 1000, 2)
        cost = provider.estimate_cost(result.usage, result.model_id)
        response = GenerateResponse(
            id=f"gen_{uuid.uuid4().hex[:20]}",
            model=provider.name,
            provider=provider.vendor,
            provider_model=result.model_id,
            content=result.content,
            tool_calls=result.tool_calls,
            finish_reason=result.finish_reason,
            usage=result.usage,
            cached=False,
            latency_ms=latency_ms,
            estimated_cost_usd=round(cost, 8),
        )
        metrics.record(
            provider.name,
            latency_ms=latency_ms,
            input_tokens=result.usage.input_tokens,
            output_tokens=result.usage.output_tokens,
            cost_usd=cost,
            tool_calls=bool(result.tool_calls),
        )
        # Truncated or filtered answers are not worth replaying.
        if request.use_cache and result.finish_reason in ("stop", "tool_calls"):
            cache.set(cache_key, response.model_dump(exclude={"id", "cached", "latency_ms"}))

        logger.info(
            "%s (%s) %s in=%d out=%d tools=%d cost=$%.6f %.0fms",
            provider.name,
            result.model_id,
            result.finish_reason,
            result.usage.input_tokens,
            result.usage.output_tokens,
            len(result.tool_calls),
            cost,
            latency_ms,
        )
        return response

    @app.post("/generate", response_model=GenerateResponse, dependencies=[Depends(require_api_key)])
    async def generate(request: GenerateRequest) -> GenerateResponse:
        return await _generate(request)

    @app.get("/metrics", dependencies=[Depends(require_api_key)])
    async def get_metrics() -> dict[str, Any]:
        return {**metrics.snapshot(), "cache": cache.stats()}

    @app.post("/metrics/reset", dependencies=[Depends(require_api_key)])
    async def reset_metrics() -> dict[str, str]:
        metrics.reset()
        return {"status": "reset"}

    @app.post("/cache/clear", dependencies=[Depends(require_api_key)])
    async def clear_cache() -> dict[str, int]:
        return {"cleared": cache.clear()}

    return app


logging.basicConfig(
    level=default_settings.log_level.upper(),
    format="%(asctime)s %(levelname)-7s %(name)s  %(message)s",
)
app = create_app()

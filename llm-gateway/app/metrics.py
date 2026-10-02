"""In-process request metrics.

Counters are kept per model. Latencies live in a bounded window so the
percentiles reflect recent traffic and memory does not grow with uptime.
"""

from __future__ import annotations

import statistics
import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

_WINDOW = 500


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return round(values[0], 2)
    return round(statistics.quantiles(values, n=100, method="inclusive")[pct - 1], 2)


@dataclass
class _Counters:
    requests: int = 0
    errors: int = 0
    cache_hits: int = 0
    tool_call_responses: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    latencies: deque[float] = field(default_factory=lambda: deque(maxlen=_WINDOW))

    def snapshot(self) -> dict[str, Any]:
        window = list(self.latencies)
        return {
            "requests": self.requests,
            "errors": self.errors,
            "cache_hits": self.cache_hits,
            "tool_call_responses": self.tool_call_responses,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.input_tokens + self.output_tokens,
            "cost_usd": round(self.cost_usd, 6),
            "latency_ms": {
                "avg": round(sum(window) / len(window), 2) if window else 0.0,
                "p50": _percentile(window, 50),
                "p95": _percentile(window, 95),
            },
        }


class Metrics:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._reset()

    def _reset(self) -> None:
        self._total = _Counters()
        self._by_model: dict[str, _Counters] = {}
        self._since = datetime.now(UTC)

    def record(
        self,
        model: str,
        *,
        latency_ms: float,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cost_usd: float = 0.0,
        cached: bool = False,
        tool_calls: bool = False,
        error: bool = False,
    ) -> None:
        with self._lock:
            per_model = self._by_model.setdefault(model, _Counters())
            for counters in (self._total, per_model):
                counters.requests += 1
                counters.latencies.append(latency_ms)
                if error:
                    counters.errors += 1
                    continue
                counters.cache_hits += int(cached)
                counters.tool_call_responses += int(tool_calls)
                counters.input_tokens += input_tokens
                counters.output_tokens += output_tokens
                counters.cost_usd += cost_usd

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "since": self._since.isoformat(timespec="seconds"),
                "total": self._total.snapshot(),
                "by_model": {name: c.snapshot() for name, c in sorted(self._by_model.items())},
            }

    def reset(self) -> None:
        with self._lock:
            self._reset()

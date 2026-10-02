"""Exact-match response cache.

Requests are keyed on everything that can change the output: resolved model,
messages, tool definitions, temperature and token limit. Entries expire after
a TTL and the cache is bounded, so memory use stays flat.
"""

from __future__ import annotations

import hashlib
import json
import threading
from typing import Any

from cachetools import TTLCache


class ResponseCache:
    def __init__(self, enabled: bool, max_entries: int, ttl_seconds: int):
        self.enabled = enabled
        self._cache: TTLCache[str, dict[str, Any]] = TTLCache(maxsize=max_entries, ttl=ttl_seconds)
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    @staticmethod
    def key(payload: dict[str, Any]) -> str:
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(blob.encode()).hexdigest()

    def get(self, key: str) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        with self._lock:
            value = self._cache.get(key)
            if value is None:
                self.misses += 1
                return None
            self.hits += 1
            # Hand out a copy so callers cannot mutate the cached entry.
            return json.loads(json.dumps(value))

    def set(self, key: str, value: dict[str, Any]) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._cache[key] = json.loads(json.dumps(value))

    def clear(self) -> int:
        with self._lock:
            size = len(self._cache)
            self._cache.clear()
            return size

    def stats(self) -> dict[str, Any]:
        with self._lock:
            lookups = self.hits + self.misses
            return {
                "enabled": self.enabled,
                "entries": len(self._cache),
                "max_entries": int(self._cache.maxsize),
                "ttl_seconds": int(self._cache.ttl),
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": round(self.hits / lookups, 4) if lookups else 0.0,
            }

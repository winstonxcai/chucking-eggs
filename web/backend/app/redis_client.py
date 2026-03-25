"""Async Redis client with in-memory fallback for running without Redis."""

from __future__ import annotations

import os
from typing import Any

REDIS_URL = os.getenv("REDIS_URL", "")

_client: "RedisLike | None" = None


class InMemoryRedis:
    """Minimal Redis-compatible in-memory store for local dev without Docker."""

    def __init__(self) -> None:
        self._data: dict[str, str] = {}
        self._expiry: dict[str, float] = {}

    async def get(self, key: str) -> str | None:
        self._expire_check(key)
        return self._data.get(key)

    async def set(self, key: str, value: str, ex: int | None = None) -> None:
        import time
        self._data[key] = value
        if ex:
            self._expiry[key] = time.time() + ex

    async def delete(self, key: str) -> None:
        self._data.pop(key, None)
        self._expiry.pop(key, None)

    async def exists(self, key: str) -> int:
        self._expire_check(key)
        return 1 if key in self._data else 0

    async def keys(self, pattern: str = "*") -> list[str]:
        import fnmatch, time
        now = time.time()
        # Clean expired first
        expired = [k for k, exp in self._expiry.items() if now > exp]
        for k in expired:
            self._data.pop(k, None)
            self._expiry.pop(k, None)
        return [k for k in self._data if fnmatch.fnmatch(k, pattern)]

    async def incr(self, key: str) -> int:
        val = int(self._data.get(key, "0")) + 1
        self._data[key] = str(val)
        return val

    async def expire(self, key: str, seconds: int) -> None:
        import time
        if key in self._data:
            self._expiry[key] = time.time() + seconds

    async def ping(self) -> bool:
        return True

    async def close(self) -> None:
        pass

    def _expire_check(self, key: str) -> None:
        import time
        exp = self._expiry.get(key)
        if exp and time.time() > exp:
            self._data.pop(key, None)
            self._expiry.pop(key, None)


RedisLike = Any  # Union of aioredis.Redis | InMemoryRedis


async def get_redis() -> RedisLike:
    """Get async Redis client. Falls back to in-memory if no REDIS_URL."""
    global _client
    if _client is not None:
        return _client

    if REDIS_URL:
        try:
            import redis.asyncio as aioredis
            _client = aioredis.from_url(REDIS_URL, decode_responses=True)
            await _client.ping()
            print(f"Connected to Redis at {REDIS_URL}")
        except Exception as e:
            print(f"Redis connection failed ({e}), using in-memory fallback")
            _client = InMemoryRedis()
    else:
        print("No REDIS_URL set, using in-memory fallback")
        _client = InMemoryRedis()

    return _client


async def close_redis() -> None:
    """Close the Redis connection on shutdown."""
    global _client
    if _client is not None:
        await _client.close()
        _client = None

"""Response cache (fastapi-cache2 on Redis) and its invalidation.

Routers declare what is cached with `@cache(namespace=...)`; services decide
when it is stale and call `ainvalidate()`. Workers use the same function, so a
new prediction clears the api's cached batch views immediately.
"""

from __future__ import annotations

from enum import StrEnum

from redis.asyncio import Redis


class CacheNS(StrEnum):
    me = "me"
    batches = "batches"
    predictions = "predictions"


class CacheInvalidator:
    """Deletes every cached response in the given namespaces."""

    def __init__(self, redis: Redis, prefix: str):
        self._redis = redis
        self._prefix = prefix

    def _pattern(self, ns: CacheNS) -> str:
        return f"{self._prefix}:{ns.value}:*"

    async def ainvalidate(self, *namespaces: CacheNS) -> int:
        deleted = 0
        for ns in namespaces:
            keys = [k async for k in self._redis.scan_iter(match=self._pattern(ns), count=500)]
            if keys:
                deleted += await self._redis.delete(*keys)
        return deleted

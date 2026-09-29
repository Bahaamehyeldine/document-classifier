"""Shared worker bootstrap: secrets from Vault, one event loop, DB engine, cache, blob store."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from redis import Redis
from redis.asyncio import Redis as AsyncRedis

from app.core.config import Settings, get_settings
from app.db.session import init_engine
from app.infra.blob import MinioBlobStore
from app.infra.cache import CacheInvalidator
from app.infra.vault import Secrets, load_secrets


@dataclass
class WorkerContext:
    settings: Settings
    secrets: Secrets
    loop: asyncio.AbstractEventLoop
    redis: Redis  # sync client for RQ
    invalidator: CacheInvalidator  # async client, used on `loop`
    blob: MinioBlobStore

    def run(self, coro):
        """Run a coroutine on the worker's single, long-lived event loop.

        Reusing one loop lets the async DB engine keep its connection pool
        across jobs (an engine must not be shared between event loops).
        """
        return self.loop.run_until_complete(coro)


def bootstrap(settings: Settings | None = None) -> WorkerContext:
    settings = settings or get_settings()
    secrets = load_secrets(settings.vault_addr, settings.vault_token, settings.vault_secret_path)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    init_engine(secrets.database_url)
    blob = MinioBlobStore(
        settings.minio_endpoint,
        secrets.minio_access_key,
        secrets.minio_secret_key,
        settings.minio_bucket,
    )
    blob.ensure_bucket()
    return WorkerContext(
        settings=settings,
        secrets=secrets,
        loop=loop,
        redis=Redis.from_url(settings.redis_url),
        invalidator=CacheInvalidator(
            AsyncRedis.from_url(settings.redis_url), settings.cache_prefix
        ),
        blob=blob,
    )

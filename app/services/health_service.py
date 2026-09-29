"""Readiness: the dependencies a request needs are reachable."""

from __future__ import annotations

from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from app.services import user_service


async def check_ready(session: AsyncSession, redis: Redis) -> None:
    """Raise if Postgres or Redis cannot serve a request."""
    await user_service.get_user_count(session)
    await redis.ping()

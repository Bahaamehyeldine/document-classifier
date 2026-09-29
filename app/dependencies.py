"""Composition root: builds what the HTTP layer hands to services.

Routers depend on ``ServiceContext`` and ``require``; they never import
SQLAlchemy, the cache, Redis or other infrastructure themselves.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from fastapi import Depends, HTTPException, Request, status
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth import current_active_user
from app.core.log import request_id_var
from app.db.session import get_session
from app.domain.schemas import Principal
from app.infra.blob import BlobStore
from app.infra.cache import CacheInvalidator
from app.services import permission_service


@dataclass(frozen=True)
class ServiceContext:
    """Per-request handles passed from a router into the service layer."""

    session: AsyncSession
    invalidator: CacheInvalidator
    blob: BlobStore
    redis: Redis
    request_id: str | None


async def service_context(
    request: Request, session: AsyncSession = Depends(get_session)
) -> ServiceContext:
    state = request.app.state
    return ServiceContext(
        session=session,
        invalidator=state.invalidator,
        blob=state.blob,
        redis=state.redis,
        request_id=request_id_var.get(),
    )


def require(obj: str, act: str) -> Callable:
    """Dependency: the current user must hold a role granting (obj, act)."""

    async def checker(
        user: Principal = Depends(current_active_user),
        session: AsyncSession = Depends(get_session),
    ) -> Principal:
        if not await permission_service.is_allowed(session, user.id, obj, act):
            raise HTTPException(status.HTTP_403_FORBIDDEN, f"Missing permission {obj}:{act}")
        return user

    return checker

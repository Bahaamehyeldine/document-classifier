"""Shared api dependencies: permissions, request id, cache keys, infra handles."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from fastapi import Depends, HTTPException, Request, Response, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.auth import current_active_user
from app.core.log import request_id_var
from app.db.session import get_session
from app.domain.schemas import Principal
from app.infra.blob import BlobStore
from app.infra.cache import CacheInvalidator
from app.services import permission_service
from app.services.permission_service import Actor


def require(obj: str, act: str) -> Callable:
    """Dependency: the current user must hold a role granting (obj, act)."""

    async def checker(
        user: Principal = Depends(current_active_user), session: AsyncSession = Depends(get_session)
    ) -> Principal:
        if not await permission_service.is_allowed(session, user.id, obj, act):
            raise HTTPException(status.HTTP_403_FORBIDDEN, f"Missing permission {obj}:{act}")
        return user

    return checker


def actor_of(user: Principal) -> Actor:
    return Actor(id=user.id, name=user.email)


def get_request_id() -> str | None:
    return request_id_var.get()


def get_invalidator(request: Request) -> CacheInvalidator:
    return request.app.state.invalidator


def get_blob(request: Request) -> BlobStore:
    return request.app.state.blob


# ---- cache keys ---------------------------------------------------------------
# fastapi-cache passes namespace as "<prefix>:<ns>"; keys must start with it so
# CacheInvalidator can delete a whole namespace with one pattern.


def url_key_builder(
    func: Callable[..., Any],
    namespace: str = "",
    *,
    request: Request | None = None,
    response: Response | None = None,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> str:
    assert request is not None  # always set for route handlers
    query = "&".join(f"{k}={v}" for k, v in sorted(request.query_params.multi_items()))
    return f"{namespace}:{request.url.path}?{query}"


def user_key_builder(
    func: Callable[..., Any],
    namespace: str = "",
    *,
    request: Request | None = None,
    response: Response | None = None,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> str:
    user = kwargs.get("user")
    return f"{namespace}:{user.id if user else 'anonymous'}"

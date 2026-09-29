"""HTTP helpers shared by routers: audit actor and response-cache keys."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from fastapi import Request, Response

from app.domain.schemas import Principal
from app.services.permission_service import Actor


def actor_of(user: Principal) -> Actor:
    return Actor(id=user.id, name=user.email)


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

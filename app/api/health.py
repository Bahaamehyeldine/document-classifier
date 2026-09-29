"""Liveness and readiness probes."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app.dependencies import ServiceContext, service_context
from app.domain.user import UserCountResponse
from app.services import health_service, user_service

router = APIRouter(prefix="/health", tags=["health"])


@router.get("/live")
async def live():
    return {"status": "ok"}


@router.get("/db", response_model=UserCountResponse)
async def health_db(ctx: ServiceContext = Depends(service_context)) -> UserCountResponse:
    return UserCountResponse(count=await user_service.get_user_count(ctx.session))


@router.get("/ready")
async def ready(ctx: ServiceContext = Depends(service_context)):
    await health_service.check_ready(ctx.session, ctx.redis)
    return {"status": "ready"}

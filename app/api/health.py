"""Liveness and readiness probes."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_session
from app.domain.user import UserCountResponse
from app.services import user_service

router = APIRouter(prefix="/health", tags=["health"])


@router.get("/live")
async def live():
    return {"status": "ok"}


@router.get("/db", response_model=UserCountResponse)
async def health_db(session: AsyncSession = Depends(get_session)) -> UserCountResponse:
    return UserCountResponse(count=await user_service.get_user_count(session))


@router.get("/ready")
async def ready(request: Request, session: AsyncSession = Depends(get_session)):
    await user_service.get_user_count(session)
    await request.app.state.redis.ping()
    return {"status": "ready"}

"""Audit log queries. Append-only: there is no update or delete."""

from __future__ import annotations

import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import AuditLog


async def add(
    session: AsyncSession,
    *,
    actor: str,
    actor_id: uuid.UUID | None,
    action: str,
    target: str,
    details: dict | None = None,
    request_id: str | None = None,
) -> AuditLog:
    entry = AuditLog(
        actor=actor,
        actor_id=actor_id,
        action=action,
        target=target,
        details=details or {},
        request_id=request_id,
    )
    session.add(entry)
    await session.flush()
    return entry


async def list_entries(
    session: AsyncSession, limit: int, offset: int, action: str | None = None
) -> list[AuditLog]:
    q = select(AuditLog)
    if action:
        q = q.where(AuditLog.action == action)
    rows = await session.scalars(q.order_by(AuditLog.id.desc()).limit(limit).offset(offset))
    return list(rows.all())

"""Audit log reads."""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.schemas import AuditEntry
from app.repositories import audit_repository


async def list_entries(
    session: AsyncSession, limit: int, offset: int, action: str | None
) -> list[AuditEntry]:
    rows = await audit_repository.list_entries(session, limit, offset, action)
    return [AuditEntry.model_validate(r) for r in rows]

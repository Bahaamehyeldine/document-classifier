"""Casbin policy storage: `p` rules (role, object, action) and `g` rules (user, role)."""

from __future__ import annotations

import uuid

from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import CasbinRule


async def count_permission_rules(session: AsyncSession) -> int:
    return (
        await session.scalar(
            select(func.count()).select_from(CasbinRule).where(CasbinRule.ptype == "p")
        )
        or 0
    )


async def load_rules(session: AsyncSession) -> list[tuple[str, str, str, str | None]]:
    rows = await session.execute(
        select(CasbinRule.ptype, CasbinRule.v0, CasbinRule.v1, CasbinRule.v2)
    )
    return [tuple(r) for r in rows.all()]


async def roles_for(session: AsyncSession, user_id: uuid.UUID) -> list[str]:
    rows = await session.scalars(
        select(CasbinRule.v1).where(CasbinRule.ptype == "g", CasbinRule.v0 == str(user_id))
    )
    return sorted(rows.all())


async def roles_for_many(session: AsyncSession, user_ids: list[uuid.UUID]) -> dict[str, list[str]]:
    rows = await session.execute(
        select(CasbinRule.v0, CasbinRule.v1).where(
            CasbinRule.ptype == "g", CasbinRule.v0.in_([str(u) for u in user_ids])
        )
    )
    out: dict[str, list[str]] = {}
    for uid, role in rows.all():
        out.setdefault(uid, []).append(role)
    return {k: sorted(v) for k, v in out.items()}


async def count_users_with_role(session: AsyncSession, role: str) -> int:
    return (
        await session.scalar(
            select(func.count())
            .select_from(CasbinRule)
            .where(CasbinRule.ptype == "g", CasbinRule.v1 == role)
        )
        or 0
    )


async def set_role(session: AsyncSession, user_id: uuid.UUID, role: str | None) -> None:
    """Replace the user's roles with `role` (or none)."""
    await session.execute(
        delete(CasbinRule).where(CasbinRule.ptype == "g", CasbinRule.v0 == str(user_id))
    )
    if role:
        session.add(CasbinRule(ptype="g", v0=str(user_id), v1=role))
    await session.flush()

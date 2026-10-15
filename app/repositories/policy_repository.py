"""Casbin policy storage: `p` rules (role, object, action) and `g` rules (user, role)."""

from __future__ import annotations

import uuid

from casbin_sqlalchemy_adapter import Adapter
from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session

from app.db.models import CasbinRule


def casbin_adapter(sync_session: Session) -> Adapter:
    """casbin-sqlalchemy-adapter reading `casbin_rule` on the caller's own connection.

    Called inside `AsyncSession.run_sync`, so policy reads see the request's transaction
    (including its uncommitted role changes) and no second connection is opened. The
    adapter must not create tables: the schema belongs to Alembic.
    """
    return Adapter(sync_session.connection(), db_class=CasbinRule, create_all_models=False)


async def count_permission_rules(session: AsyncSession) -> int:
    return (
        await session.scalar(
            select(func.count()).select_from(CasbinRule).where(CasbinRule.ptype == "p")
        )
        or 0
    )


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

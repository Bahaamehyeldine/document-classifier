"""User and invitation queries."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from fastapi_users_db_sqlalchemy import SQLAlchemyUserDatabase
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Invitation, User


def user_db(session: AsyncSession) -> SQLAlchemyUserDatabase:
    """fastapi-users storage adapter bound to the User table."""
    return SQLAlchemyUserDatabase(session, User)


async def count_users(session: AsyncSession) -> int:
    return await session.scalar(select(func.count()).select_from(User)) or 0


async def list_users(session: AsyncSession) -> list[User]:
    return list((await session.scalars(select(User).order_by(User.created_at))).all())


async def get_user(session: AsyncSession, user_id: uuid.UUID) -> User | None:
    return await session.get(User, user_id)


async def get_user_by_email(session: AsyncSession, email: str) -> User | None:
    return await session.scalar(select(User).where(func.lower(User.email) == email.lower()))


async def get_invitation_by_email(session: AsyncSession, email: str) -> Invitation | None:
    return await session.scalar(
        select(Invitation).where(func.lower(Invitation.email) == email.lower())
    )


async def upsert_invitation(
    session: AsyncSession, email: str, role: str, invited_by: uuid.UUID
) -> Invitation:
    inv = await get_invitation_by_email(session, email)
    if inv is None:
        inv = Invitation(email=email.lower(), role=role, invited_by=invited_by)
        session.add(inv)
    else:
        inv.role, inv.invited_by, inv.accepted_at = role, invited_by, None
    await session.flush()
    return inv


async def mark_invitation_accepted(session: AsyncSession, inv: Invitation) -> None:
    inv.accepted_at = datetime.now(UTC)
    await session.flush()

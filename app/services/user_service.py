"""Users, invitations and role changes."""

from __future__ import annotations

import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.schemas import (
    InvitationRead,
    MeResponse,
    Permission,
    Principal,
    Role,
    UserWithRoles,
)
from app.infra.cache import CacheInvalidator, CacheNS
from app.repositories import audit_repository, policy_repository, user_repository
from app.services import permission_service
from app.services.errors import ConflictError, NotFoundError, PermissionDeniedError
from app.services.permission_service import Actor


async def get_user_count(session: AsyncSession) -> int:
    return await user_repository.count_users(session)


async def me(session: AsyncSession, user: Principal) -> MeResponse:
    roles, perms = await permission_service.permissions_for(session, user.id)
    return MeResponse(
        id=user.id,
        email=user.email,
        roles=[Role(r) for r in roles],
        permissions=[Permission(obj=o, act=a) for o, a in perms],
    )


async def list_users(session: AsyncSession) -> list[UserWithRoles]:
    users = await user_repository.list_users(session)
    roles = await policy_repository.roles_for_many(session, [u.id for u in users])
    return [
        UserWithRoles(
            id=u.id,
            email=u.email,
            is_active=u.is_active,
            created_at=u.created_at,
            roles=[Role(r) for r in roles.get(str(u.id), [])],
        )
        for u in users
    ]


async def _change_role(
    session: AsyncSession, actor: Actor, user: Principal, role: Role | None, request_id: str | None
) -> None:
    old = await policy_repository.roles_for(session, user.id)
    new = [role.value] if role else []
    if old == new:
        return
    if Role.admin.value in old and Role.admin.value not in new:
        if await policy_repository.count_users_with_role(session, Role.admin.value) <= 1:
            raise ConflictError("Cannot remove the last admin")
    await policy_repository.set_role(session, user.id, role.value if role else None)
    await audit_repository.add(
        session,
        actor=actor.name,
        actor_id=actor.id,
        action="role.changed",
        target=f"user:{user.id}",
        details={"email": user.email, "from": old, "to": new},
        request_id=request_id,
    )


async def set_role(
    session: AsyncSession,
    invalidator: CacheInvalidator,
    actor: Actor,
    user_id: uuid.UUID,
    role: Role | None,
    request_id: str | None,
) -> list[Role]:
    user = await user_repository.get_user(session, user_id)
    if user is None:
        raise NotFoundError(f"User {user_id} not found")
    await _change_role(session, actor, user, role, request_id)
    await session.commit()
    await invalidator.ainvalidate(CacheNS.me)
    return [Role(r) for r in await policy_repository.roles_for(session, user_id)]


async def invite(
    session: AsyncSession,
    invalidator: CacheInvalidator,
    actor: Actor,
    email: str,
    role: Role,
    request_id: str | None,
) -> InvitationRead:
    """Pre-assign a role to an email. If that user already exists, grant it now."""
    if actor.id is None:
        raise PermissionDeniedError("Invitations must be issued by a user")
    inv = await user_repository.upsert_invitation(session, email, role.value, actor.id)
    await audit_repository.add(
        session,
        actor=actor.name,
        actor_id=actor.id,
        action="invitation.created",
        target=f"email:{inv.email}",
        details={"role": role.value},
        request_id=request_id,
    )
    existing = await user_repository.get_user_by_email(session, inv.email)
    if existing is not None:
        await _change_role(session, actor, existing, role, request_id)
        await user_repository.mark_invitation_accepted(session, inv)
    await session.commit()
    await invalidator.ainvalidate(CacheNS.me)
    return InvitationRead.model_validate(inv)


async def accept_invitation(
    session: AsyncSession, user: Principal, request_id: str | None
) -> Role | None:
    """Called after registration: grant the invited role, if any."""
    inv = await user_repository.get_invitation_by_email(session, user.email)
    if inv is None or inv.accepted_at is not None:
        return None
    system = Actor(id=None, name="system:registration")
    await _change_role(session, system, user, Role(inv.role), request_id)
    await user_repository.mark_invitation_accepted(session, inv)
    await session.commit()
    return Role(inv.role)


async def grant_bootstrap_admin(session: AsyncSession, user: Principal) -> None:
    """Used by the `create-admin` CLI to create the first administrator."""
    system = Actor(id=None, name="system:cli")
    await _change_role(session, system, user, Role.admin, request_id=None)
    await session.commit()

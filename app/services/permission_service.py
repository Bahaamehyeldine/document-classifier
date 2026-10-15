"""Role-based permissions with Casbin.

Policies are read from the `casbin_rule` table on every check, so a role
change applies to the user's very next request without a new login. The table
is tiny (one row per permission and per user), so this costs a single query.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

import casbin
from casbin.model import Model
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session

from app.repositories import policy_repository

RBAC_MODEL = """
[request_definition]
r = sub, obj, act

[policy_definition]
p = sub, obj, act

[role_definition]
g = _, _

[policy_effect]
e = some(where (p.eft == allow))

[matchers]
m = g(r.sub, p.sub) && r.obj == p.obj && r.act == p.act
"""


@dataclass(frozen=True)
class Actor:
    """Who is performing an action, for authorization and the audit log."""

    id: uuid.UUID | None
    name: str  # email for users, "system:<service>" for workers


def _enforcer(sync_session: Session) -> casbin.Enforcer:
    """A fresh enforcer; constructing it with the adapter loads the current policy."""
    model = Model()
    model.load_model_from_text(RBAC_MODEL)
    return casbin.Enforcer(model, policy_repository.casbin_adapter(sync_session))


async def is_allowed(session: AsyncSession, user_id: uuid.UUID, obj: str, act: str) -> bool:
    # The adapter is synchronous; run_sync executes it on the session's own connection.
    return await session.run_sync(lambda s: _enforcer(s).enforce(str(user_id), obj, act))


def _roles_and_permissions(
    sync_session: Session, user_id: uuid.UUID
) -> tuple[list[str], list[tuple[str, str]]]:
    enforcer = _enforcer(sync_session)
    roles = sorted(enforcer.get_roles_for_user(str(user_id)))
    perms = sorted(
        {(p[1], p[2]) for role in roles for p in enforcer.get_permissions_for_user(role)}
    )
    return roles, perms


async def permissions_for(
    session: AsyncSession, user_id: uuid.UUID
) -> tuple[list[str], list[tuple[str, str]]]:
    return await session.run_sync(_roles_and_permissions, user_id)


async def policy_is_seeded(session: AsyncSession) -> bool:
    return await policy_repository.count_permission_rules(session) > 0

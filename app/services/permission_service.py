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


def _enforcer(rules: list[tuple[str, str, str, str | None]]) -> casbin.Enforcer:
    model = Model()
    model.load_model_from_text(RBAC_MODEL)
    enforcer = casbin.Enforcer(model)
    for ptype, v0, v1, v2 in rules:
        if ptype == "p":
            enforcer.add_policy(v0, v1, v2)
        elif ptype == "g":
            enforcer.add_grouping_policy(v0, v1)
    return enforcer


async def is_allowed(session: AsyncSession, user_id: uuid.UUID, obj: str, act: str) -> bool:
    enforcer = _enforcer(await policy_repository.load_rules(session))
    return enforcer.enforce(str(user_id), obj, act)


async def permissions_for(
    session: AsyncSession, user_id: uuid.UUID
) -> tuple[list[str], list[tuple[str, str]]]:
    rules = await policy_repository.load_rules(session)
    roles = sorted(v1 for p, v0, v1, _ in rules if p == "g" and v0 == str(user_id))
    perms = sorted({(v1, v2) for p, v0, v1, v2 in rules if p == "p" and v0 in roles and v2})
    return roles, perms


async def policy_is_seeded(session: AsyncSession) -> bool:
    return await policy_repository.count_permission_rules(session) > 0

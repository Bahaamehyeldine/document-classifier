"""Authentication: fastapi-users with JWT bearer tokens.

The signing key is read from Vault at startup (app.state.secrets), never from
the environment or the code.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from typing import Any

from fastapi import Depends, Request
from fastapi_users import BaseUserManager, FastAPIUsers, UUIDIDMixin
from fastapi_users.authentication import AuthenticationBackend, BearerTransport, JWTStrategy
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.log import get_logger, request_id_var
from app.db.session import get_session
from app.repositories.user_repository import user_db as make_user_db
from app.services import user_service

log = get_logger(__name__)


class UserManager(UUIDIDMixin, BaseUserManager[Any, uuid.UUID]):
    def __init__(self, session: AsyncSession, token_secret: str):
        super().__init__(make_user_db(session))
        self.session = session
        self.reset_password_token_secret = token_secret
        self.verification_token_secret = token_secret

    async def on_after_register(self, user, request: Request | None = None) -> None:
        role = await user_service.accept_invitation(self.session, user, request_id_var.get())
        log.info("user.registered", user_id=str(user.id), granted_role=role.value if role else None)


async def get_user_manager(
    request: Request, session: AsyncSession = Depends(get_session)
) -> AsyncIterator[UserManager]:
    yield UserManager(session, request.app.state.secrets.jwt_secret)


def get_jwt_strategy(request: Request) -> JWTStrategy:
    return JWTStrategy(
        secret=request.app.state.secrets.jwt_secret,
        lifetime_seconds=request.app.state.settings.jwt_lifetime_seconds,
    )


auth_backend = AuthenticationBackend(
    name="jwt", transport=BearerTransport(tokenUrl="auth/jwt/login"), get_strategy=get_jwt_strategy
)

fastapi_users = FastAPIUsers[Any, uuid.UUID](get_user_manager, [auth_backend])
current_active_user = fastapi_users.current_user(active=True)

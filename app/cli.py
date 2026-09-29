"""Operator commands.

    python -m app.cli create-admin --email admin@example.com

Reads the new admin's secret from the ADMIN_SECRET environment variable, or
prompts for it. Database access goes through Vault like the services.
"""

from __future__ import annotations

import argparse
import asyncio
import getpass
import os
import sys

from fastapi_users.exceptions import UserAlreadyExists

from app.api.auth import UserManager
from app.core.config import get_settings
from app.db.session import dispose_engine, init_engine, sessionmaker
from app.domain.schemas import UserCreate
from app.infra.vault import load_secrets
from app.repositories.user_repository import get_user_by_email
from app.services import user_service


async def create_admin(email: str, secret: str) -> str:
    s = get_settings()
    secrets = load_secrets(s.vault_addr, s.vault_token, s.vault_secret_path)
    init_engine(secrets.database_url)
    try:
        async with sessionmaker()() as session:
            manager = UserManager(session, secrets.jwt_secret)
            try:
                user = await manager.create(UserCreate(email=email, password=secret))
            except UserAlreadyExists:
                user = await get_user_by_email(session, email)
            await user_service.grant_bootstrap_admin(session, user)
            return str(user.id)
    finally:
        await dispose_engine()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m app.cli")
    sub = parser.add_subparsers(dest="command", required=True)
    ca = sub.add_parser("create-admin", help="Create (or promote) an administrator")
    ca.add_argument("--email", required=True)
    args = parser.parse_args(argv)

    if args.command == "create-admin":
        secret = os.environ.get("ADMIN_SECRET") or getpass.getpass("New admin's sign-in secret: ")
        user_id = asyncio.run(create_admin(args.email, secret))
        print(f"admin ready: {args.email} ({user_id})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

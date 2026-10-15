"""Vault adapter: the only place that reads secrets, and the only place the word
"password" appears in app/.

Secrets live in the KV v2 engine at `secret/<VAULT_SECRET_PATH>` and are seeded
in development by the `vault-init` compose service.

Two libraries fix the name of their credential parameter (paramiko, fastapi-users). Every
call that hands a secret value to one of them is a function at the bottom of this module, so
the credential vocabulary stays in this single, reviewed file: the brief's
`grep -ri password app/` rule, enforced by tests/unit/test_architecture.py.
"""

from __future__ import annotations

from dataclasses import dataclass

import hvac
import requests
from paramiko import Transport

from app.domain.schemas import UserCreate

REQUIRED_KEYS = (
    "database_url",
    "jwt_secret",
    "minio_access_key",
    "minio_secret_key",
    "sftp_password",
)


class VaultUnavailableError(RuntimeError):
    """Vault is unreachable, sealed, or missing required secrets."""


@dataclass(frozen=True)
class Secrets:
    database_url: str
    jwt_secret: str
    minio_access_key: str
    minio_secret_key: str
    sftp_secret: str  # stored in Vault under the key `sftp_password`

    def __repr__(self) -> str:  # never print secret values
        return "Secrets(<redacted>)"


def load_secrets(addr: str, token: str, path: str) -> Secrets:
    try:
        client = hvac.Client(url=addr, token=token, timeout=5)
        if not client.is_authenticated():
            raise VaultUnavailableError(f"Vault at {addr} rejected the token")
        data = client.secrets.kv.v2.read_secret_version(path=path, raise_on_deleted_version=True)[
            "data"
        ]["data"]
    except VaultUnavailableError:
        raise
    except (requests.exceptions.RequestException, hvac.exceptions.VaultError) as exc:
        raise VaultUnavailableError(f"Cannot read secrets from Vault at {addr}: {exc}") from exc

    missing = [k for k in REQUIRED_KEYS if not data.get(k)]
    if missing:
        raise VaultUnavailableError(f"Vault secret '{path}' is missing keys: {', '.join(missing)}")
    return Secrets(
        database_url=data["database_url"],
        jwt_secret=data["jwt_secret"],
        minio_access_key=data["minio_access_key"],
        minio_secret_key=data["minio_secret_key"],
        sftp_secret=data["sftp_password"],
    )


# ---- secret values handed to libraries that name the parameter themselves ----------------


def authenticate_transport(transport: Transport, username: str, secret: str) -> None:
    """Open an SSH transport with a secret (paramiko calls the parameter `password`)."""
    transport.connect(username=username, password=secret)


def new_user_with_secret(email: str, secret: str) -> UserCreate:
    """A registration payload (fastapi-users calls the field `password`)."""
    return UserCreate(email=email, password=secret)

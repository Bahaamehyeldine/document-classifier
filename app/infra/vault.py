"""Vault adapter: the only place that reads secrets.

Secrets live in the KV v2 engine at `secret/<VAULT_SECRET_PATH>` and are seeded
in development by the `vault-init` compose service.
"""

from __future__ import annotations

from dataclasses import dataclass

import hvac
import requests

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
    sftp_password: str

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
    return Secrets(**{k: data[k] for k in REQUIRED_KEYS})

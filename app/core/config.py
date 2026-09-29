"""Runtime configuration.

Non-secret settings (hostnames, ports, queue names) come from the environment.
Secrets (database URL, JWT signing key, MinIO and SFTP credentials) come only
from Vault, via `app.infra.vault.load_secrets`. Nothing secret has a default.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclass(frozen=True)
class Settings:
    vault_addr: str = field(default_factory=lambda: _env("VAULT_ADDR", "http://vault:8200"))
    vault_token: str = field(default_factory=lambda: os.environ.get("VAULT_TOKEN", ""))
    vault_secret_path: str = field(
        default_factory=lambda: _env("VAULT_SECRET_PATH", "docclassifier")
    )

    redis_url: str = field(default_factory=lambda: _env("REDIS_URL", "redis://redis:6379/0"))
    queue_name: str = field(default_factory=lambda: _env("QUEUE_NAME", "classify"))
    cache_prefix: str = field(default_factory=lambda: _env("CACHE_PREFIX", "docclassifier"))
    cache_ttl_seconds: int = field(default_factory=lambda: int(_env("CACHE_TTL_SECONDS", "60")))

    minio_endpoint: str = field(default_factory=lambda: _env("MINIO_ENDPOINT", "minio:9000"))
    minio_bucket: str = field(default_factory=lambda: _env("MINIO_BUCKET", "documents"))

    sftp_host: str = field(default_factory=lambda: _env("SFTP_HOST", "sftp"))
    sftp_port: int = field(default_factory=lambda: int(_env("SFTP_PORT_INTERNAL", "22")))
    sftp_user: str = field(default_factory=lambda: _env("SFTP_USER", "scanner"))
    sftp_dir: str = field(default_factory=lambda: _env("SFTP_DIR", "upload"))
    sftp_poll_seconds: float = field(default_factory=lambda: float(_env("SFTP_POLL_SECONDS", "2")))

    jwt_lifetime_seconds: int = field(
        default_factory=lambda: int(_env("JWT_LIFETIME_SECONDS", "3600"))
    )


def get_settings() -> Settings:
    return Settings()

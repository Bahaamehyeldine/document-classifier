"""API test fixtures: a real Postgres database and Redis, fakes for Vault, MinIO and the queue.

Point TEST_DATABASE_URL / TEST_REDIS_URL at throwaway instances (CI provides
service containers). The schema is built with the real Alembic migrations.
"""

from __future__ import annotations

import os
import uuid
from datetime import timedelta

import pytest
import pytest_asyncio
from alembic.config import Config
from asgi_lifespan import LifespanManager
from httpx import ASGITransport, AsyncClient
from redis import Redis
from sqlalchemy import create_engine, text

from alembic import command
from app.core.config import Settings
from app.infra.vault import Secrets
from app.main import create_app

TEST_DATABASE_URL = os.environ.get(
    "TEST_DATABASE_URL", "postgresql+psycopg://postgres@localhost:5433/docclassifier_test"
)
TEST_REDIS_URL = os.environ.get("TEST_REDIS_URL", "redis://localhost:6380/15")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class FakeBlob:
    def __init__(self):
        self.objects: dict[str, bytes] = {}

    def put(self, key, data, content_type):
        self.objects[key] = data

    def get(self, key):
        return self.objects[key]

    def url(self, key, expires=timedelta(minutes=15)):
        return f"https://blob.test/{key}?expires={int(expires.total_seconds())}"


class FakeQueue:
    def __init__(self):
        self.jobs: list[tuple[uuid.UUID, str | None]] = []

    def enqueue_classification(self, document_id, request_id):
        self.jobs.append((document_id, request_id))
        return f"job-{len(self.jobs)}"


def _reset_schema():
    admin_url = TEST_DATABASE_URL.rsplit("/", 1)[0] + "/postgres"
    db = TEST_DATABASE_URL.rsplit("/", 1)[1]
    eng = create_engine(admin_url, isolation_level="AUTOCOMMIT")
    with eng.connect() as c:
        c.execute(text(f'DROP DATABASE IF EXISTS "{db}" WITH (FORCE)'))
        c.execute(text(f'CREATE DATABASE "{db}"'))
    eng.dispose()
    cfg = Config(os.path.join(ROOT, "alembic.ini"))
    cfg.set_main_option("script_location", os.path.join(ROOT, "alembic"))
    os.environ["DATABASE_URL"] = TEST_DATABASE_URL
    os.environ.pop("VAULT_TOKEN", None)
    command.upgrade(cfg, "head")


@pytest.fixture(scope="session")
def migrated_db():
    _reset_schema()
    yield TEST_DATABASE_URL


@pytest.fixture
def clean_db(migrated_db):
    eng = create_engine(migrated_db)
    with eng.begin() as c:
        c.execute(
            text(
                'TRUNCATE audit_log, prediction, document, batch, invitation, "user" '
                "RESTART IDENTITY CASCADE"
            )
        )
        c.execute(text("DELETE FROM casbin_rule WHERE ptype = 'g'"))
    eng.dispose()
    Redis.from_url(TEST_REDIS_URL).flushdb()
    yield migrated_db


@pytest.fixture
def fakes():
    return {"blob": FakeBlob(), "queue": FakeQueue()}


@pytest_asyncio.fixture
async def client(clean_db, fakes):
    settings = Settings(redis_url=TEST_REDIS_URL, cache_prefix="test")
    secrets = Secrets(
        database_url=clean_db,
        jwt_secret="test-signing-key-0123456789abcdef-0123",
        minio_access_key="x",
        minio_secret_key="y",
        sftp_password="z",
    )
    app = create_app(
        settings,
        secrets_loader=lambda s: secrets,
        model_verifier=lambda: None,
        blob_factory=lambda s, sec: fakes["blob"],
    )
    async with LifespanManager(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            ac.app = app
            yield ac


async def register(client, email, secret="S3cure-pass-1"):
    r = await client.post("/auth/register", json={"email": email, "password": secret})
    assert r.status_code == 201, r.text
    r = await client.post("/auth/jwt/login", data={"username": email, "password": secret})
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['access_token']}"}, None


async def make_user(client, email, role=None):
    """Register a user and (optionally) grant a role directly in the policy table."""
    headers, _ = await register(client, email)
    me = (await client.get("/me", headers=headers)).json()
    if role:
        eng = create_engine(TEST_DATABASE_URL)
        with eng.begin() as c:
            c.execute(
                text("INSERT INTO casbin_rule (ptype, v0, v1) VALUES ('g', :u, :r)"),
                {"u": me["id"], "r": role},
            )
        eng.dispose()
        Redis.from_url(TEST_REDIS_URL).flushdb()
    return headers, uuid.UUID(me["id"])

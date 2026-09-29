"""The api refuses to start on missing secrets, bad model artifacts or an empty policy table."""

import pytest
from asgi_lifespan import LifespanManager
from sqlalchemy import create_engine, text

from app.classifier.artifacts import ClassifierStartupError
from app.core.config import Settings
from app.infra.vault import Secrets, VaultUnavailableError
from app.main import PolicyNotSeededError, create_app
from tests.api.conftest import TEST_DATABASE_URL, TEST_REDIS_URL, FakeBlob

SECRETS = Secrets(
    database_url=TEST_DATABASE_URL,
    jwt_secret="k" * 40,
    minio_access_key="x",
    minio_secret_key="y",
    sftp_password="z",
)


def _app(**overrides):
    kwargs = dict(
        secrets_loader=lambda s: SECRETS,
        model_verifier=lambda: None,
        blob_factory=lambda s, sec: FakeBlob(),
    )
    kwargs.update(overrides)
    return create_app(Settings(redis_url=TEST_REDIS_URL), **kwargs)


async def test_refuses_when_vault_unreachable(clean_db):
    def down(_):
        raise VaultUnavailableError("connection refused")

    with pytest.raises(VaultUnavailableError):
        async with LifespanManager(_app(secrets_loader=down)):
            pass


async def test_refuses_when_model_artifacts_invalid(clean_db):
    def bad():
        raise ClassifierStartupError("Weights SHA-256 mismatch")

    with pytest.raises(ClassifierStartupError):
        async with LifespanManager(_app(model_verifier=bad)):
            pass


async def test_refuses_when_policy_table_empty(clean_db):
    eng = create_engine(TEST_DATABASE_URL)
    with eng.begin() as c:
        saved = c.execute(text("SELECT ptype, v0, v1, v2 FROM casbin_rule WHERE ptype='p'")).all()
        c.execute(text("DELETE FROM casbin_rule"))
    try:
        with pytest.raises(PolicyNotSeededError):
            async with LifespanManager(_app()):
                pass
    finally:
        with eng.begin() as c:
            for row in saved:
                c.execute(
                    text("INSERT INTO casbin_rule (ptype, v0, v1, v2) VALUES (:a,:b,:c,:d)"),
                    dict(zip("abcd", row, strict=True)),
                )
        eng.dispose()


def test_real_model_check_refuses_without_weights(tmp_path):
    from app.classifier.artifacts import verify_artifacts

    with pytest.raises(ClassifierStartupError, match="weights missing"):
        verify_artifacts(tmp_path / "classifier.pt", tmp_path / "model_card.json")

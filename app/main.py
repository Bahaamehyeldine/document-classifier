"""FastAPI application factory.

Startup refuses to proceed (the process exits) when:
  * Vault is unreachable or missing secrets,
  * the classifier weights are missing, fail their SHA-256 check, or are below
    the README quality gate (checked without loading PyTorch; the api never runs inference),
  * the Casbin policy table is empty.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi_cache import FastAPICache
from fastapi_cache.backends.redis import RedisBackend
from redis.asyncio import Redis

from app.api import health
from app.api.auth import auth_backend, fastapi_users
from app.api.routes import audit_router, batches_router, me_router, predictions_router, users_router
from app.classifier.artifacts import verify_artifacts
from app.core.config import Settings, get_settings
from app.core.log import configure_logging, get_logger, new_request_id, request_id_var
from app.db.session import dispose_engine, init_engine, sessionmaker
from app.domain.schemas import UserCreate, UserRead
from app.infra.blob import BlobStore, MinioBlobStore
from app.infra.cache import CacheInvalidator
from app.infra.vault import Secrets, load_secrets
from app.services import permission_service
from app.services.errors import ConflictError, NotFoundError, PermissionDeniedError

log = get_logger(__name__)

REQUEST_ID_HEADER = "X-Request-ID"


class PolicyNotSeededError(RuntimeError):
    pass


def create_app(
    settings: Settings | None = None,
    *,
    secrets_loader: Callable[[Settings], Secrets] | None = None,
    model_verifier: Callable[[], object] = verify_artifacts,
    blob_factory: Callable[[Settings, Secrets], BlobStore] | None = None,
) -> FastAPI:
    settings = settings or get_settings()
    secrets_loader = secrets_loader or (
        lambda s: load_secrets(s.vault_addr, s.vault_token, s.vault_secret_path)
    )
    blob_factory = blob_factory or (
        lambda s, sec: MinioBlobStore(
            s.minio_endpoint, sec.minio_access_key, sec.minio_secret_key, s.minio_bucket
        )
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        configure_logging("api")
        secrets = secrets_loader(settings)  # raises VaultUnavailableError
        model_verifier()  # raises ClassifierStartupError
        init_engine(secrets.database_url)
        async with sessionmaker()() as session:
            if not await permission_service.policy_is_seeded(session):
                raise PolicyNotSeededError("Casbin policy table is empty; run the migrations")
        redis = Redis.from_url(settings.redis_url)
        FastAPICache.init(RedisBackend(redis), prefix=settings.cache_prefix)

        app.state.settings = settings
        app.state.secrets = secrets
        app.state.redis = redis
        app.state.invalidator = CacheInvalidator(redis, settings.cache_prefix)
        app.state.blob = blob_factory(settings, secrets)
        log.info("api.started")
        try:
            yield
        finally:
            await redis.aclose()
            await dispose_engine()

    app = FastAPI(
        title="Document Classifier",
        version="0.2.0",
        description=(
            "Scanned-document layout classification (RVL-CDIP, ConvNeXt) with role-based review."
        ),
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        rid = request.headers.get(REQUEST_ID_HEADER) or new_request_id()
        token = request_id_var.set(rid)
        start = time.perf_counter()
        try:
            response = await call_next(request)
        finally:
            request_id_var.reset(token)
        elapsed_ms = (time.perf_counter() - start) * 1000
        response.headers[REQUEST_ID_HEADER] = rid
        log.info(
            "http.request",
            request_id=rid,
            method=request.method,
            path=request.url.path,
            status=response.status_code,
            latency_ms=round(elapsed_ms, 2),
            cache=response.headers.get("X-FastAPI-Cache"),
        )
        return response

    for exc_type, code in (
        (NotFoundError, 404),
        (ConflictError, 409),
        (PermissionDeniedError, 403),
    ):
        app.add_exception_handler(
            exc_type,
            lambda _req, exc, code=code: JSONResponse({"detail": str(exc)}, status_code=code),
        )

    app.include_router(health.router)
    app.include_router(
        fastapi_users.get_auth_router(auth_backend), prefix="/auth/jwt", tags=["auth"]
    )
    app.include_router(
        fastapi_users.get_register_router(UserRead, UserCreate), prefix="/auth", tags=["auth"]
    )
    for router in (me_router, users_router, batches_router, predictions_router, audit_router):
        app.include_router(router)
    return app

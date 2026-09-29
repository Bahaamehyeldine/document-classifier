"""HTTP routes. Routers only translate HTTP <-> service calls: no SQL, no cache, no infra."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, Query
from fastapi_cache.decorator import cache

from app.api.deps import actor_of, url_key_builder, user_key_builder
from app.auth import current_active_user
from app.dependencies import ServiceContext, require, service_context
from app.domain.schemas import (
    AuditEntry,
    BatchDetail,
    BatchSummary,
    InvitationCreate,
    InvitationRead,
    MeResponse,
    OverlayLink,
    PredictionOut,
    Principal,
    RelabelRequest,
    Role,
    RoleUpdate,
    UserWithRoles,
)
from app.services import audit_service, batch_service, prediction_service, user_service

CACHE_TTL = 60

me_router = APIRouter(tags=["me"])
users_router = APIRouter(prefix="/users", tags=["users"])
batches_router = APIRouter(prefix="/batches", tags=["batches"])
predictions_router = APIRouter(tags=["predictions"])
audit_router = APIRouter(prefix="/audit", tags=["audit"])


@me_router.get("/me", response_model=MeResponse)
@cache(expire=CACHE_TTL, namespace="me", key_builder=user_key_builder)
async def get_me(
    user: Principal = Depends(current_active_user),
    ctx: ServiceContext = Depends(service_context),
):
    return await user_service.me(ctx.session, user)


# ---- users (admin) -----------------------------------------------------------


@users_router.get("", response_model=list[UserWithRoles])
async def list_users(
    user: Principal = Depends(require("users", "read")),
    ctx: ServiceContext = Depends(service_context),
):
    return await user_service.list_users(ctx.session)


@users_router.post("/invitations", response_model=InvitationRead, status_code=201)
async def invite_user(
    body: InvitationCreate,
    user: Principal = Depends(require("users", "write")),
    ctx: ServiceContext = Depends(service_context),
):
    return await user_service.invite(
        ctx.session, ctx.invalidator, actor_of(user), body.email, body.role, ctx.request_id
    )


@users_router.put("/{user_id}/role", response_model=list[Role])
async def set_user_role(
    user_id: uuid.UUID,
    body: RoleUpdate,
    user: Principal = Depends(require("users", "write")),
    ctx: ServiceContext = Depends(service_context),
):
    return await user_service.set_role(
        ctx.session, ctx.invalidator, actor_of(user), user_id, body.role, ctx.request_id
    )


# ---- batches -----------------------------------------------------------------


@batches_router.get("", response_model=list[BatchSummary])
@cache(expire=CACHE_TTL, namespace="batches", key_builder=url_key_builder)
async def list_batches(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    user: Principal = Depends(require("batches", "read")),
    ctx: ServiceContext = Depends(service_context),
):
    return await batch_service.list_batches(ctx.session, limit, offset)


@batches_router.get("/{batch_id}", response_model=BatchDetail)
@cache(expire=CACHE_TTL, namespace="batches", key_builder=url_key_builder)
async def get_batch(
    batch_id: uuid.UUID,
    user: Principal = Depends(require("batches", "read")),
    ctx: ServiceContext = Depends(service_context),
):
    return await batch_service.get_batch(ctx.session, batch_id)


# ---- predictions -------------------------------------------------------------


@predictions_router.get("/predictions/recent", response_model=list[PredictionOut])
@cache(expire=CACHE_TTL, namespace="predictions", key_builder=url_key_builder)
async def recent_predictions(
    limit: int = Query(20, ge=1, le=100),
    user: Principal = Depends(require("predictions", "read")),
    ctx: ServiceContext = Depends(service_context),
):
    return await prediction_service.recent(ctx.session, limit)


@predictions_router.patch("/predictions/{prediction_id}/label", response_model=PredictionOut)
async def relabel_prediction(
    prediction_id: uuid.UUID,
    body: RelabelRequest,
    user: Principal = Depends(require("predictions", "relabel")),
    ctx: ServiceContext = Depends(service_context),
):
    return await prediction_service.relabel(
        ctx.session, ctx.invalidator, actor_of(user), prediction_id, body.label, ctx.request_id
    )


@predictions_router.get("/documents/{document_id}/overlay", response_model=OverlayLink)
async def document_overlay(
    document_id: uuid.UUID,
    user: Principal = Depends(require("predictions", "read")),
    ctx: ServiceContext = Depends(service_context),
):
    return await prediction_service.overlay_link(ctx.session, ctx.blob, document_id)


# ---- audit -------------------------------------------------------------------


@audit_router.get("", response_model=list[AuditEntry])
async def audit_log(
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    action: str | None = Query(None, description="Filter, e.g. role.changed"),
    user: Principal = Depends(require("audit", "read")),
    ctx: ServiceContext = Depends(service_context),
):
    return await audit_service.list_entries(ctx.session, limit, offset, action)

"""Pydantic domain models: the api's request and response shapes.

Distinct from the SQLAlchemy ORM models in app/db/models.py.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import StrEnum
from typing import Literal, Protocol

from fastapi_users import schemas
from pydantic import BaseModel, ConfigDict, EmailStr, Field

from app.classifier.labels import RVL_CDIP_CLASSES

DocumentLabel = Literal[RVL_CDIP_CLASSES]  # type: ignore[valid-type]


class Principal(Protocol):
    """An authenticated user as seen by routers and services."""

    id: uuid.UUID
    email: str


class Role(StrEnum):
    admin = "admin"
    reviewer = "reviewer"
    auditor = "auditor"


# ---- users ------------------------------------------------------------------


class UserRead(schemas.BaseUser[uuid.UUID]):
    pass


class UserCreate(schemas.BaseUserCreate):
    pass


class UserUpdate(schemas.BaseUserUpdate):
    pass


class Permission(BaseModel):
    obj: str
    act: str


class MeResponse(BaseModel):
    id: uuid.UUID
    email: EmailStr
    roles: list[Role]
    permissions: list[Permission]


class UserWithRoles(BaseModel):
    id: uuid.UUID
    email: EmailStr
    is_active: bool
    roles: list[Role]
    created_at: datetime


class InvitationCreate(BaseModel):
    email: EmailStr
    role: Role


class InvitationRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: uuid.UUID
    email: EmailStr
    role: Role
    created_at: datetime
    accepted_at: datetime | None


class RoleUpdate(BaseModel):
    role: Role | None = Field(description="New role, or null to revoke all roles")


# ---- batches & predictions --------------------------------------------------


class PredictionView(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: uuid.UUID
    document_id: uuid.UUID
    label: str
    confidence: float
    top5: list[tuple[str, float]]
    relabeled_to: str | None
    relabeled_by: uuid.UUID | None
    relabeled_at: datetime | None
    latency_ms: float
    created_at: datetime


class PredictionOut(PredictionView):
    needs_review: bool
    effective_label: str


class DocumentView(BaseModel):
    id: uuid.UUID
    filename: str
    status: str
    error: str | None
    prediction: PredictionOut | None


class BatchSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: uuid.UUID
    source: str
    status: str
    document_count: int
    created_at: datetime
    updated_at: datetime


class BatchDetail(BatchSummary):
    documents: list[DocumentView]


class RelabelRequest(BaseModel):
    label: DocumentLabel


class OverlayLink(BaseModel):
    url: str
    expires_in_seconds: int


# ---- audit -----------------------------------------------------------------


class AuditEntry(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    actor: str
    action: str
    target: str
    details: dict
    request_id: str | None
    created_at: datetime

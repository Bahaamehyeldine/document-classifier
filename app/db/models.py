"""SQLAlchemy ORM models. Imported only by repositories (and Alembic)."""

from __future__ import annotations

import uuid
from datetime import datetime

from fastapi_users.db import SQLAlchemyBaseUserTableUUID
from fastapi_users_db_sqlalchemy.generics import GUID
from sqlalchemy import JSON, DateTime, Enum, Float, ForeignKey, Index, Integer, String, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from app.domain.enums import BatchStatus, DocumentStatus


class Base(DeclarativeBase):
    pass


def _uuid() -> uuid.UUID:
    return uuid.uuid4()


class User(SQLAlchemyBaseUserTableUUID, Base):
    __tablename__ = "user"
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class Invitation(Base):
    """An admin pre-assigns a role to an email; the role is granted on registration."""

    __tablename__ = "invitation"
    id: Mapped[uuid.UUID] = mapped_column(GUID, primary_key=True, default=_uuid)
    email: Mapped[str] = mapped_column(String(320), unique=True, index=True)
    role: Mapped[str] = mapped_column(String(32))
    invited_by: Mapped[uuid.UUID] = mapped_column(GUID, ForeignKey("user.id"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    accepted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class Batch(Base):
    __tablename__ = "batch"
    id: Mapped[uuid.UUID] = mapped_column(GUID, primary_key=True, default=_uuid)
    source: Mapped[str] = mapped_column(String(32), default="sftp")
    status: Mapped[BatchStatus] = mapped_column(
        Enum(BatchStatus, name="batch_status"), default=BatchStatus.received
    )
    document_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )
    documents: Mapped[list[Document]] = relationship(
        back_populates="batch", order_by="Document.filename"
    )


class Document(Base):
    __tablename__ = "document"
    id: Mapped[uuid.UUID] = mapped_column(GUID, primary_key=True, default=_uuid)
    batch_id: Mapped[uuid.UUID] = mapped_column(GUID, ForeignKey("batch.id", ondelete="CASCADE"))
    filename: Mapped[str] = mapped_column(String(512))
    sha256: Mapped[str] = mapped_column(String(64))
    blob_key: Mapped[str] = mapped_column(String(1024))
    status: Mapped[DocumentStatus] = mapped_column(
        Enum(DocumentStatus, name="document_status"), default=DocumentStatus.queued
    )
    error: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    batch: Mapped[Batch] = relationship(back_populates="documents")
    prediction: Mapped[Prediction | None] = relationship(back_populates="document", uselist=False)

    __table_args__ = (Index("ix_document_batch_id", "batch_id"),)


class Prediction(Base):
    __tablename__ = "prediction"
    id: Mapped[uuid.UUID] = mapped_column(GUID, primary_key=True, default=_uuid)
    document_id: Mapped[uuid.UUID] = mapped_column(
        GUID, ForeignKey("document.id", ondelete="CASCADE"), unique=True
    )
    label: Mapped[str] = mapped_column(String(64))
    confidence: Mapped[float] = mapped_column(Float)
    top5: Mapped[list] = mapped_column(JSON)
    overlay_key: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    model_sha256: Mapped[str] = mapped_column(String(64))
    latency_ms: Mapped[float] = mapped_column(Float)
    relabeled_to: Mapped[str | None] = mapped_column(String(64), nullable=True)
    relabeled_by: Mapped[uuid.UUID | None] = mapped_column(
        GUID, ForeignKey("user.id"), nullable=True
    )
    relabeled_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    document: Mapped[Document] = relationship(back_populates="prediction")

    __table_args__ = (Index("ix_prediction_created_at", "created_at"),)


class AuditLog(Base):
    """Append-only record of role changes, relabels and batch state changes."""

    __tablename__ = "audit_log"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    actor_id: Mapped[uuid.UUID | None] = mapped_column(GUID, ForeignKey("user.id"), nullable=True)
    actor: Mapped[str] = mapped_column(String(320))  # email, or "system:<service>"
    action: Mapped[str] = mapped_column(String(64))
    target: Mapped[str] = mapped_column(String(256))
    details: Mapped[dict] = mapped_column(JSON, default=dict)
    request_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), index=True
    )


class CasbinRule(Base):
    """Casbin policy storage (standard casbin_rule layout)."""

    __tablename__ = "casbin_rule"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ptype: Mapped[str] = mapped_column(String(8))
    v0: Mapped[str] = mapped_column(String(255))
    v1: Mapped[str] = mapped_column(String(255))
    v2: Mapped[str | None] = mapped_column(String(255), nullable=True)

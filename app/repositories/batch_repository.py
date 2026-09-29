"""Batch, document and prediction queries."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.db.models import Batch, Document, Prediction
from app.domain.enums import BatchStatus, DocumentStatus


async def create_batch(session: AsyncSession, source: str = "sftp") -> Batch:
    batch = Batch(source=source, status=BatchStatus.received, document_count=0)
    session.add(batch)
    await session.flush()
    return batch


async def add_document(
    session: AsyncSession, batch: Batch, filename: str, sha256: str, blob_key: str
) -> Document:
    doc = Document(
        batch_id=batch.id,
        filename=filename,
        sha256=sha256,
        blob_key=blob_key,
        status=DocumentStatus.queued,
    )
    session.add(doc)
    batch.document_count += 1
    await session.flush()
    return doc


async def list_batches(session: AsyncSession, limit: int, offset: int) -> list[Batch]:
    rows = await session.scalars(
        select(Batch).order_by(Batch.created_at.desc(), Batch.id).limit(limit).offset(offset)
    )
    return list(rows.all())


async def get_batch(session: AsyncSession, batch_id: uuid.UUID) -> Batch | None:
    return await session.scalar(
        select(Batch)
        .where(Batch.id == batch_id)
        .options(selectinload(Batch.documents).selectinload(Document.prediction))
    )


async def get_batch_for_update(session: AsyncSession, batch_id: uuid.UUID) -> Batch | None:
    return await session.scalar(select(Batch).where(Batch.id == batch_id).with_for_update())


async def get_document(session: AsyncSession, document_id: uuid.UUID) -> Document | None:
    return await session.scalar(
        select(Document)
        .where(Document.id == document_id)
        .options(selectinload(Document.prediction))
    )


async def document_status_counts(
    session: AsyncSession, batch_id: uuid.UUID
) -> dict[DocumentStatus, int]:
    rows = await session.execute(
        select(Document.status, func.count())
        .where(Document.batch_id == batch_id)
        .group_by(Document.status)
    )
    return {status: n for status, n in rows.all()}


async def save_prediction(session: AsyncSession, doc: Document, **fields) -> Prediction:
    pred = Prediction(document_id=doc.id, **fields)
    session.add(pred)
    doc.status = DocumentStatus.classified
    doc.error = None
    await session.flush()
    return pred


async def mark_document_failed(session: AsyncSession, doc: Document, error: str) -> None:
    doc.status = DocumentStatus.failed
    doc.error = error[:1000]
    await session.flush()


async def recent_predictions(session: AsyncSession, limit: int) -> list[Prediction]:
    rows = await session.scalars(
        select(Prediction).order_by(Prediction.created_at.desc(), Prediction.id).limit(limit)
    )
    return list(rows.all())


async def get_prediction(session: AsyncSession, prediction_id: uuid.UUID) -> Prediction | None:
    return await session.get(Prediction, prediction_id)


async def relabel(
    session: AsyncSession, pred: Prediction, label: str, by: uuid.UUID | None
) -> Prediction:
    pred.relabeled_to, pred.relabeled_by, pred.relabeled_at = label, by, datetime.now(UTC)
    await session.flush()
    return pred

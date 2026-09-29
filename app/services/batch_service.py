"""Batches: ingestion, prediction recording, state changes, and read views."""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession

from app.classifier.artifacts import REVIEW_CONFIDENCE_THRESHOLD
from app.domain.enums import BatchStatus, DocumentStatus
from app.domain.schemas import BatchDetail, BatchSummary, DocumentView, PredictionOut
from app.infra.blob import BlobStore
from app.infra.cache import CacheInvalidator, CacheNS
from app.infra.queue import JobQueue
from app.repositories import audit_repository, batch_repository
from app.services.errors import NotFoundError
from app.services.permission_service import Actor

INGEST_ACTOR = Actor(id=None, name="system:sftp-ingest")
WORKER_ACTOR = Actor(id=None, name="system:worker")


@dataclass(frozen=True)
class IncomingFile:
    filename: str
    data: bytes


def prediction_out(pred) -> PredictionOut:
    return PredictionOut(
        id=pred.id,
        document_id=pred.document_id,
        label=pred.label,
        confidence=pred.confidence,
        top5=[tuple(x) for x in pred.top5],
        relabeled_to=pred.relabeled_to,
        relabeled_by=pred.relabeled_by,
        relabeled_at=pred.relabeled_at,
        latency_ms=pred.latency_ms,
        created_at=pred.created_at,
        needs_review=pred.confidence < REVIEW_CONFIDENCE_THRESHOLD,
        effective_label=pred.relabeled_to or pred.label,
    )


def _document_view(doc) -> DocumentView:
    return DocumentView(
        id=doc.id,
        filename=doc.filename,
        status=doc.status.value,
        error=doc.error,
        prediction=prediction_out(doc.prediction) if doc.prediction else None,
    )


# ---- reads -------------------------------------------------------------------


async def list_batches(session: AsyncSession, limit: int, offset: int) -> list[BatchSummary]:
    return [
        BatchSummary.model_validate(b)
        for b in await batch_repository.list_batches(session, limit, offset)
    ]


async def get_batch(session: AsyncSession, batch_id: uuid.UUID) -> BatchDetail:
    batch = await batch_repository.get_batch(session, batch_id)
    if batch is None:
        raise NotFoundError(f"Batch {batch_id} not found")
    return BatchDetail(
        **BatchSummary.model_validate(batch).model_dump(),
        documents=[_document_view(d) for d in batch.documents],
    )


# ---- ingestion (sftp-ingest worker) -----------------------------------------


async def ingest(
    session: AsyncSession,
    invalidator: CacheInvalidator,
    blob: BlobStore,
    queue: JobQueue,
    files: list[IncomingFile],
    request_id: str | None,
):
    """Store the files, record one batch, then queue one classification job per document."""
    batch = await batch_repository.create_batch(session, source="sftp")
    docs = []
    for f in files:
        key = f"raw/{batch.id}/{uuid.uuid4().hex}_{f.filename}"
        blob.put(key, f.data, content_type="image/tiff")
        docs.append(
            await batch_repository.add_document(
                session, batch, f.filename, hashlib.sha256(f.data).hexdigest(), key
            )
        )
    await audit_repository.add(
        session,
        actor=INGEST_ACTOR.name,
        actor_id=None,
        action="batch.created",
        target=f"batch:{batch.id}",
        details={"documents": len(docs), "status": batch.status.value},
        request_id=request_id,
    )
    await session.commit()
    for doc in docs:
        queue.enqueue_classification(doc.id, request_id)
    await invalidator.ainvalidate(CacheNS.batches)
    return batch


# ---- inference results (inference worker) -----------------------------------


async def _set_batch_status(
    session: AsyncSession, batch, status: BatchStatus, request_id: str | None
) -> None:
    if batch.status == status:
        return
    old = batch.status
    batch.status = status
    await audit_repository.add(
        session,
        actor=WORKER_ACTOR.name,
        actor_id=None,
        action="batch.status_changed",
        target=f"batch:{batch.id}",
        details={"from": old.value, "to": status.value},
        request_id=request_id,
    )


async def _advance_batch(
    session: AsyncSession, batch_id: uuid.UUID, request_id: str | None
) -> None:
    batch = await batch_repository.get_batch_for_update(session, batch_id)
    if batch is None:
        raise NotFoundError(f"Batch {batch_id} not found")
    counts = await batch_repository.document_status_counts(session, batch_id)
    queued = counts.get(DocumentStatus.queued, 0)
    classified = counts.get(DocumentStatus.classified, 0)
    if batch.status == BatchStatus.received:
        await _set_batch_status(session, batch, BatchStatus.processing, request_id)
    if queued == 0:
        final = BatchStatus.done if classified > 0 else BatchStatus.failed
        await _set_batch_status(session, batch, final, request_id)


async def document_for_inference(session: AsyncSession, document_id: uuid.UUID):
    doc = await batch_repository.get_document(session, document_id)
    if doc is None:
        raise NotFoundError(f"Document {document_id} not found")
    return doc


async def record_prediction(
    session: AsyncSession,
    invalidator: CacheInvalidator,
    document_id: uuid.UUID,
    *,
    label: str,
    confidence: float,
    top5: list[tuple[str, float]],
    overlay_key: str | None,
    model_sha256: str,
    latency_ms: float,
    request_id: str | None,
):
    """Idempotent: a retried job for an already-classified document is a no-op."""
    doc = await document_for_inference(session, document_id)
    if doc.prediction is not None:
        return doc.prediction
    pred = await batch_repository.save_prediction(
        session,
        doc,
        label=label,
        confidence=confidence,
        top5=[list(t) for t in top5],
        overlay_key=overlay_key,
        model_sha256=model_sha256,
        latency_ms=latency_ms,
    )
    await _advance_batch(session, doc.batch_id, request_id)
    await session.commit()
    await invalidator.ainvalidate(CacheNS.batches, CacheNS.predictions)
    return pred


async def record_failure(
    session: AsyncSession,
    invalidator: CacheInvalidator,
    document_id: uuid.UUID,
    error: str,
    request_id: str | None,
) -> None:
    doc = await document_for_inference(session, document_id)
    if doc.status != DocumentStatus.queued:
        return
    await batch_repository.mark_document_failed(session, doc, error)
    await _advance_batch(session, doc.batch_id, request_id)
    await session.commit()
    await invalidator.ainvalidate(CacheNS.batches)

"""Predictions: recent list, reviewer relabeling, overlay links."""

from __future__ import annotations

import uuid
from datetime import timedelta

from sqlalchemy.ext.asyncio import AsyncSession

from app.classifier.artifacts import REVIEW_CONFIDENCE_THRESHOLD
from app.domain.schemas import OverlayLink, PredictionOut
from app.infra.blob import BlobStore
from app.infra.cache import CacheInvalidator, CacheNS
from app.repositories import audit_repository, batch_repository
from app.services.batch_service import prediction_out
from app.services.errors import ConflictError, NotFoundError
from app.services.permission_service import Actor

OVERLAY_LINK_TTL = timedelta(minutes=15)


async def recent(session: AsyncSession, limit: int) -> list[PredictionOut]:
    return [prediction_out(p) for p in await batch_repository.recent_predictions(session, limit)]


async def relabel(
    session: AsyncSession,
    invalidator: CacheInvalidator,
    actor: Actor,
    prediction_id: uuid.UUID,
    label: str,
    request_id: str | None,
) -> PredictionOut:
    pred = await batch_repository.get_prediction(session, prediction_id)
    if pred is None:
        raise NotFoundError(f"Prediction {prediction_id} not found")
    if pred.confidence >= REVIEW_CONFIDENCE_THRESHOLD:
        raise ConflictError(
            f"Only predictions with top-1 confidence below {REVIEW_CONFIDENCE_THRESHOLD} can be "
            f"relabeled (this one is {pred.confidence:.3f})"
        )
    before = pred.relabeled_to or pred.label
    await batch_repository.relabel(session, pred, label, actor.id)
    await audit_repository.add(
        session,
        actor=actor.name,
        actor_id=actor.id,
        action="prediction.relabeled",
        target=f"prediction:{pred.id}",
        details={
            "document_id": str(pred.document_id),
            "from": before,
            "to": label,
            "model_label": pred.label,
            "confidence": pred.confidence,
        },
        request_id=request_id,
    )
    await session.commit()
    await invalidator.ainvalidate(CacheNS.batches, CacheNS.predictions)
    return prediction_out(pred)


async def overlay_link(
    session: AsyncSession, blob: BlobStore, document_id: uuid.UUID
) -> OverlayLink:
    doc = await batch_repository.get_document(session, document_id)
    if doc is None or doc.prediction is None or not doc.prediction.overlay_key:
        raise NotFoundError(f"No overlay for document {document_id}")
    return OverlayLink(
        url=blob.url(doc.prediction.overlay_key, OVERLAY_LINK_TTL),
        expires_in_seconds=int(OVERLAY_LINK_TTL.total_seconds()),
    )

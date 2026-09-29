"""Inference worker: consumes classification jobs from Redis (RQ).

For each document: fetch the scan from MinIO, classify it, write an annotated
overlay PNG back to MinIO, and record the prediction through the service layer
(which advances the batch state, writes the audit log and invalidates caches).

The worker refuses to start unless Vault is reachable and the classifier
passes its artifact checks (weights present, SHA-256, quality gate).

Run:  python -m app.workers.inference
"""

from __future__ import annotations

import io
import sys
import time
import uuid

import structlog
from PIL import UnidentifiedImageError
from rq import SimpleWorker, get_current_job
from rq.queue import Queue

from app.classifier.model import DocumentClassifier, load_classifier, render_overlay
from app.classifier.preprocessing import load_image_bytes
from app.core.log import configure_logging, get_logger, request_id_var
from app.db.session import sessionmaker
from app.services import batch_service
from app.workers.common import WorkerContext, bootstrap

log = get_logger(__name__)

_ctx: WorkerContext | None = None
_classifier: DocumentClassifier | None = None


class PermanentJobError(Exception):
    """The input can never be classified (e.g. not an image); do not retry."""


async def classify(
    ctx: WorkerContext,
    classifier: DocumentClassifier,
    document_id: uuid.UUID,
    request_id: str | None,
) -> str:
    async with sessionmaker()() as session:
        doc = await batch_service.document_for_inference(session, document_id)
        if doc.prediction is not None:
            return doc.prediction.label  # already done (retry after a crash)
        blob_key, batch_id = doc.blob_key, doc.batch_id

    raw = ctx.blob.get(blob_key)
    try:
        page = load_image_bytes(raw)
    except (UnidentifiedImageError, OSError) as exc:
        raise PermanentJobError(f"Not a readable image: {exc}") from exc

    start = time.perf_counter()
    pred = classifier.predict_image(page)
    latency_ms = (time.perf_counter() - start) * 1000

    overlay = io.BytesIO()
    render_overlay(page, pred).save(overlay, format="PNG")
    overlay_key = f"overlays/{batch_id}/{document_id}.png"
    ctx.blob.put(overlay_key, overlay.getvalue(), content_type="image/png")

    async with sessionmaker()() as session:
        await batch_service.record_prediction(
            session,
            ctx.invalidator,
            document_id,
            label=pred.label,
            confidence=pred.confidence,
            top5=pred.top5,
            overlay_key=overlay_key,
            model_sha256=classifier.card["sha256"],
            latency_ms=latency_ms,
            request_id=request_id,
        )
    log.info(
        "document.classified",
        document_id=str(document_id),
        label=pred.label,
        confidence=round(pred.confidence, 4),
        latency_ms=round(latency_ms, 1),
    )
    return pred.label


async def fail(
    ctx: WorkerContext, document_id: uuid.UUID, error: str, request_id: str | None
) -> None:
    async with sessionmaker()() as session:
        await batch_service.record_failure(session, ctx.invalidator, document_id, error, request_id)


def _loaded() -> tuple[WorkerContext, DocumentClassifier]:
    if _ctx is None or _classifier is None:
        raise RuntimeError(
            "Worker not initialised; start it with `python -m app.workers.inference`"
        )
    return _ctx, _classifier


def classify_document_job(document_id: str) -> str | None:
    """RQ entrypoint. Runs in the worker process, so the model stays loaded."""
    ctx, classifier = _loaded()
    job = get_current_job()
    request_id = (job.meta or {}).get("request_id") if job else None
    token = request_id_var.set(request_id)
    structlog.contextvars.bind_contextvars(job_id=job.id if job else None)
    doc_id = uuid.UUID(document_id)
    try:
        return ctx.run(classify(ctx, classifier, doc_id, request_id))
    except PermanentJobError as exc:
        log.warning("document.unreadable", document_id=document_id, error=str(exc))
        ctx.run(fail(ctx, doc_id, str(exc), request_id))
        return None
    except Exception as exc:
        if job is None or not job.retries_left:
            log.exception("document.failed", document_id=document_id)
            ctx.run(fail(ctx, doc_id, f"{type(exc).__name__}: {exc}", request_id))
        raise
    finally:
        structlog.contextvars.unbind_contextvars("job_id")
        request_id_var.reset(token)


def main() -> int:
    global _ctx, _classifier
    configure_logging("worker")
    try:
        _ctx = bootstrap()
        _classifier = load_classifier()
    except Exception as exc:  # refuse to start
        log.error("worker.refused_to_start", error=str(exc), error_type=type(exc).__name__)
        return 1
    log.info(
        "worker.started",
        backbone=_classifier.card["backbone"],
        model_sha256=_classifier.card["sha256"][:12],
    )
    queue = Queue(_ctx.settings.queue_name, connection=_ctx.redis)
    # SimpleWorker runs jobs in this process, keeping the model and DB pool warm.
    SimpleWorker([queue], connection=_ctx.redis).work(with_scheduler=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

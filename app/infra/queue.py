"""Redis-backed job queue (RQ). The request id travels with each job."""

from __future__ import annotations

import uuid
from typing import Protocol

from redis import Redis
from rq import Queue, Retry

CLASSIFY_JOB = "app.workers.inference.classify_document_job"


class JobQueue(Protocol):
    def enqueue_classification(self, document_id: uuid.UUID, request_id: str | None) -> str: ...


class RQJobQueue:
    def __init__(self, redis: Redis, queue_name: str):
        self._queue = Queue(queue_name, connection=redis)

    def enqueue_classification(self, document_id: uuid.UUID, request_id: str | None) -> str:
        job = self._queue.enqueue(
            CLASSIFY_JOB,
            str(document_id),
            meta={"request_id": request_id},
            job_timeout=120,
            retry=Retry(max=2, interval=[5, 30]),
        )
        return job.id

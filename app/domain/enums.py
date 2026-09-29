"""Enumerations shared by the domain, services and ORM layers."""

from __future__ import annotations

import enum


class BatchStatus(enum.StrEnum):
    received = "received"
    processing = "processing"
    done = "done"
    failed = "failed"


class DocumentStatus(enum.StrEnum):
    queued = "queued"
    classified = "classified"
    failed = "failed"

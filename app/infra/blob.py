"""MinIO (S3-compatible) blob storage adapter."""

from __future__ import annotations

import io
from datetime import timedelta
from typing import Protocol

from minio import Minio
from minio.error import S3Error


class BlobStore(Protocol):
    def put(self, key: str, data: bytes, content_type: str) -> None: ...
    def get(self, key: str) -> bytes: ...
    def url(self, key: str, expires: timedelta = timedelta(minutes=15)) -> str: ...


class MinioBlobStore:
    def __init__(
        self, endpoint: str, access_key: str, secret_key: str, bucket: str, secure: bool = False
    ):
        self._client = Minio(endpoint, access_key=access_key, secret_key=secret_key, secure=secure)
        self._bucket = bucket

    def ensure_bucket(self) -> None:
        if not self._client.bucket_exists(self._bucket):
            try:
                self._client.make_bucket(self._bucket)
            except S3Error as exc:  # another process created it first
                if exc.code not in ("BucketAlreadyOwnedByYou", "BucketAlreadyExists"):
                    raise

    def put(self, key: str, data: bytes, content_type: str) -> None:
        self._client.put_object(
            self._bucket, key, io.BytesIO(data), length=len(data), content_type=content_type
        )

    def get(self, key: str) -> bytes:
        resp = self._client.get_object(self._bucket, key)
        try:
            return resp.read()
        finally:
            resp.close()
            resp.release_conn()

    def url(self, key: str, expires: timedelta = timedelta(minutes=15)) -> str:
        return self._client.presigned_get_object(self._bucket, key, expires=expires)

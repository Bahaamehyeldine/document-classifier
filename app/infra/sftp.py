"""SFTP adapter for the scanner drop folder (paramiko)."""

from __future__ import annotations

import posixpath
import stat
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Protocol

import paramiko

CLAIM_SUFFIX = ".processing"


class DropFolder(Protocol):
    def list_new(self) -> list[str]: ...
    def list_claimed(self) -> list[str]: ...
    def claim(self, name: str) -> str: ...
    def read(self, name: str) -> bytes: ...
    def remove(self, name: str) -> None: ...


class SftpDropFolder:
    """Files are claimed by renaming to `<name>.processing` before ingestion, so a
    crash never loses a file and a restart resumes claimed files."""

    def __init__(self, sftp: paramiko.SFTPClient, directory: str):
        self._sftp = sftp
        self._dir = directory

    def _entries(self):
        return [e for e in self._sftp.listdir_attr(self._dir) if stat.S_ISREG(e.st_mode or 0)]

    def list_new(self) -> list[str]:
        return sorted(
            e.filename
            for e in self._entries()
            if not e.filename.endswith(CLAIM_SUFFIX) and not e.filename.startswith(".")
        )

    def list_claimed(self) -> list[str]:
        return sorted(e.filename for e in self._entries() if e.filename.endswith(CLAIM_SUFFIX))

    def claim(self, name: str) -> str:
        claimed = name + CLAIM_SUFFIX
        self._sftp.rename(posixpath.join(self._dir, name), posixpath.join(self._dir, claimed))
        return claimed

    def read(self, name: str) -> bytes:
        with self._sftp.open(posixpath.join(self._dir, name), "rb") as fh:
            return fh.read()

    def remove(self, name: str) -> None:
        self._sftp.remove(posixpath.join(self._dir, name))


@contextmanager
def sftp_connection(
    host: str, port: int, user: str, credential: str
) -> Iterator[paramiko.SFTPClient]:
    transport = paramiko.Transport((host, port))
    try:
        transport.connect(username=user, password=credential)
        client = paramiko.SFTPClient.from_transport(transport)
        if client is None:
            raise ConnectionError(f"Could not open an SFTP session on {host}:{port}")
        yield client
    finally:
        transport.close()

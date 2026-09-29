"""SFTP ingest worker: polls the scanner drop folder and turns files into batches.

Each poll:
  1. resumes files claimed before a crash (`*.processing`),
  2. claims new files whose size has been stable for one poll (so half-uploaded
     files are never read),
  3. stores them in MinIO, records one batch with its documents and an audit
     entry, and queues one classification job per document,
  4. deletes the claimed files from the drop folder.

Run:  python -m app.workers.sftp_ingest
"""

from __future__ import annotations

import sys
import time

import structlog

from app.core.log import configure_logging, get_logger, new_request_id, request_id_var
from app.db.session import sessionmaker
from app.infra.queue import RQJobQueue
from app.infra.sftp import CLAIM_SUFFIX, DropFolder, SftpDropFolder, sftp_connection
from app.services import batch_service
from app.services.batch_service import IncomingFile
from app.workers.common import WorkerContext, bootstrap

log = get_logger(__name__)

MAX_FILES_PER_BATCH = 200


class Poller:
    def __init__(self, ctx: WorkerContext, queue):
        self.ctx = ctx
        self.queue = queue
        self._sizes: dict[str, int] = {}

    def _stable_new_files(self, folder: DropFolder, sizes: dict[str, int]) -> list[str]:
        ready = [
            n for n in folder.list_new() if n in self._sizes and self._sizes[n] == sizes.get(n)
        ]
        self._sizes = sizes
        return ready[:MAX_FILES_PER_BATCH]

    def poll_once(self, folder: DropFolder, sizes: dict[str, int]) -> int:
        claimed = folder.list_claimed()
        for name in self._stable_new_files(folder, sizes):
            claimed.append(folder.claim(name))
        if not claimed:
            return 0

        rid = new_request_id()
        token = request_id_var.set(rid)
        try:
            files = [
                IncomingFile(filename=c.removesuffix(CLAIM_SUFFIX), data=folder.read(c))
                for c in claimed
            ]

            async def _ingest():
                async with sessionmaker()() as session:
                    return await batch_service.ingest(
                        session, self.ctx.invalidator, self.ctx.blob, self.queue, files, rid
                    )

            batch = self.ctx.run(_ingest())
            for c in claimed:
                folder.remove(c)
            log.info("batch.ingested", batch_id=str(batch.id), documents=len(files))
            return len(files)
        finally:
            request_id_var.reset(token)


def _sizes(sftp, directory: str) -> dict[str, int]:
    return {e.filename: e.st_size for e in sftp.listdir_attr(directory)}


def main() -> int:
    configure_logging("sftp-ingest")
    try:
        ctx = bootstrap()
    except Exception as exc:
        log.error("sftp_ingest.refused_to_start", error=str(exc), error_type=type(exc).__name__)
        return 1
    s = ctx.settings
    poller = Poller(ctx, RQJobQueue(ctx.redis, s.queue_name))
    structlog.contextvars.bind_contextvars(sftp_host=s.sftp_host)
    log.info("sftp_ingest.started", poll_seconds=s.sftp_poll_seconds)
    backoff = 1.0
    while True:
        try:
            with sftp_connection(
                s.sftp_host, s.sftp_port, s.sftp_user, ctx.secrets.sftp_password
            ) as sftp:
                folder = SftpDropFolder(sftp, s.sftp_dir)
                backoff = 1.0
                while True:
                    poller.poll_once(folder, _sizes(sftp, s.sftp_dir))
                    time.sleep(s.sftp_poll_seconds)
        except KeyboardInterrupt:
            return 0
        except Exception as exc:
            log.warning("sftp_ingest.connection_error", error=str(exc), retry_in_s=backoff)
            time.sleep(backoff)
            backoff = min(backoff * 2, 30.0)


if __name__ == "__main__":
    sys.exit(main())

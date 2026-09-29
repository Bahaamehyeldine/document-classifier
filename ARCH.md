# Architecture

## Services (docker compose)

| Service | Built from | Role |
|---|---|---|
| `api` | this repo | FastAPI app: auth, permissions, reads, relabeling. Never runs inference. |
| `worker` | this repo | RQ worker: loads the model once, classifies queued documents |
| `sftp-ingest` | this repo | Polls the scanner drop folder, stores files, records batches, queues jobs |
| `migrate` | this repo | `alembic upgrade head`, then exits; everything else waits for it |
| `vault-init` | hashicorp/vault | Seeds dev secrets into Vault, then exits |
| `db` | postgres:16 | Users, invitations, batches, documents, predictions, audit log, Casbin policy |
| `redis` | redis:7 | Job queue (RQ) and response cache |
| `minio` | Chainguard MinIO | Raw scans and annotated overlays |
| `sftp` | atmoz/sftp | Scanner drop folder |
| `vault` | hashicorp/vault (dev) | Secrets |

## Request and job flow

```
scanner ──SFTP──▶ sftp-ingest ──▶ MinIO raw/<batch>/…      (1) store
                      │
                      ├──▶ Postgres: batch + documents + audit "batch.created"   (2) record, one transaction
                      └──▶ Redis queue: one job per document, meta.request_id    (3) enqueue after commit

worker ◀── job ── Redis
  ├── MinIO: read raw scan, write overlays/<batch>/<doc>.png
  ├── Postgres: prediction, document status, batch state + audit entries
  └── Redis cache: invalidate batches and predictions namespaces

users ──JWT──▶ api ──▶ services ──▶ repositories ──▶ Postgres
                 │          └── cache invalidation on writes
                 └── cached GET responses in Redis
```

A document is queued only after its batch is committed, so a job never references a missing row. Jobs are idempotent: a retried job for an already-classified document is a no-op.

## Layering (enforced)

| Layer | Owns | Must not |
|---|---|---|
| `app/api/` | HTTP routing, request/response models, cache keys | import SQLAlchemy, Redis, ORM models, repositories or infra |
| `app/dependencies.py` | composition root: builds the per-request `ServiceContext` (session, cache invalidator, blob store, request id) and the `require()` permission check | contain business rules |
| `app/auth.py` | fastapi-users wiring: user manager, JWT strategy (key from Vault) | contain business rules |
| `app/services/` | business rules, transaction boundaries, audit entries, cache invalidation | build SQL |
| `app/repositories/` | SQL | raise HTTP errors, invalidate caches |
| `app/domain/` | Pydantic schemas, enums, the `Principal` protocol | depend on ORM models |
| `app/infra/` | adapters: Vault, MinIO, Redis queue and cache, SFTP | contain business rules |
| `app/db/models.py` | SQLAlchemy ORM models | be imported outside repositories (and Alembic) |
| `app/classifier/` | artifact checks, preprocessing, inference | know about HTTP or the database |
| `app/workers/` | process entrypoints; call services, never repositories | contain business rules |

These rules are enforced by `tests/unit/test_architecture.py` (it parses every module's imports), so a violation fails CI rather than waiting for review.

**Adding an endpoint:** add a schema in `domain/`, a query in a repository, a function in a service (commit, audit, invalidate there), and a thin router function taking `ctx: ServiceContext = Depends(service_context)` and `Depends(require("<obj>", "<act>"))`. If it needs a new permission, add a migration that inserts the `p` rule.

## Permissions

Casbin RBAC model (`app/services/permission_service.py`): `p` rules map a role to `(object, action)`, `g` rules map a user id to a role. Rules are loaded from `casbin_rule` on every check (one small query), which is what makes a role change effective on the user's next request. The migration seeds the `p` rules; `g` rules are written by invitations, role changes and the `create-admin` CLI.

## Caching

fastapi-cache2 on Redis, keys `docclassifier:<namespace>:<path+query>` (or `:<user id>` for `/me`). `CacheInvalidator` deletes a namespace by pattern and is called from services only:

| Write | Invalidates |
|---|---|
| role change, invitation | `me` |
| new batch | `batches` |
| prediction recorded / document failed | `batches`, `predictions` |
| relabel | `batches`, `predictions` |

## Startup checks

| Check | api | worker | sftp-ingest |
|---|---|---|---|
| Vault reachable, all secrets present | ✓ | ✓ | ✓ |
| Model weights present, SHA-256 matches card, test top-1 ≥ gate | ✓ (no torch load) | ✓ (then loads) | |
| Casbin policy table non-empty | ✓ | | |

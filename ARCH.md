# Architecture

## Target system (Week 6 brief)

```
scanner ──SFTP──▶ sftp-ingest ──▶ MinIO (raw TIFF) ──▶ Redis queue ──▶ worker ──▶ Postgres (predictions)
                                                                         │
                                                                         └──▶ MinIO (annotated overlay PNG)
users ──JWT──▶ api (FastAPI + Casbin roles) ──▶ services ──▶ repositories ──▶ Postgres
                         │                         └── cache invalidation (Redis)
                         └── secrets from Vault at startup
```

The API never runs inference; only the worker loads the model. Training happens on Colab, never in the compose stack.

## Layering (enforced, no exceptions)

| Layer | Owns | Must not |
|---|---|---|
| `app/api/` | HTTP routing, request/response models | touch SQLAlchemy, cache or external systems |
| `app/services/` | business logic, transactions, cache invalidation | build SQL |
| `app/repositories/` | SQL | raise HTTP errors, invalidate caches |
| `app/domain/` | Pydantic domain models | depend on ORM models |
| `app/infra/` | adapters: MinIO, Redis queue, SFTP, Vault, cache | contain business rules |
| `app/db/models.py` | SQLAlchemy ORM models | be imported outside repositories |
| `app/classifier/` | model loading, preprocessing, inference | know about HTTP or the database |

`GET /health/db` is the reference vertical slice: router → `user_service` → `user_repository` → Postgres.

## Classifier

See the *Classifier* section of the README. Startup checks live in `app/classifier/model.py::load_classifier`; the golden-set replay lives in `app/classifier/eval/golden.py`.

## Status

| Component | State |
|---|---|
| Compose infrastructure, migrations, layered slice | done |
| Classifier code, training notebook, golden test | done; weights pending a Colab run |
| Auth (fastapi-users JWT) + Casbin roles + audit log | not started |
| sftp-ingest and inference workers | not started |
| fastapi-cache2 caching, latency budgets | not started |
| CI compose smoke test | not started |

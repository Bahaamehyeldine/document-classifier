# Runbook

## Start, stop, reset
```bash
cp .env.example .env
docker compose up -d --build     # vault-init and migrate run first and exit 0
docker compose ps                # api healthy; worker, sftp-ingest running
docker compose down              # stop
docker compose down -v           # stop and delete Postgres, MinIO and SFTP data
```
Dev Vault keeps secrets in memory: after a Vault restart, run `docker compose up -d vault-init` (a new JWT key is generated, so existing tokens stop working).

## First admin
```bash
docker compose run --rm migrate python -m app.cli create-admin --email you@example.com
```
Then invite others: `POST /users/invitations {"email": ..., "role": "reviewer"}`.

## A service won't start
Every service logs one JSON line explaining why (`docker compose logs <service>`):

| Log | Cause | Fix |
|---|---|---|
| `Cannot read secrets from Vault` | Vault down or not seeded | `docker compose up -d vault vault-init` |
| `Classifier weights missing` | model not trained yet | run the Colab notebook, unzip artifacts, `git lfs pull` on a fresh clone |
| `Weights SHA-256 mismatch` | weights and model card from different runs | re-extract both from the same `classifier_artifacts.zip` |
| `below the committed threshold` | model under the README quality gate | retrain; do not lower the gate to pass |
| `Casbin policy table is empty` | migrations not applied | `docker compose run --rm migrate` |

## A document is stuck
```bash
docker compose exec db psql -U docclassifier -c \
  "select d.filename, d.status, d.error, b.status from document d join batch b on b.id = d.batch_id order by d.created_at desc limit 10"
docker compose exec redis redis-cli LLEN rq:queue:classify          # waiting jobs
docker compose logs worker | grep <request_id>                        # follow one batch end to end
```
- `queued` with jobs waiting and no worker log: the worker is down (see above).
- `failed` with `Not a readable image`: the scanner sent something that isn't an image; re-send it.
- Files ending in `.processing` in the SFTP folder are resumed automatically on the next poll.

## Tracing a request
Every api response has an `X-Request-ID` header; the same id appears in the api log line, in the queued job's metadata, in the worker's logs for that job, and in the audit log entries it produced.

## Golden-set test fails in CI
Something changed what the model outputs: the weights, `preprocessing.py`, or torch / torchvision / Pillow versions. If intentional, retrain and commit the new weights, model card and golden set together.

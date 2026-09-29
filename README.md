# Document Classifier

> **Status: in progress.** Infrastructure, database layer, layered API and the classifier code (training notebook, inference, golden-set test) are in place. Trained weights, the ingestion workers, auth/permissions and CI are next.

A backend service for ingesting and classifying documents. Documents will arrive by upload or SFTP drop, be stored in object storage, and be processed asynchronously. Pages are classified by **visual layout** (no OCR) into the 16 RVL-CDIP classes with a fine-tuned ConvNeXt.

---

## What exists today

- **FastAPI application** with a layered architecture: `api → services → repositories → db`, proven end to end by a vertical slice (`GET /health/db`)
- **PostgreSQL 16** with **SQLAlchemy 2.0** and **Alembic** migrations, including an initial `user` table (via `fastapi-users`)
- **Docker Compose infrastructure**, with health checks on each service:

| Service | Image | Purpose |
|---|---|---|
| `db` | postgres:16 | Primary relational store |
| `redis` | redis:7 | Job queue / cache (planned) |
| `minio` | minio/minio | S3-compatible document storage |
| `sftp` | atmoz/sftp | Inbound document drop (e.g. from scanners) |
| `vault` | hashicorp/vault | Secrets management (dev mode) |
| `migrate` | built from `Dockerfile` | Runs `alembic upgrade head` once the database is healthy |

## Classifier

| | |
|---|---|
| Dataset | RVL-CDIP: 16 layout classes, 320k train / 40k val / 40k test grayscale scans (academic use only) |
| Model | torchvision ConvNeXt Tiny (or Small), ImageNet-pretrained, full fine-tune, 16-way head |
| Input | first page, grayscale → RGB, 224×224, ImageNet normalization (`app/classifier/preprocessing.py`) |
| Training | Colab GPU via [`notebooks/train_rvl_cdip.ipynb`](notebooks/train_rvl_cdip.ipynb); the local stack never trains |
| Artifacts | `app/classifier/models/classifier.pt` (git LFS), `model_card.json`, 50-image golden set in `app/classifier/eval/` |

**Model quality gate: test top-1 ≥ 0.85.** The worker refuses to start if the weights are missing, their SHA-256 does not match the model card, or the model card's full-test top-1 is below this threshold.

**Golden-set replay:** `pytest app/classifier/eval/golden.py` re-runs the 50 golden images and requires identical labels and top-1 confidence within 1e-6 of the values recorded at training time.

Predictions with top-1 confidence below 0.7 are flagged for reviewer relabeling.

### Train the model
1. Open `notebooks/train_rvl_cdip.ipynb` in Colab with a GPU runtime and run all cells (about 1 h of download plus about 1–1.5 h per epoch on a T4).
2. Unzip the downloaded `classifier_artifacts.zip` at the repo root.
3. Track the weights with LFS: `git lfs install && git lfs track "app/classifier/models/*.pt"`
4. Verify: `pytest app/classifier/eval/golden.py tests/unit`

## Project structure

```
app/
├── api/            # HTTP routes (FastAPI routers)
├── services/       # Business logic
├── repositories/   # Data access (SQLAlchemy queries)
├── db/             # Engine, session, ORM models
├── domain/         # Pydantic request/response schemas
├── classifier/     # Model loading, preprocessing, inference, golden-set test
├── infra/          # External-service clients (planned)
└── main.py         # FastAPI app entry point
alembic/            # Database migrations
notebooks/          # Colab training notebook
tests/unit/         # Unit tests
docker-compose.yml  # Local infrastructure
Dockerfile          # Image used by the migrate service
```

## Getting started

**Prerequisites:** Docker with Compose, Python 3.11

```bash
git clone https://github.com/Bahaamehyeldine/document-classifier.git
cd document-classifier
cp .env.example .env

# Start the infrastructure and apply migrations
docker compose up -d

# Run the API locally
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export DATABASE_URL=postgresql+psycopg://docclassifier:docclassifier_dev_pw@localhost:5432/docclassifier
uvicorn app.main:app --reload
```

Then check the database connection:

```bash
curl http://localhost:8000/health/db
# {"count": 0}
```

Interactive API docs are at http://localhost:8000/docs.

> The credentials in `docker-compose.yml` are for local development only.

## Roadmap

- [x] Project scaffold and layered architecture
- [x] Infrastructure containers with health checks
- [x] Database models and migrations
- [ ] Document ingestion (upload API and SFTP watcher) into MinIO
- [ ] Inference worker (RQ) writing predictions and annotated overlays to MinIO
- [x] Classifier code: training notebook, inference, refuse-to-start checks, golden-set test
- [ ] Train on Colab and commit weights, model card and golden set
- [ ] Authentication with `fastapi-users` (JWT) and Casbin roles (admin / reviewer / auditor)
- [ ] Secrets loaded from Vault
- [ ] CI: lint, type-check, golden-set test, compose smoke test

## Author

**Bahaa Mehye Eddin** · [GitHub](https://github.com/Bahaamehyeldine)

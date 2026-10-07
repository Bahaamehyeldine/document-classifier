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

## Train the model on Colab (interruption-proof)

Use `notebooks/train_rvl_cdip.ipynb` with a GPU runtime. Colab wipes its local disk on every disconnect, so the notebook keeps everything slow on **Google Drive** and every step resumes:

| Step | What is stored on Drive | After an interruption |
|---|---|---|
| 2. `scripts.rvl_cache build` | 224 px pages in 10k-page shards (`cache/`), resized by the service's own preprocessing | re-run the cell: it re-streams the archive but only decodes pages not yet cached |
| 3. `scripts.audit_dataset --cache-dir` | duplicate / leakage report (`audit/`), from hashes recorded in step 2 | seconds, just re-run |
| 4. `scripts.train_local --cache-dir` | checkpoint every 300 steps and per epoch (`run/`) | re-run the cell: resumes mid-epoch |

- The cache is the one-time cost (~1 h). After that a fresh runtime needs only a few minutes to unpack it, and epochs read 224 px pages instead of decoding ~1000 px TIFFs on Colab's 2 CPU cores.
- **Drive space:** the full cache is roughly 10-14 GB (an estimate: run `python -m scripts.rvl_cache info --out DIR` and `du -sh` for the real size), which is tight on a free 15 GB account. `TRAIN_PER_CLASS = 6000` in the notebook's config caches 96k training pages instead (validation and test stay complete), for roughly 4-6 GB and a ~3x faster epoch. Changing it requires a new cache directory.
- **Test numbers:** the model card reports the official test score (this is the quality gate, comparable to published RVL-CDIP work) and a leakage-controlled score that drops test pages with a pixel-identical twin in train or validation. The splits themselves are never modified.
- **Golden set:** must be real full-size TIFFs, so it is drawn from the cache's "golden pool" (the original bytes of 1 in 20 test pages), restricted to leakage-controlled pages when an audit exists.
- Colab's T4 has no fast bfloat16, so the script automatically trains in float16 with loss scaling there (and in bfloat16 on Ampere or newer).

## Train the model on a local GPU

Tested path: Windows + WSL2 (Ubuntu) + NVIDIA laptop GPU. The Windows NVIDIA driver is enough; do not install a driver inside WSL. **Needs enough RAM for WSL** (see the last bullet below); on a 16 GB laptop with Docker Desktop also running, prefer Colab.

```bash
cd ~/projects/document-classifier
nvidia-smi                                   # must list the GPU

# 1. Training environment (once)
python3 -m venv .venv-train && source .venv-train/bin/activate
pip install --upgrade pip
pip install torch==2.14.0 torchvision==0.29.0 --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements-train.txt
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# 2. Dataset (~39 GB download, resumable: re-run if interrupted)
scripts/download_rvl_cdip.sh                 # into ~/data/rvl-cdip

# 3. Five-minute dry run on a small subset (artifacts go to ~/data/rvl-cdip-run/smoke)
python -m scripts.train_local --smoke

# 4. Full run: 2 epochs, full test evaluation, golden set, model card
python -m scripts.train_local                # re-run the same command to resume after an interruption

# 5. Verify and commit
pytest app/classifier/eval/golden.py tests/unit
sudo apt install -y git-lfs && git lfs install
git add app/classifier/models app/classifier/eval
git commit -m "feat(classifier): trained ConvNeXt Tiny weights, model card and golden set"
git push
```

- **Interrupted run:** re-run the same command. A checkpoint is saved every 300 steps (a few minutes) and at each epoch end, so little is lost.
- **Run stalls, then dies (system RAM):** WSL gets only half of the machine's RAM by default, and each data-loading worker costs several hundred MB. The script prints total and available RAM at startup. Lower `--workers` (default 6), close heavy Windows apps, or raise WSL's limit: put `[wsl2]`, `memory=11GB` and `swap=8GB` in `%UserProfile%\.wslconfig`, then run `wsl --shutdown`.
- **CUDA out of memory (GPU):** lower `--batch-size` (e.g. 64) and delete the run folder, since a checkpoint only resumes with the same settings.
- **Changed options mid-run:** delete `~/data/rvl-cdip-run/<backbone>` to start over.
- **Dataset download refused (401/403):** create a Hugging Face token and run `HF_TOKEN=<token> scripts/download_rvl_cdip.sh`.
- The pinned `torch==2.14.0` matters: CI replays the golden set on CPU with the same version, and the recorded confidences must match within 1e-6.

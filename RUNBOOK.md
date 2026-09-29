# Runbook

## Start the stack
```bash
cp .env.example .env
docker compose up -d          # db, redis, minio, sftp, vault; migrate runs and exits
docker compose ps             # every service healthy, migrate exited 0
curl localhost:8000/health/db # when the API is running
```

## Classifier will not start
`ClassifierStartupError` names the cause:

| Message | Fix |
|---|---|
| `Classifier weights missing` | Run the Colab notebook and unzip `classifier_artifacts.zip` at the repo root; `git lfs pull` on a fresh clone. |
| `Weights SHA-256 mismatch` | The weights and model card come from different runs. Re-extract both from the same zip. |
| `below the committed threshold` | The trained model is below the README quality gate. Retrain; do not lower the gate to pass. |
| `class list does not match` | The model card was edited or produced by a different label order. Retrain. |

## Golden-set test fails in CI
Something changed what the model outputs: weights, `preprocessing.py`, or torch/torchvision/Pillow versions.
If the change is intentional, retrain and commit the new weights, model card and golden set together.

## Reset local data
```bash
docker compose down -v   # removes the Postgres, MinIO and SFTP volumes
```

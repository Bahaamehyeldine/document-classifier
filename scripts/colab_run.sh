#!/usr/bin/env bash
# Run or resume the whole Colab training pipeline.
#
# Every step checks whether its result already exists and skips itself if so.
# After a disconnect you therefore run the same command again and it continues
# from the first unfinished step:
#
#   runtime still alive  -> setup and dataset are skipped, training resumes in seconds
#   runtime was deleted  -> Colab wiped its disk, so the environment and dataset are
#                           restored automatically, then training resumes from the
#                           last checkpoint on Google Drive
#
# Called by notebooks/train_rvl_cdip.ipynb. Settings come from the environment.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-/content/rvl-cdip}"
DRIVE_DIR="${DRIVE_DIR:-/content/drive/MyDrive/document-classifier}"
WORK_DIR="${WORK_DIR:-$DRIVE_DIR/run}"
STATE_DIR="${STATE_DIR:-/content}"          # markers for work that dies with the runtime
BACKBONE="${BACKBONE:-convnext_tiny}"
EPOCHS="${EPOCHS:-2}"
BATCH_SIZE="${BATCH_SIZE:-96}"
TRAIN_SUBSET="${TRAIN_SUBSET:-}"
DEVICE="${DEVICE:-cuda}"
HF_BASE="${HF_BASE:-https://huggingface.co/datasets/aharley/rvl_cdip/resolve/main/data}"
ARTIFACTS_ZIP="$DRIVE_DIR/classifier_artifacts.zip"

step() { printf '\n==> %s\n' "$*"; }
skip() { printf '    [already done] %s\n' "$*"; }
fail() { printf '\nERROR: %s\n' "$*" >&2; exit 1; }

cd "$REPO_DIR"

# ---- 0. Nothing to do? ------------------------------------------------------------
if [[ -f "$ARTIFACTS_ZIP" && -z "${FORCE:-}" ]]; then
  step "Training already finished"
  echo "    Artifacts: $ARTIFACTS_ZIP"
  echo "    To train again from scratch, delete $DRIVE_DIR in Google Drive and re-run."
  exit 0
fi

# ---- 1. Runtime ---------------------------------------------------------------------
step "1/6 Runtime"
if [[ "$DEVICE" == "cuda" ]]; then
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null \
    || fail "No GPU. Use Runtime > Change runtime type > T4 GPU, then run the cell again."
fi
if [[ "$DRIVE_DIR" == /content/drive/* && ! -d /content/drive/MyDrive ]]; then
  fail "Google Drive is not mounted. Run the notebook cell (it mounts Drive) rather than this script."
fi
mkdir -p "$WORK_DIR"
if [[ -f "$WORK_DIR/$BACKBONE/last.pt" ]]; then
  echo "    Found a checkpoint in $WORK_DIR/$BACKBONE: training will resume from it."
else
  echo "    No checkpoint yet: this is a fresh run."
fi

# ---- 2. Pinned training environment (same versions the service and CI use) -----------
step "2/6 Python environment"
req_hash="$(cat requirements.txt requirements-train.txt | sha256sum | cut -c1-12)"
deps_marker="$STATE_DIR/.deps-$req_hash"
if [[ -f "$deps_marker" ]]; then
  skip "pinned requirements installed"
else
  pip install -q -r requirements-train.txt pytest
  touch "$deps_marker"
fi
python -c "import torch, torchvision; print('    torch', torch.__version__, '| torchvision', torchvision.__version__, '| cuda', torch.cuda.is_available())"

# ---- 3. Dataset ----------------------------------------------------------------------
step "3/6 RVL-CDIP dataset"
if [[ -f "$DATA_ROOT/.extracted" ]]; then
  skip "dataset extracted in $DATA_ROOT"
else
  mkdir -p "$DATA_ROOT/labels"
  for split in train val test; do
    curl -L --fail -sS --retry 5 "$HF_BASE/$split.txt" -o "$DATA_ROOT/labels/$split.txt"
  done
  echo "    Streaming the ~36 GB archive straight into tar (30-40 min); the tarball is never stored."
  used_kb() { df -k --output=used "$DATA_ROOT" | tail -1; }
  start_kb="$(used_kb)"
  ( while sleep 120; do
      printf '    ... about %d of ~37 GB extracted\n' $(( ($(used_kb) - start_kb) / 1048576 ))
    done ) &
  monitor=$!
  trap 'kill "$monitor" 2>/dev/null || true' EXIT
  curl -L --fail -sS --retry 5 "$HF_BASE/rvl-cdip.tar.gz" | tar -xz -C "$DATA_ROOT"
  kill "$monitor" 2>/dev/null || true
  trap - EXIT
  touch "$DATA_ROOT/.extracted"
fi
wc -l "$DATA_ROOT"/labels/*.txt | sed 's/^/    /'

# ---- 4. Train, evaluate, pick the golden set, write the model card --------------------
# scripts/train.py resumes from $WORK_DIR by itself, mid-epoch if needed.
step "4/6 Training"
train_args=(--data-root "$DATA_ROOT" --work-dir "$WORK_DIR" --backbone "$BACKBONE"
  --epochs "$EPOCHS" --batch-size "$BATCH_SIZE" --device "$DEVICE")
if [[ -n "$TRAIN_SUBSET" ]]; then train_args+=(--train-subset "$TRAIN_SUBSET"); fi
# shellcheck disable=SC2086  # EXTRA_ARGS is intentionally word-split
python -m scripts.train "${train_args[@]}" ${EXTRA_ARGS:-}

# ---- 5. Verify: the golden replay and artifact checks that CI runs --------------------
step "5/6 Verify artifacts"
python -m pytest -q -x --tb=short app/classifier/eval/golden.py \
  || fail "Artifacts failed verification (see above), so nothing was published to Drive."

# ---- 6. Publish to Drive --------------------------------------------------------------
step "6/6 Save artifacts to Google Drive"
tmp_zip="$(mktemp -u "$STATE_DIR/artifacts-XXXXXX.zip")"
zip -qr "$tmp_zip" \
  app/classifier/models/classifier.pt app/classifier/models/model_card.json \
  app/classifier/eval/golden_images app/classifier/eval/golden_expected.json
cp "$tmp_zip" "$ARTIFACTS_ZIP.part" && mv "$ARTIFACTS_ZIP.part" "$ARTIFACTS_ZIP" && rm -f "$tmp_zip"
ls -lh "$ARTIFACTS_ZIP" | sed 's/^/    /'
python - <<'PY'
import json
card = json.load(open("app/classifier/models/model_card.json"))
print("    test:", card["metrics"]["test"])
print("    sha256:", card["sha256"])
print("    trained on:", card["environment"]["gpu"])
PY
echo
echo "Done. Unzip classifier_artifacts.zip at the repo root and commit (weights go through git LFS)."

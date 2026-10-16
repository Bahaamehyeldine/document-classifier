#!/usr/bin/env bash
# Controlled comparison of freeze policies and backbones, scored on VALIDATION data only.
#
# Reuses the 224 px cache the main run built on Drive, trains each configuration on a
# subset, and writes one experiment.json per run. Re-running skips finished experiments, so
# it is safe after a Colab disconnect. The test split is never read (train.py --experiment
# returns before it), so nothing here can leak test data into a modelling decision.
#
#   bash scripts/run_experiments.sh        # about 40-60 min on a T4
#
# Settings come from the environment: DRIVE_DIR, TRAIN_SUBSET, VAL_SUBSET, EPOCHS, BATCH_SIZE.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DRIVE_DIR="${DRIVE_DIR:-/content/drive/MyDrive/document-classifier}"
CACHE_DIR="${CACHE_DIR:-$DRIVE_DIR/cache}"
LOCAL_CACHE="${LOCAL_CACHE:-/content/cache-local}"
WORK_DIR="${WORK_DIR:-$DRIVE_DIR/run}"
STATE_DIR="${STATE_DIR:-/content}"
TRAIN_SUBSET="${TRAIN_SUBSET:-40000}"
VAL_SUBSET="${VAL_SUBSET:-5000}"
EPOCHS="${EPOCHS:-2}"
BATCH_SIZE="${BATCH_SIZE:-96}"
WORKERS="${WORKERS:-2}"

fail() { printf '\nERROR: %s\n' "$*" >&2; exit 1; }
cd "$REPO_DIR"

[[ -f "$CACHE_DIR/meta.json" ]] \
  || fail "No dataset cache at $CACHE_DIR. Run the notebook's main cell first; the experiments reuse its cache."

# Same pinned environment (and marker) as scripts/colab_run.sh, so a runtime that already ran
# the main cell skips this.
req_hash="$(cat requirements.txt requirements-train.txt | sha256sum | cut -c1-12)"
deps_marker="$STATE_DIR/.deps-$req_hash"
if [[ ! -f "$deps_marker" ]]; then
  pip install -q -r requirements-train.txt pytest
  touch "$deps_marker"
fi

# name : backbone : freeze : learning rate : seed
# A linear probe (head) at two learning rates, partial unfreeze, full fine-tune twice with
# different seeds (that gap is the noise floor), and the larger backbone.
configs=(
  "head_lr4e-4:convnext_tiny:head:4e-4:42"
  "head_lr2e-3:convnext_tiny:head:2e-3:42"
  "partial_lr4e-4:convnext_tiny:partial:4e-4:42"
  "full_lr4e-4_seed42:convnext_tiny:none:4e-4:42"
  "full_lr4e-4_seed7:convnext_tiny:none:4e-4:7"
  "small_full_lr4e-4_seed42:convnext_small:none:4e-4:42"
)

for cfg in "${configs[@]}"; do
  IFS=':' read -r name backbone freeze lr seed <<< "$cfg"
  if [[ -f "$WORK_DIR/experiments/$name/experiment.json" ]]; then
    printf '\n==> %s [already done]\n' "$name"
    continue
  fi
  printf '\n==> %s (%s, freeze=%s, lr=%s, seed=%s)\n' "$name" "$backbone" "$freeze" "$lr" "$seed"
  python -m scripts.train --experiment "$name" --backbone "$backbone" --freeze "$freeze" \
    --lr "$lr" --seed "$seed" --epochs "$EPOCHS" --batch-size "$BATCH_SIZE" \
    --train-subset "$TRAIN_SUBSET" --val-subset "$VAL_SUBSET" \
    --cache-dir "$CACHE_DIR" --local-cache "$LOCAL_CACHE" --work-dir "$WORK_DIR" \
    --workers "$WORKERS"
done

printf '\n==> summary\n'
python -m scripts.summarize_experiments "$WORK_DIR/experiments"

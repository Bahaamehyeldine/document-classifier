#!/usr/bin/env bash
# Download and extract RVL-CDIP (~39 GB archive, ~37 GB extracted).
# Resumable: re-run the same command if the download is interrupted.
#
# Usage: scripts/download_rvl_cdip.sh [DATA_DIR]      (default: ~/data/rvl-cdip)
set -euo pipefail

DATA_DIR="${1:-$HOME/data/rvl-cdip}"
BASE="https://huggingface.co/datasets/aharley/rvl_cdip/resolve/main/data"
ARCHIVE="$DATA_DIR/rvl-cdip.tar.gz"

mkdir -p "$DATA_DIR/labels"
auth=()
if [[ -n "${HF_TOKEN:-}" ]]; then auth=(-H "Authorization: Bearer $HF_TOKEN"); fi

if [[ -d "$DATA_DIR/images" && -f "$DATA_DIR/.extracted" ]]; then
  echo "Images already extracted in $DATA_DIR/images"
else
  echo "Downloading archive to $ARCHIVE (resumes if partially downloaded)..."
  curl -L --fail --retry 10 --retry-delay 5 -C - "${auth[@]}" -o "$ARCHIVE" "$BASE/rvl-cdip.tar.gz"
  echo "Extracting (takes a while)..."
  tar -xzf "$ARCHIVE" -C "$DATA_DIR"
  touch "$DATA_DIR/.extracted"
  rm -f "$ARCHIVE"
fi

for split in train val test; do
  if [[ ! -s "$DATA_DIR/labels/$split.txt" ]]; then
    curl -L --fail --retry 5 "${auth[@]}" -o "$DATA_DIR/labels/$split.txt" "$BASE/$split.txt"
  fi
done

echo
echo "Done. Images: $(find "$DATA_DIR/images" -type f | wc -l)"
wc -l "$DATA_DIR"/labels/*.txt

#!/usr/bin/env bash
# Download and extract RVL-CDIP (~39 GB archive, ~37 GB extracted).
# Resumable: dropped connections are retried automatically, and re-running the
# same command continues from the bytes already on disk.
#
# Usage: scripts/download_rvl_cdip.sh [DATA_DIR]      (default: ~/data/rvl-cdip)
set -euo pipefail

DATA_DIR="${1:-$HOME/data/rvl-cdip}"
BASE="https://huggingface.co/datasets/aharley/rvl_cdip/resolve/main/data"
ARCHIVE="$DATA_DIR/rvl-cdip.tar.gz"
MAX_ATTEMPTS=50

mkdir -p "$DATA_DIR/labels"
auth=()
if [[ -n "${HF_TOKEN:-}" ]]; then auth=(-H "Authorization: Bearer $HF_TOKEN"); fi

# HTTP/1.1 avoids the HTTP/2 stream resets (curl exit 92) seen on long CDN downloads;
# --retry-all-errors also retries resets and partial transfers, not just timeouts.
fetch() {  # fetch URL OUTPUT: resumable download with retries around curl itself
  local url="$1" out="$2" attempt=1
  until curl -L --fail --http1.1 --retry 5 --retry-all-errors --retry-delay 10 \
      --speed-limit 1024 --speed-time 60 -C - "${auth[@]}" -o "$out" "$url"; do
    local code=$?
    if [[ $code -eq 33 ]]; then  # server does not support resuming: file already complete
      return 0
    fi
    if (( attempt >= MAX_ATTEMPTS )); then
      echo "Download failed after $attempt attempts (curl exit $code)." >&2
      return "$code"
    fi
    echo "curl exit $code; resuming in 15 s (attempt $((attempt + 1))/$MAX_ATTEMPTS)..." >&2
    attempt=$((attempt + 1))
    sleep 15
  done
}

if [[ -d "$DATA_DIR/images" && -f "$DATA_DIR/.extracted" ]]; then
  echo "Images already extracted in $DATA_DIR/images"
else
  echo "Downloading archive to $ARCHIVE (resumes if partially downloaded)..."
  fetch "$BASE/rvl-cdip.tar.gz" "$ARCHIVE"
  echo "Checking archive integrity..."
  gzip -t "$ARCHIVE"
  echo "Extracting (takes a while)..."
  tar -xzf "$ARCHIVE" -C "$DATA_DIR"
  touch "$DATA_DIR/.extracted"
  rm -f "$ARCHIVE"
fi

for split in train val test; do
  if [[ ! -s "$DATA_DIR/labels/$split.txt" ]]; then
    fetch "$BASE/$split.txt" "$DATA_DIR/labels/$split.txt"
  fi
done

echo
echo "Done. Images: $(find "$DATA_DIR/images" -type f | wc -l)"
wc -l "$DATA_DIR"/labels/*.txt

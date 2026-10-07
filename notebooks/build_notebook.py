"""Builds notebooks/train_rvl_cdip.ipynb (run: python notebooks/build_notebook.py).

The notebook is deliberately thin: every step is a tested script in scripts/, and every step is
safe to re-run after a Colab interruption (it skips what is already on Drive).
"""

import nbformat as nbf

cells = []


def md(s):
    cells.append(nbf.v4.new_markdown_cell(s.strip()))


def code(s):
    cells.append(nbf.v4.new_code_cell(s.strip()))


md("""
# Document Classifier: fine-tune ConvNeXt on RVL-CDIP (Colab, interruption-proof)

Produces the artifacts the service ships with:

| Artifact | Path in repo |
|---|---|
| Trained weights (`state_dict`) | `app/classifier/models/classifier.pt` (commit with git LFS) |
| Model card (SHA-256, metrics, environment, git commit) | `app/classifier/models/model_card.json` |
| 50 golden test images | `app/classifier/eval/golden_images/` |
| Golden expected outputs | `app/classifier/eval/golden_expected.json` |

**Before you run:** *Runtime → Change runtime type → GPU* (T4 is enough). Keep this tab open.

## If the runtime is interrupted
Colab wipes its local disk when it disconnects, so nothing important lives there. Everything that
took time is kept on **Google Drive**, and every step below skips what is already done:

1. **Cache** (step 2): the dataset resized to 224 px (exactly the service's preprocessing), in
   shards of 10k pages. Saved shard by shard, so a disconnect loses seconds, not hours.
2. **Training checkpoints** (step 4): written to Drive every 300 steps, and at each epoch.

So after any interruption: *Runtime → Run all* again (or just re-run the cell that was running).
A finished step prints its result immediately instead of redoing the work.
""")

code("""
# ---- Config ----
REPO_URL = "https://github.com/Bahaamehyeldine/document-classifier.git"
REPO_BRANCH = "colab-resumable-cache"   # switch to "main" once this branch is merged
DRIVE_DIR = "/content/drive/MyDrive/document-classifier"

BACKBONE = "convnext_tiny"       # or "convnext_small" (slower, about 1 point better)
EPOCHS = 3
BATCH_SIZE = 64                  # fits a 16 GB T4 in float16
TRAIN_PER_CLASS = None           # None = all 320k training pages. 6000 = 96k pages: a quicker, smaller
                                 # cache (validation and test stay complete). Changing it needs a new cache.
""")

code("""
# ---- Mount Drive, check the GPU and the free Drive space ----
import os, shutil
from google.colab import drive
drive.mount("/content/drive")
os.makedirs(DRIVE_DIR, exist_ok=True)

!nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
total, used, free = shutil.disk_usage("/content/drive")
print(f"Drive free: {free / 1e9:.1f} GB of {total / 1e9:.1f} GB")
print("The full cache is roughly 10-14 GB (estimate; `info` below prints the real size); with "
      "TRAIN_PER_CLASS=6000 roughly 4-6 GB.")
if free < (12e9 if TRAIN_PER_CLASS is None else 6e9):
    print("WARNING: this may not fit. Free some Drive space or set TRAIN_PER_CLASS.")
""")

code("""
# ---- Get the repo: training must use the service's exact preprocessing ----
!rm -rf /content/repo && git clone -q --depth 1 -b {REPO_BRANCH} {REPO_URL} /content/repo
%cd /content/repo
!git log --oneline -1
!pip install -q tqdm
""")

md("""
## 2. Build the 224 px cache (one-time, resumable; ~1 h)
Streams the 39 GB archive once, resizes every page with `RESIZE_TRANSFORM` (the first half of the
service's eval transform) and saves it to Drive in shards. It also hashes each full-size page for the
duplicate / leakage audit, and keeps the original TIFFs of 1 in 20 test pages for the golden set.

**Re-run this cell until it prints `CACHE COMPLETE`.** After an interruption it resumes: it re-streams
the archive but decodes and stores only pages that are not cached yet.
License: RVL-CDIP is academic / research use only (see `LICENSES.md`).
""")

code("""
CACHE = f"{DRIVE_DIR}/cache"
per_class = f"--train-per-class {TRAIN_PER_CLASS}" if TRAIN_PER_CLASS else ""
!python -m scripts.rvl_cache build --out {CACHE} --workers 2 {per_class}
!du -sh {CACHE}
""")

md("""
## 3. Audit the dataset (seconds; reads the hashes recorded in step 2)
Counts exact duplicates, train/test leakage, conflicting labels and unreadable scans **without changing
the official splits**. The training script then reports the official test score *and* a
leakage-controlled score, and draws the golden set from leakage-controlled pages.
""")

code("""
AUDIT = f"{DRIVE_DIR}/audit"
!python -m scripts.audit_dataset --cache-dir {CACHE} --out-dir {AUDIT} | head -60
""")

md("""
## 4. Train (resumable)
Full fine-tune of an ImageNet-pretrained ConvNeXt with a new 16-way head. The best epoch is chosen on
**validation only**; the 40k test split is evaluated once, at the end, on that frozen checkpoint.
Checkpoints go to Drive every 300 steps: **after an interruption, re-run this cell** and it continues
mid-epoch from the last checkpoint. (Unpacking the cache onto the new runtime's disk takes a few
minutes first.)
""")

code("""
RUN = f"{DRIVE_DIR}/run"
!python -m scripts.train_local --cache-dir {CACHE} --local-cache /content/cache-local \\
    --work-dir {RUN} --audit-dir {AUDIT} --backbone {BACKBONE} --epochs {EPOCHS} \\
    --batch-size {BATCH_SIZE} --workers 2 --checkpoint-every 300
""")

md("""
## 5. Package the artifacts
`train_local` wrote them into the cloned repo. Zip them to Drive (so they survive anything) and download.
""")

code("""
import json
card = json.load(open("/content/repo/app/classifier/models/model_card.json"))
print(json.dumps({k: card[k] for k in ("backbone", "freeze_policy", "sha256", "git")}, indent=2))
print("test (official):", card["metrics"]["test"])
print("test (leakage-controlled):", card["metrics"]["test_leakage_controlled"])
print("golden:", card["metrics"]["golden"])
""")

code("""
!cd /content/repo && zip -qr {DRIVE_DIR}/classifier_artifacts.zip app/classifier/models app/classifier/eval -x "*/__pycache__/*"
!cp {RUN}/{BACKBONE}/best_state_dict.pt {DRIVE_DIR}/ 2>/dev/null; ls -lh {DRIVE_DIR}/classifier_artifacts.zip
from google.colab import files
files.download(f"{DRIVE_DIR}/classifier_artifacts.zip")
""")

md("""
## 6. Install into the repo
Unzip into the repo root (paths already match `app/classifier/`), then:
```bash
git lfs install && git lfs track "app/classifier/models/*.pt"
git add .gitattributes app/classifier && git commit -m "Add trained classifier artifacts"
pip install -r requirements.txt && pytest app/classifier/eval/golden.py
```
""")

nb = nbf.v4.new_notebook(
    cells=cells,
    metadata={
        "accelerator": "GPU",
        "colab": {"gpuType": "T4", "provenance": []},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
    },
)
nbf.write(nb, "notebooks/train_rvl_cdip.ipynb")
print("ok")

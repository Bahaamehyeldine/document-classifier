"""Builds notebooks/train_rvl_cdip.ipynb (run: python notebooks/build_notebook.py).

The notebook is deliberately thin: it prepares a Colab GPU runtime and calls
scripts/train.py, so Colab and local runs share one training implementation.
"""

import nbformat as nbf

cells = []


def md(s):
    cells.append(nbf.v4.new_markdown_cell(s.strip()))


def code(s):
    cells.append(nbf.v4.new_code_cell(s.strip()))


md("""
# Document Classifier: fine-tune ConvNeXt on RVL-CDIP (Colab)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Bahaamehyeldine/document-classifier/blob/main/notebooks/train_rvl_cdip.ipynb)

Produces the four artifacts the service ships with:

| Artifact | Path in repo |
|---|---|
| Trained weights (`state_dict`) | `app/classifier/models/classifier.pt` (commit with git LFS) |
| Model card (SHA-256, metrics, environment) | `app/classifier/models/model_card.json` |
| 50 golden test images | `app/classifier/eval/golden_images/` |
| Golden expected outputs | `app/classifier/eval/golden_expected.json` |

**How to run:** *Runtime → Change runtime type → T4 GPU*, then *Runtime → Run all* and allow the Google Drive prompt.

**Time on a free T4:** about 40 min to download and extract the dataset, then roughly 1–1.5 h per epoch. Keep the tab open.

**If Colab disconnects:** run all cells again. Checkpoints are written to Google Drive every 300 steps, so training
continues from where it stopped (the dataset is downloaded again, because Colab's disk is wiped).

All training logic lives in [`scripts/train.py`](../scripts/train.py), which imports the service's own preprocessing
and model constructor, so training and the inference worker cannot drift apart.
""")

code("""
# ---- Config ----
REPO_URL = "https://github.com/Bahaamehyeldine/document-classifier.git"
REPO_BRANCH = "main"
BACKBONE = "convnext_tiny"   # or "convnext_small" (slower, about 1 point better)
EPOCHS = 2
BATCH_SIZE = 96
TRAIN_SUBSET = None          # e.g. 80_000 for a faster, less accurate run; None = all 320k
DATA_ROOT = "/content/rvl-cdip"
DRIVE_DIR = "/content/drive/MyDrive/document-classifier"
""")

code("""
# ---- GPU check and Google Drive (checkpoints and final artifacts survive a disconnect) ----
import os, subprocess
gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                     capture_output=True, text=True)
assert gpu.returncode == 0, "No GPU. Use Runtime → Change runtime type → T4 GPU, then run all again."
print("GPU:", gpu.stdout.strip())

from google.colab import drive
drive.mount("/content/drive")
WORK_DIR = f"{DRIVE_DIR}/run"
os.makedirs(WORK_DIR, exist_ok=True)
!df -h /content | tail -1
""")

code("""
# ---- Repo and the pinned training environment (same versions the service and CI use) ----
!rm -rf /content/repo && git clone -q --depth 1 -b {REPO_BRANCH} {REPO_URL} /content/repo
!pip install -q -r /content/repo/requirements-train.txt
!cd /content/repo && python -c "import torch, torchvision; print('torch', torch.__version__, '| torchvision', torchvision.__version__, '| cuda', torch.cuda.is_available())"
""")

md("""
## 1. Download RVL-CDIP

Streams the ~36 GB archive straight into `tar`, so the tarball never sits on disk next to its extracted copy.
Licence: academic / research use only (see `LICENSES.md`).
""")

code("""
HF = "https://huggingface.co/datasets/aharley/rvl_cdip/resolve/main/data"
if not os.path.exists(f"{DATA_ROOT}/.extracted"):
    os.makedirs(f"{DATA_ROOT}/labels", exist_ok=True)
    !bash -c 'set -euo pipefail; curl -L --fail -sS --retry 5 "{HF}/rvl-cdip.tar.gz" | tar -xz -C {DATA_ROOT}' && touch {DATA_ROOT}/.extracted
    for split in ("train", "val", "test"):
        !curl -L --fail -sS --retry 5 "{HF}/{split}.txt" -o {DATA_ROOT}/labels/{split}.txt
assert os.path.exists(f"{DATA_ROOT}/.extracted"), "Download or extraction failed; run this cell again."
!wc -l {DATA_ROOT}/labels/*.txt; df -h /content | tail -1
""")

md("""
## 2. Train, evaluate, pick the golden set, write the model card

One command does all of it: fine-tune, keep the best epoch by validation top-1, evaluate once on the 40k test split,
pick the 50 golden images, record their outputs on CPU in float32 (what CI replays), and write the model card.
Safe to re-run: it resumes from the last checkpoint on Drive.
""")

code("""
subset = f"--train-subset {TRAIN_SUBSET}" if TRAIN_SUBSET else ""
!cd /content/repo && python -m scripts.train --data-root {DATA_ROOT} --work-dir {WORK_DIR} \\
    --backbone {BACKBONE} --epochs {EPOCHS} --batch-size {BATCH_SIZE} {subset}
""")

md("## 3. Verify: the golden replay and artifact checks that CI runs")

code("""
!cd /content/repo && pip install -q pytest && python -m pytest -q app/classifier/eval/golden.py
!cat /content/repo/app/classifier/models/model_card.json | python -c "import json,sys; c=json.load(sys.stdin); print(json.dumps({k: c[k] for k in ('backbone','sha256')}, indent=2)); print(c['metrics']['test']); print(c['environment'])"
""")

md("""
## 4. Save and download

The zip is copied to Google Drive and downloaded by the browser. Unzip it at the repo root, then:
```bash
pytest -q app/classifier/eval/golden.py tests/unit
git lfs install && git add app/classifier/models app/classifier/eval
git commit -m "feat(classifier): trained ConvNeXt Tiny weights, model card and golden set" && git push
```
""")

code("""
!cd /content/repo && rm -f /content/classifier_artifacts.zip && zip -qr /content/classifier_artifacts.zip app/classifier/models/classifier.pt app/classifier/models/model_card.json app/classifier/eval/golden_images app/classifier/eval/golden_expected.json
!cp /content/classifier_artifacts.zip {DRIVE_DIR}/ && ls -lh {DRIVE_DIR}/classifier_artifacts.zip
from google.colab import files; files.download("/content/classifier_artifacts.zip")
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

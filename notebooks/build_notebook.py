"""Builds notebooks/train_rvl_cdip.ipynb (run: python notebooks/build_notebook.py).

The notebook is deliberately thin: one cell mounts Google Drive, fetches the repo and
calls scripts/colab_run.sh, an idempotent pipeline that skips every finished step.
Re-running that one cell after a disconnect continues from where the run stopped.
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

## How to run

1. *Runtime → Change runtime type → T4 GPU*.
2. Run the **Run or resume** cell below and allow the Google Drive prompt.

## If Colab disconnects

**Run the same cell again. Nothing else.** It needs no other cell and no leftover variables, and it skips
every step that is already finished:

| Step | Skipped when | Otherwise |
|---|---|---|
| Python environment | already installed in this runtime | a few minutes |
| Dataset | already extracted in this runtime | 30–40 min |
| Training | resumes from the last checkpoint in Google Drive (saved every 300 steps) | 1–1.5 h per epoch on a T4 |
| Evaluate, golden set, model card, zip | the zip is already in Drive | about 15 min |

What that means in practice:

- **Connection dropped, runtime still alive:** setup and dataset are skipped, training continues within seconds.
- **Colab deleted the runtime:** its disk is wiped, so the environment and dataset are restored automatically first,
  then training continues from the Drive checkpoint. No training progress is lost beyond the last few minutes.

All logic lives in the repo: [`scripts/colab_run.sh`](../scripts/colab_run.sh) (the steps above) and
[`scripts/train.py`](../scripts/train.py) (training, which imports the service's own preprocessing and model
constructor so training and the inference worker cannot drift apart).
""")

code("""
# ==== Run or resume (after any disconnect, run only this cell) ====
# Keep these settings unchanged between runs: a checkpoint only resumes with the settings it was saved with.
REPO_URL = "https://github.com/Bahaamehyeldine/document-classifier.git"
REPO_BRANCH = "main"
BACKBONE = "convnext_tiny"   # or "convnext_small" (slower, about 1 point better)
EPOCHS = 2
BATCH_SIZE = 96
TRAIN_SUBSET = None          # e.g. 80_000 for a faster, less accurate run; None = all 320k
DRIVE_DIR = "/content/drive/MyDrive/document-classifier"   # checkpoints and final artifacts

import os
from google.colab import drive

drive.mount("/content/drive")   # does nothing if Drive is already mounted

if os.path.isdir("/content/repo/.git"):
    !git -C /content/repo pull -q --ff-only
else:
    !git clone -q --depth 1 -b {REPO_BRANCH} {REPO_URL} /content/repo

os.environ.update(BACKBONE=BACKBONE, EPOCHS=str(EPOCHS), BATCH_SIZE=str(BATCH_SIZE),
                  TRAIN_SUBSET=str(TRAIN_SUBSET or ""), DRIVE_DIR=DRIVE_DIR)
!bash /content/repo/scripts/colab_run.sh
""")

md("""
## Download the artifacts

The zip is already safe in Google Drive (`document-classifier/classifier_artifacts.zip`); this cell also downloads it
through the browser. It is self-contained too, so it works in a fresh runtime. Unzip at the repo root, then:
```bash
pytest -q app/classifier/eval/golden.py tests/unit
git lfs install && git add app/classifier/models app/classifier/eval
git commit -m "feat(classifier): trained ConvNeXt Tiny weights, model card and golden set" && git push
```
""")

code("""
import os
from google.colab import drive, files

drive.mount("/content/drive")
ZIP = "/content/drive/MyDrive/document-classifier/classifier_artifacts.zip"
if os.path.exists(ZIP):
    files.download(ZIP)
else:
    print("No artifacts yet: training has not finished. Run the 'Run or resume' cell above.")
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

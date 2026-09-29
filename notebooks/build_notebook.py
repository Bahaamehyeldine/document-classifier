"""Builds notebooks/train_rvl_cdip.ipynb (run: python notebooks/build_notebook.py)."""

import nbformat as nbf

cells = []


def md(s):
    cells.append(nbf.v4.new_markdown_cell(s.strip()))


def code(s):
    cells.append(nbf.v4.new_code_cell(s.strip()))


md("""
# Document Classifier: fine-tune ConvNeXt on RVL-CDIP (Colab)

Produces the four artifacts the service ships with:

| Artifact | Path in repo |
|---|---|
| Trained weights (`state_dict`) | `app/classifier/models/classifier.pt` (commit with git LFS) |
| Model card (SHA-256, metrics, environment) | `app/classifier/models/model_card.json` |
| 50 golden test images | `app/classifier/eval/golden_images/` |
| Golden expected outputs | `app/classifier/eval/golden_expected.json` |

**Before you run:** *Runtime → Change runtime type → GPU* (T4 is enough; A100/L4 is faster).
Budget: about 1 h to download and extract the ~37 GB dataset, then about 1–1.5 h per epoch on a T4 for ConvNeXt Tiny.
Use `TRAIN_SUBSET` to trade accuracy for time.

The eval preprocessing is imported from the repo (`app/classifier/preprocessing.py`), so training and the worker cannot drift apart.
""")

code("""
# ---- Config ----
REPO_URL = "https://github.com/Bahaamehyeldine/document-classifier.git"
REPO_BRANCH = "main"
BACKBONE = "convnext_tiny"          # or "convnext_small" (slower, ~1 pt better)
EPOCHS = 2
TRAIN_SUBSET = None                  # e.g. 80_000 for a faster run; None = all 320k
BATCH_SIZE = 96
LR = 4e-4
WEIGHT_DECAY = 0.05
NUM_WORKERS = 8
SEED = 42
DATA_ROOT = "/content/rvl"
DRIVE_OUT = "/content/drive/MyDrive/document-classifier-artifacts"
""")

code("""
!nvidia-smi --query-gpu=name,memory.total --format=csv
from google.colab import drive
drive.mount("/content/drive")
import os; os.makedirs(DRIVE_OUT, exist_ok=True)
""")

code("""
# ---- Get the repo so we train with the service's exact preprocessing ----
!rm -rf /content/repo && git clone -q --depth 1 -b {REPO_BRANCH} {REPO_URL} /content/repo
import sys; sys.path.insert(0, "/content/repo")
from app.classifier.labels import RVL_CDIP_CLASSES, NUM_CLASSES
from app.classifier.preprocessing import (INPUT_SIZE, IMAGENET_MEAN, IMAGENET_STD, load_image, to_tensor)
from app.classifier.model import build_model, sha256_file
print(NUM_CLASSES, "classes:", RVL_CDIP_CLASSES)
""")

md("""
## 1. Download RVL-CDIP (stream-extract, ~1 h)
Streams the archive straight into `tar`, so the 37 GB tarball never sits on disk next to its extracted copy.
License: academic / research use only (keep the note in `LICENSES.md`).
""")

code("""
import os
if not os.path.isdir(f"{DATA_ROOT}/images"):
    os.makedirs(DATA_ROOT, exist_ok=True)
    !curl -sL "https://huggingface.co/datasets/aharley/rvl_cdip/resolve/main/data/rvl-cdip.tar.gz" | tar -xz -C {DATA_ROOT}
# The split files are also published next to the archive; fetch them if the tarball lacks labels/.
HF = "https://huggingface.co/datasets/aharley/rvl_cdip/resolve/main/data"
os.makedirs(f"{DATA_ROOT}/labels", exist_ok=True)
for split in ("train", "val", "test"):
    if not os.path.exists(f"{DATA_ROOT}/labels/{split}.txt"):
        !curl -sL "{HF}/{split}.txt" -o {DATA_ROOT}/labels/{split}.txt
assert os.path.isdir(f"{DATA_ROOT}/images"), "Download failed: no images/ folder. See adamharley.com/rvl-cdip for mirrors."
!ls {DATA_ROOT}; wc -l {DATA_ROOT}/labels/*.txt; df -h /content | tail -1
""")

code("""
import random, torch, numpy as np
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

def read_split(name):
    rows = []
    for line in open(f"{DATA_ROOT}/labels/{name}.txt"):
        rel, lab = line.split()
        rows.append((f"{DATA_ROOT}/images/{rel}", int(lab)))
    return rows

train_rows, val_rows, test_rows = read_split("train"), read_split("val"), read_split("test")
if TRAIN_SUBSET:
    train_rows = random.sample(train_rows, TRAIN_SUBSET)
print(len(train_rows), len(val_rows), len(test_rows))

# Light, layout-preserving augmentation for training only.
TRAIN_TRANSFORM = v2.Compose([
    v2.ToImage(),
    v2.RandomAffine(degrees=2, translate=(0.02, 0.02), scale=(0.95, 1.05), fill=255),
    v2.Resize((INPUT_SIZE, INPUT_SIZE), antialias=True),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

class RVLDataset(Dataset):
    def __init__(self, rows, train):
        self.rows, self.train = rows, train
    def __len__(self):
        return len(self.rows)
    def __getitem__(self, i):
        path, label = self.rows[i]
        try:
            img = load_image(path)            # same loader as the worker
        except Exception:                     # RVL-CDIP has a few corrupt TIFFs
            return None
        x = TRAIN_TRANSFORM(img) if self.train else to_tensor(img)
        return x, label, i

def collate(batch):
    batch = [b for b in batch if b is not None]
    xs, ys, idx = zip(*batch)
    return torch.stack(xs), torch.tensor(ys), torch.tensor(idx)

def loader(rows, train):
    return DataLoader(RVLDataset(rows, train), batch_size=BATCH_SIZE, shuffle=train,
        num_workers=NUM_WORKERS, pin_memory=True, collate_fn=collate, persistent_workers=True)

train_dl, val_dl, test_dl = loader(train_rows, True), loader(val_rows, False), loader(test_rows, False)
""")

md("## 2. Model: ImageNet-pretrained ConvNeXt, new 16-way head, full fine-tune")

code("""
from torchvision import models
WEIGHTS_ENUM = {"convnext_tiny": "ConvNeXt_Tiny_Weights.IMAGENET1K_V1",
                "convnext_small": "ConvNeXt_Small_Weights.IMAGENET1K_V1"}[BACKBONE]
FREEZE_POLICY = "none (full fine-tune, all layers trainable)"

model = getattr(models, BACKBONE)(weights="IMAGENET1K_V1")
model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, NUM_CLASSES)
device = "cuda"
model = model.to(device).to(memory_format=torch.channels_last)

opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
steps = EPOCHS * len(train_dl)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=steps, pct_start=0.05)
loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
scaler = torch.amp.GradScaler()
""")

code("""
from tqdm.auto import tqdm

@torch.inference_mode()
def evaluate(dl):
    model.eval(); all_p, all_y, all_i = [], [], []
    for x, y, i in tqdm(dl, leave=False):
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(x.to(device, non_blocking=True).to(memory_format=torch.channels_last))
        all_p.append(torch.softmax(logits.float(), 1).cpu()); all_y.append(y); all_i.append(i)
    return torch.cat(all_p), torch.cat(all_y), torch.cat(all_i)

def topk_acc(p, y, k):
    return (p.topk(k, 1).indices == y[:, None]).any(1).float().mean().item()

best_val = 0.0
for epoch in range(EPOCHS):
    model.train()
    for x, y, _ in tqdm(train_dl, desc=f"epoch {epoch+1}/{EPOCHS}"):
        x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last); y = y.to(device)
        with torch.autocast("cuda", dtype=torch.float16):
            loss = loss_fn(model(x), y)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
    p, yv, _ = evaluate(val_dl)
    val_top1 = topk_acc(p, yv, 1)
    print(f"epoch {epoch+1}: val top-1 {val_top1:.4f}  top-5 {topk_acc(p, yv, 5):.4f}")
    if val_top1 > best_val:
        best_val = val_top1
        torch.save(model.state_dict(), f"{DRIVE_OUT}/best_state_dict.pt")
""")

md("## 3. Evaluate the best checkpoint on the full 40k test split (once)")

code("""
model.load_state_dict(torch.load(f"{DRIVE_OUT}/best_state_dict.pt", map_location=device))
test_p, test_y, test_i = evaluate(test_dl)
test_top1, test_top5 = topk_acc(test_p, test_y, 1), topk_acc(test_p, test_y, 5)
pred = test_p.argmax(1)
per_class = {c: (pred[test_y == k] == k).float().mean().item() for k, c in enumerate(RVL_CDIP_CLASSES)}
print(f"TEST top-1 {test_top1:.4f}  top-5 {test_top5:.4f}  (n={len(test_y)})")
for c, a in per_class.items(): print(f"  {c:24s} {a:.4f}")
""")

md("""
## 4. Pick the 50-image golden set
Per class: 2 **easy** (confidently correct) + 1 **ambiguous** (smallest top-1/top-2 margin), which gives 48.
Then the 2 most ambiguous remaining images overall, for 50 in total.
""")

code("""
top2 = test_p.topk(2, 1).values
margin = (top2[:, 0] - top2[:, 1])
conf = top2[:, 0]
chosen = []
for k in range(NUM_CLASSES):
    idx = (test_y == k).nonzero().squeeze(1)
    correct = idx[pred[idx] == k]
    easy = correct[conf[correct].argsort(descending=True)[:2]]
    amb = idx[margin[idx].argsort()[:1]]
    chosen += easy.tolist() + amb.tolist()
rest = [j for j in margin.argsort().tolist() if j not in set(chosen)]
chosen += rest[:50 - len(chosen)]
assert len(set(chosen)) == 50
golden_rows = [test_rows[test_i[j].item()] for j in chosen]
""")

md("""
## 5. Record golden expected outputs **on CPU in float32**
CI replays on CPU, so the expected confidences must come from CPU fp32 too. GPU/fp16 numbers would not match within 1e-6.
""")

code("""
import json, shutil, os, platform, datetime, PIL, torchvision
ART = "/content/artifacts"
shutil.rmtree(ART, ignore_errors=True)
os.makedirs(f"{ART}/models"); os.makedirs(f"{ART}/eval/golden_images")

cpu_model = build_model(BACKBONE)                     # same constructor the service uses
cpu_model.load_state_dict({k: v.cpu() for k, v in model.state_dict().items()})
cpu_model.eval()
torch.save(cpu_model.state_dict(), f"{ART}/models/classifier.pt")

images, g_correct1, g_correct5 = [], 0, 0
with torch.inference_mode():
    for n, (path, true) in enumerate(golden_rows):
        fname = f"{n:02d}_{RVL_CDIP_CLASSES[true]}.tif"
        shutil.copy(path, f"{ART}/eval/golden_images/{fname}")
        p = torch.softmax(cpu_model(to_tensor(load_image(path)).unsqueeze(0)), 1)[0]
        top5 = p.topk(5)
        lab = top5.indices[0].item()
        g_correct1 += lab == true; g_correct5 += true in top5.indices.tolist()
        images.append({"file": fname, "true_label": RVL_CDIP_CLASSES[true],
                       "expected_label": RVL_CDIP_CLASSES[lab],
                       "expected_confidence": top5.values[0].item(),
                       "top5": [[RVL_CDIP_CLASSES[i], v] for i, v in zip(top5.indices.tolist(), top5.values.tolist())]})
json.dump({"generated_on": "cpu-float32", "images": images}, open(f"{ART}/eval/golden_expected.json", "w"), indent=2)
print(f"golden top-1 {g_correct1/50:.2f}  top-5 {g_correct5/50:.2f}")
""")

md("## 6. Model card")

code("""
card = {
    "name": "rvl-cdip-document-classifier",
    "backbone": BACKBONE,
    "pretrained_weights": WEIGHTS_ENUM,
    "freeze_policy": FREEZE_POLICY,
    "classes": list(RVL_CDIP_CLASSES),
    "input": {"size": INPUT_SIZE, "channels": "grayscale replicated to RGB", "normalization": "ImageNet"},
    "training": {"epochs": EPOCHS, "train_images": len(train_rows), "batch_size": BATCH_SIZE, "lr": LR,
                 "optimizer": "AdamW", "schedule": "OneCycle", "label_smoothing": 0.1, "seed": SEED,
                 "best_val_top1": best_val},
    "metrics": {
        "test": {"top1": test_top1, "top5": test_top5, "n": int(len(test_y))},
        "golden": {"top1": g_correct1 / 50, "top5": g_correct5 / 50, "n": 50},
        "per_class_test_accuracy": per_class,
    },
    "sha256": sha256_file(__import__("pathlib").Path(f"{ART}/models/classifier.pt")),
    "environment": {"python": platform.python_version(), "torch": torch.__version__,
                    "torchvision": torchvision.__version__, "pillow": PIL.__version__,
                    "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0)},
    "dataset": {"name": "RVL-CDIP", "license": "academic / research use only"},
    "trained_at": datetime.datetime.utcnow().isoformat() + "Z",
}
json.dump(card, open(f"{ART}/models/model_card.json", "w"), indent=2)
print(json.dumps({k: card[k] for k in ("backbone", "sha256")}, indent=2), card["metrics"]["test"])
""")

md("""
## 7. Save and download
Unzip into the repo root (paths match `app/classifier/`), then:
```bash
git lfs install && git lfs track "app/classifier/models/*.pt"
git add .gitattributes app/classifier && git commit -m "Add trained classifier artifacts"
pip install -r requirements.txt && pytest app/classifier/eval/golden.py
```
""")

code("""
!cd {ART} && mkdir -p pkg/app/classifier && cp -r models eval pkg/app/classifier/ && cd pkg && zip -qr ../classifier_artifacts.zip app
!cp {ART}/classifier_artifacts.zip {DRIVE_OUT}/
from google.colab import files; files.download(f"{ART}/classifier_artifacts.zip")
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

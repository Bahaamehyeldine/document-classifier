"""Train the RVL-CDIP document classifier on a local GPU and write the service artifacts.

Produces exactly what the service verifies at startup:
    app/classifier/models/classifier.pt         state_dict (commit with git LFS)
    app/classifier/models/model_card.json       SHA-256, metrics, environment
    app/classifier/eval/golden_images/*.tif     50 held-out test scans
    app/classifier/eval/golden_expected.json    CPU float32 outputs for the replay test

Uses the service's own preprocessing (app/classifier/preprocessing.py) and model
constructor, so training and inference cannot drift apart.

    python -m scripts.train_local --smoke          # ~5 min dry run on a small subset
    python -m scripts.train_local                  # full run (resumes from the last epoch)
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import random
import shutil
import time
from pathlib import Path

import numpy as np
import PIL
import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from torchvision.transforms import v2
from tqdm import tqdm

from app.classifier.artifacts import MIN_TEST_TOP1, sha256_file
from app.classifier.labels import NUM_CLASSES, RVL_CDIP_CLASSES
from app.classifier.model import build_model
from app.classifier.preprocessing import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    INPUT_SIZE,
    load_image,
    to_tensor,
)

REPO = Path(__file__).resolve().parents[1]
MODELS_DIR = REPO / "app" / "classifier" / "models"
EVAL_DIR = REPO / "app" / "classifier" / "eval"
WEIGHTS_ENUM = {
    "convnext_tiny": "ConvNeXt_Tiny_Weights.IMAGENET1K_V1",
    "convnext_small": "ConvNeXt_Small_Weights.IMAGENET1K_V1",
}
FREEZE_POLICY = "none (full fine-tune, all layers trainable)"

# Light, layout-preserving augmentation for training only.
TRAIN_TRANSFORM = v2.Compose(
    [
        v2.ToImage(),
        v2.RandomAffine(degrees=2, translate=(0.02, 0.02), scale=(0.95, 1.05), fill=255),
        v2.Resize((INPUT_SIZE, INPUT_SIZE), antialias=True),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--data-root", type=Path, default=Path.home() / "data" / "rvl-cdip")
    p.add_argument("--work-dir", type=Path, default=Path.home() / "data" / "rvl-cdip-run")
    p.add_argument("--backbone", choices=sorted(WEIGHTS_ENUM), default="convnext_tiny")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=96)
    p.add_argument("--lr", type=float, default=4e-4)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--workers", type=int, default=min(12, os.cpu_count() or 4))
    p.add_argument("--train-subset", type=int, default=None, help="use N random training images")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--smoke",
        action="store_true",
        help="quick end-to-end check: 2k train / 1k val / 1k test images, 1 epoch, "
        "artifacts written to --work-dir/smoke instead of the repo",
    )
    p.add_argument("--device", default="cuda")
    p.add_argument("--no-pretrained", action="store_true", help=argparse.SUPPRESS)  # offline tests
    return p.parse_args()


def read_split(root: Path, name: str) -> list[tuple[Path, int]]:
    rows = []
    for line in (root / "labels" / f"{name}.txt").read_text().splitlines():
        if line.strip():
            rel, label = line.split()
            rows.append((root / "images" / rel, int(label)))
    return rows


class RVLDataset(Dataset):
    def __init__(self, rows: list[tuple[Path, int]], train: bool):
        self.rows, self.train = rows, train

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        path, label = self.rows[i]
        try:
            img = load_image(path)  # same loader as the inference worker
        except Exception:  # RVL-CDIP contains a few corrupt TIFFs
            return None
        x = TRAIN_TRANSFORM(img) if self.train else to_tensor(img)
        return x, label, i


def collate(batch):
    batch = [b for b in batch if b is not None]
    xs, ys, idx = zip(*batch, strict=True)
    return torch.stack(xs), torch.tensor(ys), torch.tensor(idx)


def loader(rows, train: bool, args) -> DataLoader:
    return DataLoader(
        RVLDataset(rows, train),
        batch_size=args.batch_size,
        shuffle=train,
        num_workers=args.workers,
        pin_memory=args.device == "cuda",
        collate_fn=collate,
        persistent_workers=args.workers > 0,
        prefetch_factor=4 if args.workers > 0 else None,
    )


def autocast(device: str):
    # bfloat16 on GPU: no loss scaling needed and well supported on recent NVIDIA cards.
    if device == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return torch.autocast("cpu", enabled=False)


@torch.inference_mode()
def evaluate(model, dl, device: str):
    model.eval()
    probs, labels, idxs = [], [], []
    for x, y, i in tqdm(dl, desc="eval", leave=False):
        x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
        with autocast(device):
            logits = model(x)
        probs.append(torch.softmax(logits.float(), 1).cpu())
        labels.append(y)
        idxs.append(i)
    return torch.cat(probs), torch.cat(labels), torch.cat(idxs)


def topk(p: torch.Tensor, y: torch.Tensor, k: int) -> float:
    return (p.topk(k, 1).indices == y[:, None]).any(1).float().mean().item()


def pick_golden(test_p, test_y, n: int = 50) -> list[int]:
    """2 easy (confidently correct) + 1 ambiguous (smallest top-1/top-2 margin) per class,
    then the most ambiguous remaining images overall, up to n."""
    top2 = test_p.topk(2, 1).values
    margin, conf, pred = top2[:, 0] - top2[:, 1], top2[:, 0], test_p.argmax(1)
    chosen: list[int] = []
    for k in range(NUM_CLASSES):
        idx = (test_y == k).nonzero().squeeze(1)
        if len(idx) == 0:
            continue
        correct = idx[pred[idx] == k]
        chosen += correct[conf[correct].argsort(descending=True)[:2]].tolist()
        chosen += idx[margin[idx].argsort()[:1]].tolist()
    seen = set(chosen)
    chosen += [j for j in margin.argsort().tolist() if j not in seen][: n - len(chosen)]
    return list(dict.fromkeys(chosen))[:n]


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit(
                "CUDA is not available. Install the CUDA build of PyTorch (see RUNBOOK.md) "
                "or pass --device cpu for a dry run."
            )
        torch.backends.cudnn.benchmark = True
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    train_rows = read_split(args.data_root, "train")
    val_rows = read_split(args.data_root, "val")
    test_rows = read_split(args.data_root, "test")
    if args.smoke:
        train_rows = random.sample(train_rows, min(2000, len(train_rows)))
        val_rows = random.sample(val_rows, min(1000, len(val_rows)))
        test_rows = random.sample(test_rows, min(1000, len(test_rows)))
        args.epochs = 1
    elif args.train_subset:
        train_rows = random.sample(train_rows, args.train_subset)
    print(f"train {len(train_rows):,}  val {len(val_rows):,}  test {len(test_rows):,}")

    work = args.work_dir / ("smoke" if args.smoke else args.backbone)
    work.mkdir(parents=True, exist_ok=True)
    models_dir = work / "artifacts" / "models" if args.smoke else MODELS_DIR
    eval_dir = work / "artifacts" / "eval" if args.smoke else EVAL_DIR

    device = args.device
    model = getattr(models, args.backbone)(weights=None if args.no_pretrained else "IMAGENET1K_V1")
    model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, NUM_CLASSES)
    model = model.to(device).to(memory_format=torch.channels_last)

    train_dl = loader(train_rows, True, args)
    val_dl = loader(val_rows, False, args)
    test_dl = loader(test_rows, False, args)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = args.epochs * len(train_dl)
    # 5 % warm-up, but at least a few steps so very short (smoke) runs stay valid.
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=total_steps, pct_start=max(0.05, 3 / total_steps)
    )
    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=0.1)

    # ---- resume -------------------------------------------------------------
    ckpt_path, best_path = work / "last.pt", work / "best_state_dict.pt"
    start_epoch, best_val = 0, 0.0
    run_config = {
        "backbone": args.backbone,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "train_images": len(train_rows),
        "lr": args.lr,
    }
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        if ckpt.get("config") != run_config:
            raise SystemExit(
                f"{ckpt_path} was saved with different settings:\n  saved:   {ckpt.get('config')}\n"
                f"  current: {run_config}\nRe-run with the same options, or delete {work} "
                "to start a new run."
            )
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        sched.load_state_dict(ckpt["sched"])
        start_epoch, best_val = ckpt["epoch"] + 1, ckpt["best_val"]
        print(f"Resuming after epoch {start_epoch} (best val top-1 {best_val:.4f})")

    # ---- train --------------------------------------------------------------
    t0 = time.time()
    for epoch in range(start_epoch, args.epochs):
        model.train()
        bar = tqdm(train_dl, desc=f"epoch {epoch + 1}/{args.epochs}")
        for x, y, _ in bar:
            x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            y = y.to(device, non_blocking=True)
            with autocast(device):
                loss = loss_fn(model(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            bar.set_postfix(loss=f"{loss.item():.3f}")
        p, yv, _ = evaluate(model, val_dl, device)
        val_top1 = topk(p, yv, 1)
        print(f"epoch {epoch + 1}: val top-1 {val_top1:.4f}  top-5 {topk(p, yv, 5):.4f}")
        if val_top1 > best_val:
            best_val = val_top1
            torch.save(model.state_dict(), best_path)
        torch.save(
            {
                "model": model.state_dict(),
                "opt": opt.state_dict(),
                "sched": sched.state_dict(),
                "epoch": epoch,
                "best_val": best_val,
                "config": run_config,
            },
            ckpt_path,
        )
    train_minutes = (time.time() - t0) / 60

    # ---- full test evaluation, once, on the best checkpoint -------------------
    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    test_p, test_y, test_i = evaluate(model, test_dl, device)
    test_top1, test_top5 = topk(test_p, test_y, 1), topk(test_p, test_y, 5)
    pred = test_p.argmax(1)
    per_class = {
        c: (pred[test_y == k] == k).float().mean().item() if (test_y == k).any() else None
        for k, c in enumerate(RVL_CDIP_CLASSES)
    }
    print(f"\nTEST top-1 {test_top1:.4f}  top-5 {test_top5:.4f}  (n={len(test_y):,})")
    for c, a in per_class.items():
        print(f"  {c:24s} {a:.4f}" if a is not None else f"  {c:24s} n/a")

    # ---- golden set: expected outputs recorded on CPU in float32 --------------
    # CI replays on CPU; GPU / reduced-precision numbers would not match within 1e-6.
    golden_rows = [test_rows[test_i[j].item()] for j in pick_golden(test_p, test_y)]
    models_dir.mkdir(parents=True, exist_ok=True)
    golden_dir = eval_dir / "golden_images"
    if golden_dir.exists():
        for f in golden_dir.glob("*.tif"):
            f.unlink()
    golden_dir.mkdir(parents=True, exist_ok=True)

    cpu_model = build_model(args.backbone)  # the constructor the service uses
    cpu_model.load_state_dict({k: v.float().cpu() for k, v in model.state_dict().items()})
    cpu_model.eval()
    weights_path = models_dir / "classifier.pt"
    torch.save(cpu_model.state_dict(), weights_path)

    images, correct1, correct5 = [], 0, 0
    with torch.inference_mode():
        for n, (path, true) in enumerate(golden_rows):
            fname = f"{n:02d}_{RVL_CDIP_CLASSES[true]}.tif"
            shutil.copy(path, golden_dir / fname)
            x = to_tensor(load_image(golden_dir / fname)).unsqueeze(0)
            p = torch.softmax(cpu_model(x), 1)[0]
            vals, idx = p.topk(5)
            label = idx[0].item()
            correct1 += label == true
            correct5 += true in idx.tolist()
            images.append(
                {
                    "file": fname,
                    "true_label": RVL_CDIP_CLASSES[true],
                    "expected_label": RVL_CDIP_CLASSES[label],
                    "expected_confidence": vals[0].item(),
                    "top5": [
                        [RVL_CDIP_CLASSES[i], v]
                        for i, v in zip(idx.tolist(), vals.tolist(), strict=True)
                    ],
                }
            )
    (eval_dir / "golden_expected.json").write_text(
        json.dumps({"generated_on": "cpu-float32", "images": images}, indent=2) + "\n"
    )

    # ---- model card -----------------------------------------------------------
    n_golden = len(images)
    card = {
        "name": "rvl-cdip-document-classifier",
        "backbone": args.backbone,
        "pretrained_weights": WEIGHTS_ENUM[args.backbone],
        "freeze_policy": FREEZE_POLICY,
        "classes": list(RVL_CDIP_CLASSES),
        "input": {
            "size": INPUT_SIZE,
            "channels": "grayscale replicated to RGB",
            "normalization": "ImageNet",
        },
        "training": {
            "epochs": args.epochs,
            "train_images": len(train_rows),
            "batch_size": args.batch_size,
            "lr": args.lr,
            "optimizer": "AdamW",
            "schedule": "OneCycle",
            "label_smoothing": 0.1,
            "precision": "bfloat16 autocast" if device == "cuda" else "float32",
            "seed": args.seed,
            "best_val_top1": best_val,
            "train_minutes": round(train_minutes, 1),
        },
        "metrics": {
            "test": {"top1": test_top1, "top5": test_top5, "n": int(len(test_y))},
            "golden": {"top1": correct1 / n_golden, "top5": correct5 / n_golden, "n": n_golden},
            "per_class_test_accuracy": per_class,
        },
        "sha256": sha256_file(weights_path),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "pillow": PIL.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
            "platform": platform.platform(),
        },
        "dataset": {"name": "RVL-CDIP", "license": "academic / research use only"},
        "trained_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
    }
    (models_dir / "model_card.json").write_text(json.dumps(card, indent=2) + "\n")

    print(f"\nArtifacts written to {models_dir} and {eval_dir}")
    gate = "PASSES" if test_top1 >= MIN_TEST_TOP1 else "is BELOW"
    print(f"Test top-1 {test_top1:.4f} {gate} the service quality gate ({MIN_TEST_TOP1}).")


if __name__ == "__main__":
    main()

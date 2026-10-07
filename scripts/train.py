"""Train the RVL-CDIP document classifier on a GPU and write the service artifacts.

Runs unchanged on Colab (notebooks/train_rvl_cdip.ipynb calls it) and on a local NVIDIA GPU.

Produces exactly what the service verifies at startup:
    app/classifier/models/classifier.pt         state_dict (commit with git LFS)
    app/classifier/models/model_card.json       SHA-256, metrics, environment
    app/classifier/eval/golden_images/*.tif     50 held-out test scans
    app/classifier/eval/golden_expected.json    CPU float32 outputs for the replay test

Uses the service's own preprocessing (app/classifier/preprocessing.py) and model
constructor, so training and inference cannot drift apart.

    python -m scripts.train --smoke          # ~5 min dry run on a small subset
    python -m scripts.train                  # full run; re-run to resume an interrupted one

Data comes from extracted TIFFs (--data-root) or, for Colab, from the resumable 224 px cache
built by `scripts/rvl_cache.py` (--cache-dir). Checkpoints go to --work-dir (put it on Drive
on Colab) every few hundred steps, so re-running the same command after an interruption
continues where it stopped.

If `clean_test.txt` from `scripts/audit_dataset.py` is found in --audit-dir, the model card
reports the official test score (the quality gate, comparable to published work) and a
leakage-controlled score, and the golden set is drawn from the leakage-controlled pages.
"""

from __future__ import annotations

import argparse
import datetime as dt
import itertools
import json
import os
import platform
import random
import shutil
import subprocess
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
from scripts.rvl_cache import CachedDataset, CachedRVL, safe_name

REPO = Path(__file__).resolve().parents[1]
MODELS_DIR = REPO / "app" / "classifier" / "models"
EVAL_DIR = REPO / "app" / "classifier" / "eval"
WEIGHTS_ENUM = {
    "convnext_tiny": "ConvNeXt_Tiny_Weights.IMAGENET1K_V1",
    "convnext_small": "ConvNeXt_Small_Weights.IMAGENET1K_V1",
}
FREEZE_POLICY = "none (full fine-tune, all layers trainable)"

# Light, layout-preserving augmentation for training only. Resizing first means the
# affine warp runs on 224 x 224 pixels rather than the ~1000 x 750 scan, which makes
# each data-loading worker several times faster (so fewer workers, and less RAM, suffice).
TRAIN_TRANSFORM = v2.Compose(
    [
        v2.ToImage(),
        v2.Resize((INPUT_SIZE, INPUT_SIZE), antialias=True),
        v2.RandomAffine(degrees=2, translate=(0.02, 0.02), scale=(0.95, 1.05), fill=255),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--data-root", type=Path, default=Path.home() / "data" / "rvl-cdip")
    p.add_argument("--work-dir", type=Path, default=Path.home() / "data" / "rvl-cdip-run")
    p.add_argument(
        "--cache-dir",
        type=Path,
        help="train from the 224 px cache built by scripts/rvl_cache.py instead of --data-root",
    )
    p.add_argument(
        "--local-cache",
        type=Path,
        help="fast local disk for the unpacked cache (default: --work-dir/cache-local; on Colab "
        "use /content/cache-local, not Drive)",
    )
    p.add_argument(
        "--audit-dir",
        type=Path,
        help="where scripts/audit_dataset.py wrote clean_test.txt (default: --work-dir/audit)",
    )
    p.add_argument("--backbone", choices=sorted(WEIGHTS_ENUM), default="convnext_tiny")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=96)
    p.add_argument("--lr", type=float, default=4e-4)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument(
        "--workers",
        type=int,
        default=min(6, os.cpu_count() or 4),
        help="data-loading processes; each costs RAM, so keep this low under WSL (default 6)",
    )
    p.add_argument(
        "--checkpoint-every",
        type=int,
        default=300,
        help="save a resumable checkpoint every N training steps (default 300, a few minutes)",
    )
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


def memory_gb() -> tuple[float, float] | None:
    """(total, available) system RAM in GB on Linux, else None."""
    try:
        info = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
        to_gb = lambda key: int(info[key].split()[0]) / 1024**2  # noqa: E731
        return to_gb("MemTotal"), to_gb("MemAvailable")
    except (OSError, KeyError, ValueError):
        return None


def git_state() -> dict | None:
    """Commit the artifacts were trained from, so the model card is reproducible."""
    try:
        run = lambda *a: subprocess.check_output(a, cwd=REPO, text=True).strip()  # noqa: E731
        return {
            "commit": run("git", "rev-parse", "HEAD"),
            "dirty": bool(run("git", "status", "--porcelain", "--untracked-files=no")),
        }
    except (OSError, subprocess.CalledProcessError):
        return None


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


def loader(rows, train: bool, args, cache: CachedRVL | None = None) -> DataLoader:
    dataset = CachedDataset(cache, rows, train) if cache else RVLDataset(rows, train)
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=train,
        num_workers=args.workers,
        pin_memory=args.device == "cuda",
        collate_fn=collate,
        persistent_workers=args.workers > 0,
        prefetch_factor=2 if args.workers > 0 else None,
    )


def amp_dtype(device: str) -> torch.dtype | None:
    """bfloat16 where the GPU supports it natively (no loss scaling needed), float16
    with a GradScaler on older cards such as Colab's T4, plain float32 on CPU."""
    if device != "cuda":
        return None
    native_bf16 = torch.cuda.is_bf16_supported(including_emulation=False)
    return torch.bfloat16 if native_bf16 else torch.float16


def autocast(device: str):
    dtype = amp_dtype(device)
    if dtype is None:
        return torch.autocast("cpu", enabled=False)
    return torch.autocast("cuda", dtype=dtype)


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
    # load_classifier() turns autograd off process-wide for the worker; don't inherit that.
    torch.set_grad_enabled(True)
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

    mem = memory_gb()
    if mem:
        print(f"RAM: {mem[0]:.1f} GB total, {mem[1]:.1f} GB available; {args.workers} workers")
        if mem[1] < 4 + 0.6 * args.workers:
            print(
                "WARNING: little free memory for this many workers. If the run stalls or is "
                "killed, lower --workers or raise WSL's memory limit (see RUNBOOK.md)."
            )

    git = git_state()  # taken before any artifact is written, so "dirty" means the code
    cache: CachedRVL | None = None
    if args.cache_dir:
        cache = CachedRVL(args.cache_dir, args.local_cache or args.work_dir / "cache-local")
        train_rows, val_rows, test_rows = (cache.rows(s) for s in ("train", "validation", "test"))
    else:
        train_rows = read_split(args.data_root, "train")
        val_rows = read_split(args.data_root, "val")
        test_rows = read_split(args.data_root, "test")

    def rel_of(ref) -> str:
        """Path of a page as written in the published label files."""
        return str(cache.paths[ref]) if cache else str(ref.relative_to(args.data_root / "images"))

    audit_dir = args.audit_dir or args.work_dir / "audit"
    clean_file = audit_dir / "clean_test.txt"
    clean_paths = set(clean_file.read_text().split()) if clean_file.exists() else None
    summary_file = audit_dir / "summary.json"
    audit_summary = (
        {
            k: v
            for k, v in json.loads(summary_file.read_text()).items()
            if k != "largest_duplicate_groups"
        }
        if summary_file.exists()
        else None
    )
    print(
        f"leakage-controlled test set: {len(clean_paths):,} pages"
        if clean_paths is not None
        else "no audit found (run scripts/audit_dataset.py): official test score only"
    )
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

    train_dl = loader(train_rows, True, args, cache)
    val_dl = loader(val_rows, False, args, cache)
    test_dl = loader(test_rows, False, args, cache)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = args.epochs * len(train_dl)
    # 5 % warm-up, but at least a few steps so very short (smoke) runs stay valid. The cap
    # keeps the warm-up phase shorter than the whole run (OneCycleLR divides by zero otherwise).
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=total_steps, pct_start=min(0.5, max(0.05, 3 / total_steps))
    )
    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    precision = amp_dtype(device)
    scaler = torch.amp.GradScaler(device, enabled=precision is torch.float16)
    precision_name = f"{str(precision).removeprefix('torch.')} autocast" if precision else "float32"
    print(f"Precision: {precision_name}")

    # ---- resume -------------------------------------------------------------
    ckpt_path, best_path = work / "last.pt", work / "best_state_dict.pt"
    steps_per_epoch = len(train_dl)
    global_step, best_val, elapsed = 0, 0.0, 0.0
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
        if "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])
        global_step, best_val, elapsed = ckpt["step"], ckpt["best_val"], ckpt.get("elapsed", 0.0)
        print(
            f"Resuming from step {global_step:,}/{total_steps:,} "
            f"(epoch {global_step // steps_per_epoch + 1}, best val top-1 {best_val:.4f})"
        )

    def save_checkpoint() -> None:
        """Write atomically so a crash mid-save cannot leave a truncated checkpoint."""
        tmp = ckpt_path.with_suffix(".tmp")
        torch.save(
            {
                "model": model.state_dict(),
                "opt": opt.state_dict(),
                "sched": sched.state_dict(),
                "scaler": scaler.state_dict(),
                "step": global_step,
                "best_val": best_val,
                "elapsed": elapsed + time.time() - t0,
                "config": run_config,
            },
            tmp,
        )
        os.replace(tmp, ckpt_path)

    # ---- train --------------------------------------------------------------
    # Checkpoints are written every --checkpoint-every steps as well as at each epoch
    # end. A run resumed mid-epoch finishes that epoch's remaining steps on a fresh
    # shuffle, so the step count and learning-rate schedule are unchanged.
    t0 = time.time()
    while global_step < total_steps:
        epoch = global_step // steps_per_epoch
        done_in_epoch = global_step % steps_per_epoch
        model.train()
        bar = tqdm(
            total=steps_per_epoch, initial=done_in_epoch, desc=f"epoch {epoch + 1}/{args.epochs}"
        )
        for x, y, _ in itertools.islice(train_dl, steps_per_epoch - done_in_epoch):
            x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            y = y.to(device, non_blocking=True)
            with autocast(device):
                loss = loss_fn(model(x), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            global_step += 1
            bar.update(1)
            bar.set_postfix(loss=f"{loss.item():.3f}")
            if global_step % args.checkpoint_every == 0 and global_step % steps_per_epoch:
                save_checkpoint()
        bar.close()
        p, yv, _ = evaluate(model, val_dl, device)
        val_top1 = topk(p, yv, 1)
        print(f"epoch {epoch + 1}: val top-1 {val_top1:.4f}  top-5 {topk(p, yv, 5):.4f}")
        if val_top1 > best_val:
            best_val = val_top1
            torch.save(model.state_dict(), best_path)
        save_checkpoint()
    train_minutes = (elapsed + time.time() - t0) / 60

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

    # ---- leakage-controlled test score ---------------------------------------
    leakage_controlled = None
    if clean_paths is not None:
        keep = torch.tensor([rel_of(test_rows[i][0]) in clean_paths for i in test_i.tolist()])
        if keep.any():
            leakage_controlled = {
                "top1": topk(test_p[keep], test_y[keep], 1),
                "top5": topk(test_p[keep], test_y[keep], 5),
                "n": int(keep.sum()),
                "policy": "official test pages that decode, have no pixel-identical twin in "
                "train or validation, are not in a conflicting-label duplicate group, and are "
                "the first occurrence of their pixels within test",
            }
            print(
                f"TEST (leakage-controlled) top-1 {leakage_controlled['top1']:.4f}  "
                f"top-5 {leakage_controlled['top5']:.4f}  (n={leakage_controlled['n']:,})"
            )

    # ---- golden set: expected outputs recorded on CPU in float32 --------------
    # CI replays on CPU; GPU / reduced-precision numbers would not match within 1e-6.
    # Golden pages must be real full-size TIFFs. Extracted data has them all; the cache keeps
    # the originals of 1 in 20 test pages (its "golden pool"), so the pick is made from those.
    # When an audit is available, only leakage-controlled pages are eligible.
    pool = {p.name for p in cache.golden_dir.iterdir()} if cache else None

    def golden_source(ref) -> Path | None:
        if cache is None:
            return ref
        name = safe_name(rel_of(ref))
        return cache.golden_dir / name if name in pool else None

    eligible = [
        j
        for j in range(len(test_y))
        if golden_source(test_rows[test_i[j].item()][0]) is not None
        and (clean_paths is None or rel_of(test_rows[test_i[j].item()][0]) in clean_paths)
    ]
    cand = torch.tensor(eligible, dtype=torch.long)
    golden_rows = [
        test_rows[test_i[cand[j]].item()] for j in pick_golden(test_p[cand], test_y[cand])
    ]
    if len(golden_rows) < 50 and not args.smoke:
        raise SystemExit(
            f"Only {len(golden_rows)} pages are eligible for the golden set (need 50)."
        )
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
        for n, (ref, true) in enumerate(golden_rows):
            fname = f"{n:02d}_{RVL_CDIP_CLASSES[true]}.tif"
            shutil.copy(golden_source(ref), golden_dir / fname)
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
            "precision": precision_name,
            "seed": args.seed,
            "best_val_top1": best_val,
            "train_minutes": round(train_minutes, 1),
        },
        "metrics": {
            "test": {"top1": test_top1, "top5": test_top5, "n": int(len(test_y))},
            "test_leakage_controlled": leakage_controlled,
            "golden": {
                "top1": correct1 / n_golden if n_golden else None,
                "top5": correct5 / n_golden if n_golden else None,
                "n": n_golden,
            },
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
        "dataset": {
            "name": "RVL-CDIP",
            "source": "huggingface.co/datasets/aharley/rvl_cdip (official 320k/40k/40k splits)",
            "license": "academic / research use only",
            "input": "224 px cache (scripts/rvl_cache.py)" if cache else "extracted TIFFs",
            "unreadable_images_dropped": cache.meta["decode_errors"] if cache else None,
            "audit": audit_summary,
        },
        "git": git,
        "trained_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
    }
    (models_dir / "model_card.json").write_text(json.dumps(card, indent=2) + "\n")

    print(f"\nArtifacts written to {models_dir} and {eval_dir}")
    gate = "PASSES" if test_top1 >= MIN_TEST_TOP1 else "is BELOW"
    print(f"Test top-1 {test_top1:.4f} {gate} the service quality gate ({MIN_TEST_TOP1}).")


if __name__ == "__main__":
    main()

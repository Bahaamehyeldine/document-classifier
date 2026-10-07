"""Audit RVL-CDIP for exact duplicates, cross-split leakage and unreadable scans.

The published splits are never modified. This script measures what is wrong with them and
writes, next to the raw dataset:

    manifest.csv      one row per image: split, path, label, pixel SHA-256, size, mode, error
    summary.json      counts per duplicate category, label conflicts, decode errors, ...
    clean_test.txt    test images kept in the leakage-controlled evaluation (one path per line)

`scripts/train_local.py` reports two test numbers when `clean_test.txt` exists: the official
RVL-CDIP test score (comparable to published work) and a leakage-controlled score.

Exact duplicates are found by hashing decoded pixels (size + 8-bit grayscale buffer), not file
bytes: two TIFFs can encode identical pixels with different compression or metadata. Near-
duplicates and "uniform-looking" pages are reported but never removed on sight; some genuinely
blank or dark pages belong to the task.

Leakage-controlled test set = official test images that
  - decode correctly,
  - have no pixel-identical twin in train or validation,
  - are not part of a duplicate group whose members carry conflicting labels, and
  - are the first occurrence of their pixels within the test split.

    python -m scripts.audit_dataset            # ~15 min on 8 workers; reuses an existing manifest
    python -m scripts.audit_dataset --force    # recompute from the images
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

from PIL import Image, ImageStat
from tqdm import tqdm

SPLIT_FILES = {"train": "train.txt", "validation": "val.txt", "test": "test.txt"}
NEAR_UNIFORM_STD = 5.0  # grey levels (0-255); a flag for inspection, never a deletion rule
FIELDS = ["split", "path", "label", "sha256", "width", "height", "mode", "std", "error"]


def read_labels(root: Path) -> list[tuple[str, str, int]]:
    rows = []
    for split, fname in SPLIT_FILES.items():
        for line in (root / "labels" / fname).read_text().splitlines():
            if line.strip():
                rel, label = line.split()
                rows.append((split, rel, int(label)))
    return rows


def hash_one(job: tuple[str, str, int, str]) -> dict:
    split, rel, label, root = job
    row = {
        "split": split,
        "path": rel,
        "label": label,
        "sha256": "",
        "width": "",
        "height": "",
        "mode": "",
        "std": "",
        "error": "",
    }
    try:
        with Image.open(Path(root) / "images" / rel) as img:
            img.seek(0)
            row["mode"] = img.mode
            gray = img.convert("L")
            w, h = gray.size
            digest = hashlib.sha256(w.to_bytes(4, "big") + h.to_bytes(4, "big"))
            digest.update(gray.tobytes())
            row.update(
                sha256=digest.hexdigest(),
                width=w,
                height=h,
                std=round(ImageStat.Stat(gray).stddev[0], 3),
            )
    except Exception as exc:  # corrupt TIFFs exist in RVL-CDIP; record, never crash
        row["error"] = f"{type(exc).__name__}: {exc}"[:200]
    return row


def build_manifest(root: Path, out: Path, workers: int) -> list[dict]:
    jobs = [(s, rel, label, str(root)) for s, rel, label in read_labels(root)]
    print(f"Hashing {len(jobs):,} images with {workers} workers...")
    rows: list[dict] = []
    with Pool(workers) as pool:
        for row in tqdm(pool.imap(hash_one, jobs, chunksize=128), total=len(jobs)):
            rows.append(row)
    tmp = out.with_suffix(".tmp")
    with tmp.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, out)
    return rows


def load_manifest(path: Path) -> list[dict]:
    with path.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        r["label"] = int(r["label"])
    return rows


def category(splits: set[str]) -> str:
    if len(splits) == 1:
        return f"within_{next(iter(splits))}"
    return "_".join(s for s in ("train", "validation", "test") if s in splits)


def analyse(rows: list[dict]) -> tuple[dict, list[str]]:
    ok = [r for r in rows if not r["error"]]
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in ok:
        groups[r["sha256"]].append(r)
    dup_groups = {h: g for h, g in groups.items() if len(g) > 1}

    per_category: dict[str, dict[str, int]] = defaultdict(lambda: {"groups": 0, "images": 0})
    conflicting_hashes = set()
    for h, g in dup_groups.items():
        cat = category({r["split"] for r in g})
        per_category[cat]["groups"] += 1
        per_category[cat]["images"] += len(g)
        if len({r["label"] for r in g}) > 1:
            conflicting_hashes.add(h)

    train_val_hashes = {r["sha256"] for r in ok if r["split"] in ("train", "validation")}
    leaked_test = [r for r in ok if r["split"] == "test" and r["sha256"] in train_val_hashes]
    conflicted_test = [r for r in ok if r["split"] == "test" and r["sha256"] in conflicting_hashes]

    seen_in_test: set[str] = set()
    clean: list[str] = []
    for r in ok:
        if r["split"] != "test":
            continue
        h = r["sha256"]
        if h in train_val_hashes or h in conflicting_hashes or h in seen_in_test:
            continue
        seen_in_test.add(h)
        clean.append(r["path"])

    errors = Counter(r["split"] for r in rows if r["error"])
    uniform = Counter(r["split"] for r in ok if float(r["std"]) < NEAR_UNIFORM_STD)
    modes = Counter(r["mode"] for r in ok)
    largest = sorted(dup_groups.values(), key=len, reverse=True)[:15]
    n_test = sum(1 for r in rows if r["split"] == "test")

    summary = {
        "images": {s: sum(1 for r in rows if r["split"] == s) for s in SPLIT_FILES},
        "decode_errors": dict(errors),
        "native_modes": dict(modes),
        "near_uniform_flagged": dict(uniform),
        "near_uniform_threshold_std": NEAR_UNIFORM_STD,
        "exact_duplicate_groups_total": len(dup_groups),
        "exact_duplicate_images_total": sum(len(g) for g in dup_groups.values()),
        "duplicates_by_category": dict(sorted(per_category.items())),
        "groups_with_conflicting_labels": len(conflicting_hashes),
        "test_images_with_twin_in_train_or_validation": len(leaked_test),
        "test_images_in_conflicting_label_groups": len(conflicted_test),
        "official_test_n": n_test,
        "leakage_controlled_test_n": len(clean),
        "largest_duplicate_groups": [
            {
                "size": len(g),
                "splits": dict(Counter(r["split"] for r in g)),
                "labels": sorted({r["label"] for r in g}),
                "example": g[0]["path"],
            }
            for g in largest
        ],
    }
    return summary, clean


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--data-root", type=Path, default=Path.home() / "data" / "rvl-cdip")
    p.add_argument("--out-dir", type=Path, default=Path.home() / "data" / "rvl-cdip-run" / "audit")
    p.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 4))
    p.add_argument("--force", action="store_true", help="recompute even if a manifest exists")
    p.add_argument(
        "--cache-dir",
        type=Path,
        help="read the hashes recorded while building the training cache (scripts/rvl_cache.py) "
        "instead of re-reading the images; takes seconds",
    )
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out_dir / "manifest.csv"
    t0 = time.time()
    if args.cache_dir:
        from scripts.rvl_cache import read_manifest

        rows = read_manifest(args.cache_dir)
        with manifest_path.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
    elif manifest_path.exists() and not args.force:
        print(f"Reusing {manifest_path} (pass --force to recompute)")
        rows = load_manifest(manifest_path)
    else:
        rows = build_manifest(args.data_root, manifest_path, args.workers)
    summary, clean = analyse(rows)
    summary["audited_in_minutes"] = round((time.time() - t0) / 60, 1)

    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.out_dir / "clean_test.txt").write_text("\n".join(clean) + "\n")
    print(
        json.dumps({k: v for k, v in summary.items() if k != "largest_duplicate_groups"}, indent=2)
    )
    print(f"\nWrote {manifest_path.name}, summary.json and clean_test.txt to {args.out_dir}")


if __name__ == "__main__":
    main()

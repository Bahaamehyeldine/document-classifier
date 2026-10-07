"""Resumable 224 px cache of RVL-CDIP, built once and reused by every training run.

Why it exists: Colab wipes its local disk whenever the runtime is interrupted, and the raw
archive is ~39 GB of full-size TIFFs. Re-downloading and re-decoding that after every
interruption is what makes training on Colab painful. Instead:

  1. `build` streams the archive once and stores each page, already resized by the service's own
     preprocessing (RESIZE_TRANSFORM), in compressed shards (~10k pages each) on Drive.
     A shard is written as soon as it is full, so an interruption loses at most a few
     seconds of work; re-running the same command skips everything already stored.
  2. Training reads these 224 x 224 uint8 pages (a few ms each) instead of decoding TIFFs.
     Because the cache stores exactly what the first half of the eval transform produces,
     the tensors the model sees are identical to what the inference worker computes.

Each page is also hashed on its full-size pixels while it is decoded, so the dataset audit
(`scripts/audit_dataset.py --cache-dir`) needs no second pass over the images.

Original bytes are kept for 1 in 20 test images (the "golden pool"): the golden set must be
real full-size TIFFs, and those are the only originals that survive in the cache.

    python -m scripts.rvl_cache build --out /content/drive/MyDrive/rvl-cache
    python -m scripts.rvl_cache build --out DIR --train-per-class 6000   # smaller, faster cache
    python -m scripts.rvl_cache info  --out DIR
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import io
import json
import os
import queue
import random
import subprocess
import tarfile
import threading
import time
import urllib.request
import zlib
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageStat
from torch.utils.data import Dataset
from torchvision.transforms import v2

from app.classifier.preprocessing import INPUT_SIZE, NORMALIZE_TRANSFORM, RESIZE_TRANSFORM

HF_BASE = "https://huggingface.co/datasets/aharley/rvl_cdip/resolve/main/data"
ARCHIVE_URL = f"{HF_BASE}/rvl-cdip.tar.gz"
SPLITS = ("train", "validation", "test")
SPLIT_CODE = {name: i for i, name in enumerate(SPLITS)}
LABEL_FILES = {"train": "train.txt", "validation": "val.txt", "test": "test.txt"}
SHARD_SIZE = 10_000
CHUNK_SIZE = 256  # members read ahead of the decoders; bounds memory to ~25 MB
GOLDEN_POOL_EVERY = 20
CACHE_VERSION = 1

# Same augmentation as the file-based training path, minus the resize the cache already did.
TRAIN_AUGMENT = v2.Compose(
    [
        v2.RandomAffine(degrees=2, translate=(0.02, 0.02), scale=(0.95, 1.05), fill=255),
        NORMALIZE_TRANSFORM,
    ]
)


# ---------------------------------------------------------------------------- labels


def read_labels(labels_dir: Path) -> dict[str, tuple[str, int]]:
    """{relative path: (split, class index)} from the three published label files."""
    labels_dir.mkdir(parents=True, exist_ok=True)
    out: dict[str, tuple[str, int]] = {}
    for split, fname in LABEL_FILES.items():
        path = labels_dir / fname
        if not path.exists():
            urllib.request.urlretrieve(f"{HF_BASE}/{fname}", path)  # noqa: S310
        for line in path.read_text().splitlines():
            if line.strip():
                rel, label = line.split()
                out[rel] = (split, int(label))
    return out


def select_wanted(
    labels: dict[str, tuple[str, int]], train_per_class: int | None, seed: int
) -> dict[str, tuple[str, int]]:
    """All validation/test pages, and either all training pages or a seeded per-class sample."""
    if not train_per_class:
        return dict(labels)
    wanted = {rel: v for rel, v in labels.items() if v[0] != "train"}
    by_class: dict[int, list[str]] = defaultdict(list)
    for rel, (split, label) in labels.items():
        if split == "train":
            by_class[label].append(rel)
    rng = random.Random(seed)
    for label, rels in sorted(by_class.items()):
        for rel in rng.sample(sorted(rels), min(train_per_class, len(rels))):
            wanted[rel] = ("train", label)
    return wanted


def in_golden_pool(rel: str) -> bool:
    return int(hashlib.md5(rel.encode()).hexdigest(), 16) % GOLDEN_POOL_EVERY == 0  # noqa: S324


def safe_name(rel: str) -> str:
    return rel.replace("/", "__")


# -------------------------------------------------------------------- decoding workers


def _init_worker() -> None:
    torch.set_num_threads(1)


def _prep(job: tuple[str, bytes]) -> dict:
    rel, data = job
    try:
        with Image.open(io.BytesIO(data)) as img:
            img.seek(0)
            mode = img.mode
            gray = img.convert("L")
            width, height = gray.size
            digest = hashlib.sha256(width.to_bytes(4, "big") + height.to_bytes(4, "big"))
            digest.update(gray.tobytes())
            std = ImageStat.Stat(gray).stddev[0]
            page = RESIZE_TRANSFORM(gray.convert("RGB")).as_subclass(torch.Tensor)[0].numpy()
        return {
            "rel": rel,
            "x": page,
            "sha256": digest.hexdigest(),
            "width": width,
            "height": height,
            "std": std,
            "mode": mode,
            "error": "",
        }
    except Exception as exc:  # corrupt TIFFs exist in RVL-CDIP; record them, never crash
        return {
            "rel": rel,
            "x": np.zeros((INPUT_SIZE, INPUT_SIZE), np.uint8),
            "sha256": "",
            "width": 0,
            "height": 0,
            "std": 0.0,
            "mode": "",
            "error": f"{type(exc).__name__}: {exc}"[:200],
        }


# ------------------------------------------------------------------------ build


class _Buffer:
    KEYS = ("x", "path", "split", "label", "sha256", "width", "height", "std", "mode", "error")

    def __init__(self) -> None:
        self.rows: dict[str, list] = {k: [] for k in self.KEYS}

    def __len__(self) -> int:
        return len(self.rows["path"])

    def add(self, res: dict, split: str, label: int) -> None:
        r = self.rows
        r["x"].append(res["x"])
        r["path"].append(res["rel"])
        r["split"].append(SPLIT_CODE[split])
        r["label"].append(label)
        for k in ("sha256", "width", "height", "std", "mode", "error"):
            r[k].append(res[k])

    def write(self, out_dir: Path) -> None:
        if not len(self):
            return
        index = 1 + max(
            (int(p.stem.split("_")[1]) for p in out_dir.glob("shard_*.npz")), default=-1
        )
        r = self.rows
        final = out_dir / f"shard_{index:05d}.npz"
        tmp = out_dir / f".shard_{index:05d}.tmp.npz"
        np.savez_compressed(
            tmp,
            x=np.stack(r["x"]).astype(np.uint8),
            path=np.array(r["path"]),
            split=np.array(r["split"], np.uint8),
            label=np.array(r["label"], np.int16),
            sha256=np.array(r["sha256"]),
            width=np.array(r["width"], np.int32),
            height=np.array(r["height"], np.int32),
            std=np.array(r["std"], np.float32),
            mode=np.array(r["mode"]),
            error=np.array(r["error"]),
        )
        os.replace(tmp, final)
        self.rows = {k: [] for k in self.KEYS}


def _member_rel(name: str, wanted: dict) -> str | None:
    if name.startswith("./"):
        name = name[2:]
    if name in wanted:
        return name
    if name.startswith("images/") and name[7:] in wanted:
        return name[7:]
    return None


def _load_done(out_dir: Path) -> set[str]:
    done: set[str] = set()
    for shard in sorted(out_dir.glob("shard_*.npz")):
        with np.load(shard, allow_pickle=False) as z:
            done.update(z["path"].tolist())
    return done


def _read_chunks(tar, wanted, done, q, chunk_size, stop) -> None:
    """Reader thread: pull wanted members off the (sequential) stream into a bounded queue."""
    try:
        batch: list[tuple[str, bytes]] = []
        for member in tar:
            if stop.is_set():
                break
            if not member.isfile():
                continue
            rel = _member_rel(member.name, wanted)
            if rel is None or rel in done:
                continue
            fh = tar.extractfile(member)
            if fh is None:
                continue
            batch.append((rel, fh.read()))
            if len(batch) >= chunk_size:
                q.put(batch)
                batch = []
        if batch and not stop.is_set():
            q.put(batch)
        q.put(None)
    except BaseException as exc:  # noqa: BLE001 - handed to the main thread
        q.put(exc)


class _Stream:
    """The archive as a sequential tar stream: a local file, or `curl` piped from the URL."""

    def __init__(self, archive: Path | None, url: str):
        self.proc = None
        if archive is not None:
            self.tar = tarfile.open(archive, mode="r|gz")  # noqa: SIM115
        else:
            # No curl --retry here: retrying mid-stream would splice duplicate bytes into the
            # tar. If the connection drops the stream ends and the caller restarts the pass,
            # skipping everything already stored.
            cmd = ["curl", "-sL", "--fail", "--http1.1", "--speed-limit", "1024"]
            cmd += ["--speed-time", "60", url]
            self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)  # noqa: S603
            self.tar = tarfile.open(fileobj=self.proc.stdout, mode="r|gz")  # noqa: SIM115

    def close(self) -> None:
        try:
            self.tar.close()
        finally:
            if self.proc is not None:
                self.proc.kill()
                self.proc.wait()


def build_cache(
    out_dir: Path,
    *,
    archive: Path | None = None,
    url: str = ARCHIVE_URL,
    labels_dir: Path | None = None,
    train_per_class: int | None = None,
    seed: int = 42,
    workers: int | None = None,
    shard_size: int = SHARD_SIZE,
    chunk_size: int = CHUNK_SIZE,
    limit: int | None = None,
    max_attempts: int = 30,
) -> dict:
    """Build (or resume) the cache. Returns the contents of meta.json."""
    out_dir.mkdir(parents=True, exist_ok=True)
    golden_dir = out_dir / "golden_pool"
    golden_dir.mkdir(exist_ok=True)
    workers = workers or min(8, os.cpu_count() or 2)

    config = {"version": CACHE_VERSION, "train_per_class": train_per_class, "seed": seed}
    meta_path = out_dir / "meta.json"
    if meta_path.exists():
        saved = json.loads(meta_path.read_text())
        if saved.get("config") != config:
            raise SystemExit(
                f"{out_dir} was built with {saved.get('config')}, not {config}. "
                "Use the same options to resume, or a new --out directory."
            )
    meta_path.write_text(json.dumps({"config": config, "complete": False}, indent=2))

    wanted = select_wanted(read_labels(labels_dir or out_dir / "labels"), train_per_class, seed)
    done = _load_done(out_dir)
    print(f"{len(wanted):,} pages wanted, {len(done):,} already cached")

    new_pages = 0
    attempt = 0
    with Pool(workers, initializer=_init_worker) as pool:
        while len(done) < len(wanted) and (limit is None or new_pages < limit):
            attempt += 1
            buf = _Buffer()
            stream = None
            stop = threading.Event()
            reached_end = False  # the stream was read to its end (or stopped on purpose)
            try:
                stream = _Stream(archive, url)
                q: queue.Queue = queue.Queue(maxsize=3)
                reader = threading.Thread(
                    target=_read_chunks,
                    args=(stream.tar, wanted, done, q, chunk_size, stop),
                    daemon=True,
                )
                reader.start()
                started = time.time()
                while True:
                    item = q.get()
                    if item is None:
                        reached_end = True
                        break
                    if isinstance(item, BaseException):
                        raise item
                    for (rel, data), res in zip(
                        item, pool.map(_prep, item, chunksize=8), strict=True
                    ):
                        split, label = wanted[rel]
                        buf.add(res, split, label)
                        done.add(rel)
                        new_pages += 1
                        if split == "test" and not res["error"] and in_golden_pool(rel):
                            target = golden_dir / safe_name(rel)
                            if not target.exists():
                                tmp = target.with_suffix(".part")
                                tmp.write_bytes(data)
                                os.replace(tmp, target)
                    if len(buf) >= shard_size:
                        buf.write(out_dir)
                        rate = new_pages / max(time.time() - started, 1e-6)
                        print(f"{len(done):,}/{len(wanted):,} cached ({rate:.0f} pages/s)")
                    if limit is not None and new_pages >= limit:
                        stop.set()
                        while True:  # let the reader finish and exit
                            tail = q.get()
                            if tail is None or isinstance(tail, BaseException):
                                break
                        reached_end = True
                        break
            except (tarfile.TarError, EOFError, OSError, zlib.error, ConnectionError) as exc:
                if stream is not None and stream.proc is not None and stream.proc.poll() == 22:
                    raise RuntimeError(f"Archive request failed (HTTP error): {url}") from exc
                print(f"stream interrupted ({type(exc).__name__}: {exc}); resuming")
                if attempt >= max_attempts:
                    raise
                time.sleep(min(60, 5 * attempt))
            finally:
                stop.set()
                buf.write(out_dir)  # keep whatever was decoded, even if the stream broke
                if stream is not None:
                    stream.close()
            if reached_end or archive is not None:
                break  # only a broken stream is retried; pages absent from the archive are not

    missing = len(wanted) - len(done)
    meta = {
        "config": config,
        "complete": missing == 0,
        "missing_from_archive": missing,
        "pages": {s: 0 for s in SPLITS},
        "decode_errors": {s: 0 for s in SPLITS},
        "golden_pool": len(list(golden_dir.iterdir())),
    }
    for shard in sorted(out_dir.glob("shard_*.npz")):
        with np.load(shard, allow_pickle=False) as z:
            for code, split in enumerate(SPLITS):
                in_split = z["split"] == code
                bad = in_split & (z["error"] != "")
                meta["pages"][split] += int(in_split.sum() - bad.sum())
                meta["decode_errors"][split] += int(bad.sum())
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    return meta


# ------------------------------------------------------------------------- reading


class CachedRVL:
    """Read access to a finished cache. Shards are unpacked once to `local_dir` as .npy so
    pages can be memory-mapped; delete `local_dir` any time, it is rebuilt from the cache."""

    def __init__(self, cache_dir: Path, local_dir: Path):
        self.cache_dir = Path(cache_dir)
        meta = json.loads((self.cache_dir / "meta.json").read_text())
        if not meta.get("complete"):
            raise SystemExit(
                f"{cache_dir} is incomplete ({meta.get('missing_from_archive', '?')} pages "
                "missing). Re-run `python -m scripts.rvl_cache build` with the same options."
            )
        self.meta = meta
        local_dir = Path(local_dir)
        local_dir.mkdir(parents=True, exist_ok=True)
        cols: dict[str, list[np.ndarray]] = defaultdict(list)
        self._x: list[np.ndarray] = []
        self._starts = [0]
        for shard in sorted(self.cache_dir.glob("shard_*.npz")):
            npy = local_dir / f"{shard.stem}.npy"
            with np.load(shard, allow_pickle=False) as z:
                if not npy.exists():
                    tmp = npy.with_suffix(".tmp.npy")
                    np.save(tmp, z["x"])
                    os.replace(tmp, npy)
                for k in ("path", "split", "label", "sha256", "width", "height", "std", "mode"):
                    cols[k].append(z[k])
                cols["error"].append(z["error"])
            self._x.append(np.load(npy, mmap_mode="r"))
            self._starts.append(self._starts[-1] + len(cols["path"][-1]))
        self.paths = np.concatenate(cols["path"])
        self.split = np.concatenate(cols["split"])
        self.label = np.concatenate(cols["label"])
        self.sha256 = np.concatenate(cols["sha256"])
        self.width = np.concatenate(cols["width"])
        self.height = np.concatenate(cols["height"])
        self.std = np.concatenate(cols["std"])
        self.mode = np.concatenate(cols["mode"])
        self.error = np.concatenate(cols["error"])
        self.golden_dir = self.cache_dir / "golden_pool"

    def rows(self, split: str) -> list[tuple[int, int]]:
        """[(global index, class)] for the readable pages of a split."""
        idx = np.nonzero((self.split == SPLIT_CODE[split]) & (self.error == ""))[0]
        return [(int(i), int(self.label[i])) for i in idx]

    def page(self, gi: int) -> np.ndarray:
        k = bisect.bisect_right(self._starts, gi) - 1
        return self._x[k][gi - self._starts[k]]

    def golden_original(self, gi: int) -> Path | None:
        path = self.golden_dir / safe_name(str(self.paths[gi]))
        return path if path.is_file() else None


def read_manifest(cache_dir: Path) -> list[dict]:
    """Per-page metadata straight from the shards, without unpacking any pixels."""
    rows: list[dict] = []
    for shard in sorted(Path(cache_dir).glob("shard_*.npz")):
        with np.load(shard, allow_pickle=False) as z:
            cols = {k: z[k].tolist() for k in z.files if k != "x"}
        for i in range(len(cols["path"])):
            rows.append(
                {
                    "split": SPLITS[cols["split"][i]],
                    "path": cols["path"][i],
                    "label": int(cols["label"][i]),
                    "sha256": cols["sha256"][i],
                    "width": cols["width"][i],
                    "height": cols["height"][i],
                    "mode": cols["mode"][i],
                    "std": cols["std"][i],
                    "error": cols["error"][i],
                }
            )
    return rows


class CachedDataset(Dataset):
    """(normalised 3 x 224 x 224 tensor, class, row number) from cached uint8 pages."""

    def __init__(self, cache: CachedRVL, rows: list[tuple[int, int]], train: bool):
        self.cache, self.rows, self.train = cache, rows, train

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        gi, label = self.rows[i]
        page = torch.from_numpy(np.array(self.cache.page(gi)))  # copy out of the memory map
        x = page.unsqueeze(0).expand(3, -1, -1).contiguous()  # grayscale replicated to RGB
        x = TRAIN_AUGMENT(x) if self.train else NORMALIZE_TRANSFORM(x)
        return x, label, i


# ---------------------------------------------------------------------------- CLI


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="build or resume the cache")
    b.add_argument("--out", type=Path, required=True)
    b.add_argument("--archive", type=Path, help="local rvl-cdip.tar.gz instead of downloading")
    b.add_argument("--url", default=ARCHIVE_URL)
    b.add_argument("--labels-dir", type=Path)
    b.add_argument("--train-per-class", type=int, help="cache N training pages per class only")
    b.add_argument("--seed", type=int, default=42)
    b.add_argument("--workers", type=int)
    b.add_argument("--limit", type=int, help="stop after N new pages (for testing)")
    i = sub.add_parser("info", help="summarise a cache")
    i.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    if args.cmd == "info":
        print((args.out / "meta.json").read_text())
        return
    meta = build_cache(
        args.out,
        archive=args.archive,
        url=args.url,
        labels_dir=args.labels_dir,
        train_per_class=args.train_per_class,
        seed=args.seed,
        workers=args.workers,
        limit=args.limit,
    )
    print(json.dumps(meta, indent=2))
    if not meta["complete"]:
        raise SystemExit("CACHE INCOMPLETE: run the same command again (it resumes)")
    print("CACHE COMPLETE")


if __name__ == "__main__":
    main()

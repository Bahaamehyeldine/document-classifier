"""The training cache must be resumable and feed the model exactly what the service computes."""

import io
import tarfile

import numpy as np
import pytest
import torch
from PIL import Image

from app.classifier.preprocessing import load_image, to_tensor
from scripts import rvl_cache
from scripts.audit_dataset import analyse
from scripts.rvl_cache import (
    CachedDataset,
    CachedRVL,
    build_cache,
    in_golden_pool,
    read_manifest,
    safe_name,
)

N_PER_SPLIT = {"train": 24, "val": 8, "test": 40}


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    """A tiny RVL-CDIP lookalike: tar.gz of TIFFs under images/, three label files, one corrupt
    scan, and one test page that is pixel-identical to a training page."""
    root = tmp_path_factory.mktemp("rvl")
    rng = np.random.default_rng(0)
    labels_dir = root / "labels"
    labels_dir.mkdir()
    originals: dict[str, bytes] = {}
    lines: dict[str, list[str]] = {s: [] for s in N_PER_SPLIT}
    n = 0
    for split, count in N_PER_SPLIT.items():
        for i in range(count):
            rel = f"images{'abcd'[i % 4]}/{'abcd'[i % 4]}/x/y/{split}{i}/{n:04d}.tif"
            h, w = int(rng.integers(30, 70)), int(rng.integers(30, 70))
            arr = rng.integers(0, 255, (h, w), dtype=np.uint8)
            buf = io.BytesIO()
            Image.fromarray(arr).save(buf, format="TIFF")
            originals[rel] = buf.getvalue()
            lines[split].append(f"{rel} {i % 4}")
            n += 1
    bad = "imagesz/z/x/y/bad/9999.tif"
    originals[bad] = b"this is not a tiff"
    lines["test"].append(f"{bad} 3")
    for split, fname in (("train", "train.txt"), ("val", "val.txt"), ("test", "test.txt")):
        (labels_dir / fname).write_text("\n".join(lines[split]) + "\n")

    archive = root / "rvl-cdip.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for rel, data in originals.items():
            info = tarfile.TarInfo(f"images/{rel}")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return {"archive": archive, "labels": labels_dir, "originals": originals, "bad": bad}


def build(ds, out, **kw):
    kw.setdefault("workers", 2)
    kw.setdefault("shard_size", 10)
    kw.setdefault("chunk_size", 5)
    return build_cache(out, archive=ds["archive"], labels_dir=ds["labels"], **kw)


def test_full_build_counts_and_corrupt_scan(dataset, tmp_path):
    meta = build(dataset, tmp_path / "c")
    assert meta["complete"]
    assert meta["pages"] == {"train": 24, "validation": 8, "test": 40}
    assert meta["decode_errors"] == {"train": 0, "validation": 0, "test": 1}
    cache = CachedRVL(tmp_path / "c", tmp_path / "local")
    assert len(cache.rows("test")) == 40  # the corrupt page is cached but not served
    manifest = read_manifest(tmp_path / "c")
    assert len(manifest) == 73
    assert [r["path"] for r in manifest if r["error"]] == [dataset["bad"]]
    # The audit can run from the cache alone: planted twin, no images needed.
    twin = dict(next(r for r in manifest if r["split"] == "train" and not r["error"]))
    twin.update(split="test", path="images/twin.tif")
    summary, clean = analyse(manifest + [twin])
    assert summary["test_images_with_twin_in_train_or_validation"] == 1
    assert "images/twin.tif" not in clean and dataset["bad"] not in clean
    assert summary["decode_errors"] == {"test": 1}


def test_interrupted_build_resumes_without_duplicates_or_gaps(dataset, tmp_path):
    out = tmp_path / "c"
    first = build(dataset, out, limit=7)
    assert not first["complete"]
    done_after_first = rvl_cache._load_done(out)
    assert 0 < len(done_after_first) < 73

    second = build(dataset, out)  # same command again, as after a Colab disconnect
    assert second["complete"]
    paths = [p for shard in sorted(out.glob("shard_*.npz")) for p in np.load(shard)["path"]]
    assert len(paths) == len(set(paths)) == 73

    reference = build(dataset, tmp_path / "ref")
    assert reference["pages"] == second["pages"]
    a, b = CachedRVL(out, tmp_path / "la"), CachedRVL(tmp_path / "ref", tmp_path / "lb")
    by_path_a = {str(p): a.page(i).tolist() for i, p in enumerate(a.paths)}
    by_path_b = {str(p): b.page(i).tolist() for i, p in enumerate(b.paths)}
    assert by_path_a == by_path_b


def test_cached_tensors_are_identical_to_what_the_service_computes(dataset, tmp_path):
    out = tmp_path / "c"
    build(dataset, out)
    cache = CachedRVL(out, tmp_path / "local")
    ds = CachedDataset(cache, cache.rows("validation"), train=False)
    originals_dir = tmp_path / "orig"
    originals_dir.mkdir()
    for k in range(len(ds)):
        x, label, _ = ds[k]
        rel = str(cache.paths[cache.rows("validation")[k][0]])
        (originals_dir / "page.tif").write_bytes(dataset["originals"][rel])
        expected = to_tensor(load_image(originals_dir / "page.tif"))
        assert x.dtype == torch.float32 and x.shape == (3, 224, 224)
        assert torch.equal(x, expected), rel
        assert label == cache.label[cache.rows("validation")[k][0]]


def test_training_view_is_augmented_but_well_formed(dataset, tmp_path):
    build(dataset, tmp_path / "c")
    cache = CachedRVL(tmp_path / "c", tmp_path / "local")
    ds = CachedDataset(cache, cache.rows("train"), train=True)
    x, _, i = ds[0]
    assert x.shape == (3, 224, 224) and x.dtype == torch.float32 and i == 0
    assert torch.isfinite(x).all()


def test_golden_pool_keeps_original_tiffs_of_test_pages_only(dataset, tmp_path):
    out = tmp_path / "c"
    build(dataset, out)
    expected = {
        safe_name(rel)
        for rel in dataset["originals"]
        if rel.split("/")[-2].startswith("test") and in_golden_pool(rel)
    }
    assert {p.name for p in (out / "golden_pool").iterdir()} == expected
    for p in (out / "golden_pool").iterdir():
        assert p.read_bytes() in dataset["originals"].values()


def test_train_per_class_samples_training_only(dataset, tmp_path):
    meta = build(dataset, tmp_path / "c", train_per_class=2)
    assert meta["pages"]["train"] == 8  # 2 per class x 4 classes
    assert meta["pages"]["validation"] == 8 and meta["pages"]["test"] == 40
    again = build(dataset, tmp_path / "d", train_per_class=2)
    paths = lambda d: {str(p) for s in sorted(d.glob("shard_*.npz")) for p in np.load(s)["path"]}  # noqa: E731
    assert paths(tmp_path / "c") == paths(tmp_path / "d") and again["complete"]  # seeded


def test_resuming_with_different_options_is_refused(dataset, tmp_path):
    build(dataset, tmp_path / "c", train_per_class=2)
    with pytest.raises(SystemExit, match="was built with"):
        build(dataset, tmp_path / "c")


def test_unfinished_cache_cannot_be_used_for_training(dataset, tmp_path):
    build(dataset, tmp_path / "c", limit=5)
    with pytest.raises(SystemExit, match="incomplete"):
        CachedRVL(tmp_path / "c", tmp_path / "local")


def test_training_script_runs_end_to_end_from_the_cache_and_resumes(dataset, tmp_path, monkeypatch):
    """CPU smoke run of scripts/train.py: cache in, artifacts + model card out."""
    import json
    import sys

    from scripts import train as train_local

    cache_dir, work = tmp_path / "cache", tmp_path / "work"
    build(dataset, cache_dir)
    audit = work / "audit"
    audit.mkdir(parents=True)
    summary, clean = analyse(read_manifest(cache_dir))
    (audit / "summary.json").write_text(json.dumps(summary))
    (audit / "clean_test.txt").write_text("\n".join(clean) + "\n")

    argv = [
        "train_local", "--smoke", "--device", "cpu", "--no-pretrained", "--workers", "0",
        "--batch-size", "8", "--cache-dir", str(cache_dir), "--work-dir", str(work),
        "--local-cache", str(tmp_path / "local"),
    ]  # fmt: skip
    monkeypatch.setattr(sys, "argv", argv)
    train_local.main()

    artifacts = work / "smoke" / "artifacts"
    card = json.loads((artifacts / "models" / "model_card.json").read_text())
    assert card["metrics"]["test"]["n"] == 40
    assert card["metrics"]["test_leakage_controlled"]["n"] == len(clean)
    assert card["dataset"]["unreadable_images_dropped"]["test"] == 1
    assert card["dataset"]["audit"]["official_test_n"] == 41
    assert card["dataset"]["input"].startswith("224 px cache")
    from app.classifier.artifacts import sha256_file

    assert card["sha256"] == sha256_file(artifacts / "models" / "classifier.pt")
    for entry in json.loads((artifacts / "eval" / "golden_expected.json").read_text())["images"]:
        assert (artifacts / "eval" / "golden_images" / entry["file"]).is_file()

    first_sha = card["sha256"]  # a second run resumes from the finished checkpoint
    train_local.main()
    assert json.loads((artifacts / "models" / "model_card.json").read_text())["sha256"] == first_sha

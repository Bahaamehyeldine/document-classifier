"""Measure the README latency budgets (p95) against a running compose stack.

    docker compose up -d                       # api, worker, sftp-ingest and the infrastructure
    LATENCY_PASSWORD=... python -m scripts.latency_check --email admin@example.com

What it measures, and how:
  api cached reads    GET /me, /batches, /batches/{id}, /predictions/recent; the first request
                      fills the cache, the rest must answer X-FastAPI-Cache: HIT
  api uncached reads  the same endpoints with `Cache-Control: no-cache`, which makes
                      fastapi-cache skip its cache (header MISS), so the database is queried
  inference           ConvNeXt on the 50 committed golden pages, in this process on the CPU
                      (the worker does exactly this call per document)
  end to end          drop a single TIFF over SFTP, then time until GET /batches/{id} shows the
                      batch as done with its prediction; includes the ingest poll interval

Times are client side over localhost, so they include HTTP and JSON but no real network.
Exits non-zero when a budget is exceeded (use --no-check to only print).
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
import uuid

import httpx
import paramiko

BUDGETS_MS = {"cached": 50.0, "uncached": 200.0, "inference": 1000.0, "end_to_end": 10_000.0}


def p95(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]


def summary(name: str, values: list[float]) -> dict:
    ordered = sorted(values)
    return {
        "name": name,
        "n": len(values),
        "p50": ordered[len(ordered) // 2],
        "p95": p95(values),
        "max": ordered[-1],
    }


def login(client: httpx.Client, email: str, password: str) -> None:
    r = client.post("/auth/jwt/login", data={"username": email, "password": password})
    r.raise_for_status()
    client.headers["Authorization"] = f"Bearer {r.json()['access_token']}"


def timed_get(client: httpx.Client, path: str, headers: dict | None = None) -> tuple[float, str]:
    start = time.perf_counter()
    r = client.get(path, headers=headers)
    elapsed = (time.perf_counter() - start) * 1000
    r.raise_for_status()
    return elapsed, r.headers.get("x-fastapi-cache", "none")


def read_paths(client: httpx.Client) -> list[str]:
    batches = client.get("/batches").json()
    paths = ["/me", "/batches", "/predictions/recent"]
    if batches:
        paths.append(f"/batches/{batches[0]['id']}")
    return paths


def measure_reads(client: httpx.Client, reads: int) -> tuple[list[float], list[float]]:
    cached, uncached = [], []
    for path in read_paths(client):
        timed_get(client, path)  # fill the cache
        for _ in range(reads):
            ms, state = timed_get(client, path)
            if state != "HIT":
                raise SystemExit(f"{path}: expected a cache HIT, got {state!r}")
            cached.append(ms)
        for _ in range(reads):
            ms, state = timed_get(client, path, {"Cache-Control": "no-cache"})
            if state == "HIT":
                raise SystemExit(f"{path}: no-cache request was served from the cache")
            uncached.append(ms)
    return cached, uncached


def measure_inference(images: int = 50) -> list[float]:
    from app.classifier.eval.golden import GOLDEN_IMAGES
    from app.classifier.model import load_classifier
    from app.classifier.preprocessing import load_image

    clf = load_classifier()
    pages = [load_image(p) for p in sorted(GOLDEN_IMAGES.glob("*.tif"))[:images]]
    for page in pages[:3]:  # warm-up: first calls allocate buffers and pick kernels
        clf.predict_image(page)
    times = []
    for page in pages:
        start = time.perf_counter()
        clf.predict_image(page)
        times.append((time.perf_counter() - start) * 1000)
    return times


def measure_end_to_end(client: httpx.Client, args: argparse.Namespace) -> list[float]:
    from scripts.sftp_upload import sample_page

    times = []
    for i in range(args.drops):
        known = {
            b["id"] for b in client.get("/batches", headers={"Cache-Control": "no-cache"}).json()
        }
        transport = paramiko.Transport((args.sftp_host, args.sftp_port))
        transport.connect(username=args.sftp_user, password=os.environ["SFTP_PASSWORD"])
        sftp = paramiko.SFTPClient.from_transport(transport)
        with sftp.open(f"upload/latency_{uuid.uuid4().hex[:8]}_{i}.tif", "wb") as fh:
            fh.write(sample_page())
        transport.close()
        start = time.perf_counter()  # the drop is complete; the clock starts now

        deadline = start + 60
        batch_id = None
        while time.perf_counter() < deadline:
            fresh = client.get("/batches", headers={"Cache-Control": "no-cache"}).json()
            new = [b for b in fresh if b["id"] not in known]
            if new:
                batch_id = new[0]["id"]
                break
            time.sleep(0.1)
        while batch_id and time.perf_counter() < deadline:
            r = client.get(f"/batches/{batch_id}", headers={"Cache-Control": "no-cache"})
            if r.json()["status"] == "done":
                times.append((time.perf_counter() - start) * 1000)
                break
            time.sleep(0.1)
        else:
            raise SystemExit(f"drop {i}: batch not done within 60 s")
        print(f"  drop {i + 1}/{args.drops}: {times[-1] / 1000:.2f} s")
    return times


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--api", default=f"http://localhost:{os.environ.get('API_PORT', '8000')}")
    p.add_argument("--email", required=True, help="a user who can read batches (e.g. admin)")
    p.add_argument("--reads", type=int, default=100, help="requests per endpoint and mode")
    p.add_argument("--drops", type=int, default=10, help="SFTP drops for the end-to-end figure")
    p.add_argument("--sftp-host", default="localhost")
    p.add_argument("--sftp-port", type=int, default=int(os.environ.get("SFTP_PORT", "2222")))
    p.add_argument("--sftp-user", default="scanner")
    p.add_argument("--no-check", action="store_true", help="print only; always exit 0")
    args = p.parse_args()
    if "LATENCY_PASSWORD" not in os.environ or "SFTP_PASSWORD" not in os.environ:
        raise SystemExit("Set LATENCY_PASSWORD (the user's password) and SFTP_PASSWORD (scanner).")

    results = {}
    with httpx.Client(base_url=args.api, timeout=30) as client:
        login(client, args.email, os.environ["LATENCY_PASSWORD"])
        print("api reads (cached and uncached)...")
        results["cached"], results["uncached"] = measure_reads(client, args.reads)
        print("end to end (SFTP drop -> done)...")
        results["end_to_end"] = measure_end_to_end(client, args)
    # Last on purpose: importing torch and running the model leaves busy OpenMP worker threads
    # in this process, which inflated the HTTP timings above by ~10x when it ran first.
    print("inference (CPU, golden pages)...")
    results["inference"] = measure_inference()

    failed = False
    print("\n| Path | n | p50 | p95 | max | budget (p95) |\n|---|---|---|---|---|---|")
    for key in ("cached", "uncached", "inference", "end_to_end"):
        s = summary(key, results[key])
        budget = BUDGETS_MS[key]
        ok = s["p95"] <= budget
        failed |= not ok
        print(
            f"| {key} | {s['n']} | {s['p50']:.1f} ms | {s['p95']:.1f} ms | {s['max']:.1f} ms | "
            f"< {budget:.0f} ms {'OK' if ok else 'EXCEEDED'} |"
        )
    return 0 if args.no_check or not failed else 1


if __name__ == "__main__":
    sys.exit(main())

"""Ingestion -> worker -> API, relabeling rules and cache invalidation, on real Postgres + Redis."""

import io
from dataclasses import dataclass

from PIL import Image

from app.classifier.model import Prediction
from app.db.session import sessionmaker
from app.services import batch_service
from app.services.batch_service import IncomingFile
from app.workers import inference
from app.workers.sftp_ingest import Poller
from tests.api.conftest import make_user


def tiff_bytes(color=255) -> bytes:
    buf = io.BytesIO()
    Image.new("L", (300, 400), color).save(buf, format="TIFF")
    return buf.getvalue()


class StubClassifier:
    card = {"sha256": "f" * 64, "backbone": "convnext_tiny"}

    def __init__(self, label="invoice", confidence=0.93):
        self.label, self.confidence = label, confidence

    def predict_image(self, img):
        top5 = [
            (self.label, self.confidence),
            ("form", 0.03),
            ("letter", 0.02),
            ("memo", 0.01),
            ("budget", 0.01),
        ]
        return Prediction(label=self.label, label_index=11, confidence=self.confidence, top5=top5)


@dataclass
class Ctx:
    blob: object
    invalidator: object


async def ingest(client, fakes, n=2, data=None):
    files = [IncomingFile(f"scan_{i}.tif", data or tiff_bytes()) for i in range(n)]
    async with sessionmaker()() as session:
        batch = await batch_service.ingest(
            session,
            client.app.state.invalidator,
            fakes["blob"],
            fakes["queue"],
            files,
            "rid-ingest",
        )
    return batch.id


async def run_worker(client, fakes, classifier=None):
    ctx = Ctx(fakes["blob"], client.app.state.invalidator)
    for doc_id, rid in list(fakes["queue"].jobs):
        try:
            await inference.classify(ctx, classifier or StubClassifier(), doc_id, rid)
        except inference.PermanentJobError as exc:
            await inference.fail(ctx, doc_id, str(exc), rid)
    fakes["queue"].jobs.clear()


async def test_ingest_queues_one_job_per_document_with_request_id(client, fakes):
    admin, _ = await make_user(client, "admin@example.com", "admin")
    before = await client.get("/batches", headers=admin)
    assert before.json() == []
    assert (await client.get("/batches", headers=admin)).headers["X-FastAPI-Cache"] == "HIT"

    batch_id = await ingest(client, fakes, n=3)

    assert [rid for _, rid in fakes["queue"].jobs] == ["rid-ingest"] * 3
    listed = (await client.get("/batches", headers=admin)).json()  # cache was invalidated
    assert listed[0]["id"] == str(batch_id) and listed[0]["status"] == "received"
    assert listed[0]["document_count"] == 3
    assert len([k for k in fakes["blob"].objects if k.startswith(f"raw/{batch_id}/")]) == 3


async def test_worker_classifies_and_batch_moves_to_done(client, fakes):
    auditor, _ = await make_user(client, "aud@example.com", "auditor")
    batch_id = await ingest(client, fakes, n=2)
    await run_worker(client, fakes)

    detail = (await client.get(f"/batches/{batch_id}", headers=auditor)).json()
    assert detail["status"] == "done"
    for doc in detail["documents"]:
        assert doc["status"] == "classified"
        assert (
            doc["prediction"]["label"] == "invoice" and doc["prediction"]["needs_review"] is False
        )
    assert any(k.startswith(f"overlays/{batch_id}/") for k in fakes["blob"].objects)

    changes = (await client.get("/audit?action=batch.status_changed", headers=auditor)).json()
    transitions = [(e["details"]["from"], e["details"]["to"]) for e in reversed(changes)]
    assert transitions == [("received", "processing"), ("processing", "done")]
    assert all(e["request_id"] == "rid-ingest" for e in changes)


async def test_overlay_link(client, fakes):
    auditor, _ = await make_user(client, "aud@example.com", "auditor")
    batch_id = await ingest(client, fakes, n=1)
    await run_worker(client, fakes)
    doc = (await client.get(f"/batches/{batch_id}", headers=auditor)).json()["documents"][0]
    link = (await client.get(f"/documents/{doc['id']}/overlay", headers=auditor)).json()
    assert (
        link["url"].startswith("https://blob.test/overlays/") and link["expires_in_seconds"] == 900
    )


async def test_unreadable_file_fails_document_and_batch(client, fakes):
    auditor, _ = await make_user(client, "aud@example.com", "auditor")
    batch_id = await ingest(client, fakes, n=1, data=b"not an image")
    await run_worker(client, fakes)
    detail = (await client.get(f"/batches/{batch_id}", headers=auditor)).json()
    assert detail["status"] == "failed"
    assert detail["documents"][0]["status"] == "failed"
    assert "readable image" in detail["documents"][0]["error"]


async def test_retried_job_is_idempotent(client, fakes):
    await ingest(client, fakes, n=1)
    jobs = list(fakes["queue"].jobs)
    await run_worker(client, fakes)
    fakes["queue"].jobs.extend(jobs)  # same job delivered again
    await run_worker(client, fakes, StubClassifier(label="memo"))
    admin, _ = await make_user(client, "admin@example.com", "admin")
    preds = (await client.get("/predictions/recent", headers=admin)).json()
    assert [p["label"] for p in preds] == ["invoice"]


async def test_reviewer_relabels_low_confidence_only(client, fakes):
    reviewer, rid = await make_user(client, "rev@example.com", "reviewer")
    auditor, _ = await make_user(client, "aud@example.com", "auditor")

    await ingest(client, fakes, n=1)
    await run_worker(client, fakes, StubClassifier(label="form", confidence=0.41))
    await ingest(client, fakes, n=1)
    await run_worker(client, fakes, StubClassifier(label="invoice", confidence=0.95))

    recent = (await client.get("/predictions/recent", headers=reviewer)).json()
    assert (await client.get("/predictions/recent", headers=reviewer)).headers[
        "X-FastAPI-Cache"
    ] == "HIT"
    low = next(p for p in recent if p["confidence"] < 0.7)
    high = next(p for p in recent if p["confidence"] >= 0.7)
    assert low["needs_review"] is True

    ok = await client.patch(
        f"/predictions/{low['id']}/label", json={"label": "questionnaire"}, headers=reviewer
    )
    assert ok.status_code == 200
    assert ok.json()["effective_label"] == "questionnaire" and ok.json()["label"] == "form"

    refreshed = (await client.get("/predictions/recent", headers=reviewer)).json()  # invalidated
    assert next(p for p in refreshed if p["id"] == low["id"])["relabeled_to"] == "questionnaire"

    assert (
        await client.patch(
            f"/predictions/{high['id']}/label", json={"label": "memo"}, headers=reviewer
        )
    ).status_code == 409
    assert (
        await client.patch(
            f"/predictions/{low['id']}/label", json={"label": "memo"}, headers=auditor
        )
    ).status_code == 403
    assert (
        await client.patch(
            f"/predictions/{low['id']}/label", json={"label": "not-a-class"}, headers=reviewer
        )
    ).status_code == 422

    entry = (await client.get("/audit?action=prediction.relabeled", headers=auditor)).json()[0]
    assert entry["actor"] == "rev@example.com"
    assert entry["details"]["from"] == "form" and entry["details"]["to"] == "questionnaire"


class FakeDropFolder:
    def __init__(self, files):
        self.files = dict(files)
        self.removed = []

    def list_new(self):
        return sorted(n for n in self.files if not n.endswith(".processing"))

    def list_claimed(self):
        return sorted(n for n in self.files if n.endswith(".processing"))

    def claim(self, name):
        self.files[name + ".processing"] = self.files.pop(name)
        return name + ".processing"

    def read(self, name):
        return self.files[name]

    def remove(self, name):
        self.removed.append(name)
        del self.files[name]

    def sizes(self):
        return {n: len(d) for n, d in self.files.items()}


async def test_sftp_poller_waits_for_stable_size_and_resumes_claims(client, fakes):
    pending = []

    class DeferredCtx(Ctx):
        def run(self, coro):  # production runs this on the worker's loop; here we await it below
            pending.append(coro)

            class _Batch:
                id = "pending"

            return _Batch

    poller = Poller(DeferredCtx(fakes["blob"], client.app.state.invalidator), fakes["queue"])
    folder = FakeDropFolder({"a.tif": tiff_bytes(), "b.tif.processing": tiff_bytes(0)})

    assert (
        poller.poll_once(folder, folder.sizes()) == 1
    )  # resumes the claimed file; a.tif seen once
    await pending.pop()
    assert folder.removed == ["b.tif.processing"]

    assert poller.poll_once(folder, folder.sizes()) == 1  # a.tif size unchanged -> claimed now
    await pending.pop()
    assert folder.removed[-1] == "a.tif.processing" and folder.files == {}
    assert len(fakes["queue"].jobs) == 2

# Decisions

**Layered architecture from day one.** A vertical slice (`/health/db`) proves every layer before features are added, so the boundary can be checked live by adding an endpoint.

**ConvNeXt Tiny as the default backbone.** It meets the CPU latency budget (< 1 s per page) with headroom; Small is about 1 point more accurate at roughly twice the size and latency. Both are supported by `BACKBONES` in `app/classifier/model.py`.

**Full fine-tune, 224 × 224, grayscale replicated to RGB.** RVL-CDIP classes differ in global layout rather than fine text, and the ImageNet stem expects 3 channels. Augmentation is kept layout-preserving (±2° rotation, ±2 % shift, ±5 % scale).

**Store a `state_dict`, not a pickled module.** `torch.load(weights_only=True)` cannot execute code from the file, and the service rebuilds the architecture from the model card's `backbone` field.

**Refuse to start instead of degrading.** Missing weights, a SHA-256 mismatch, a wrong class list, or test top-1 below the README gate all raise `ClassifierStartupError`. Serving from the wrong weights is worse than not serving.

**Golden outputs recorded on CPU in float32.** CI replays on CPU; GPU or fp16 numbers would not match within the 1e-6 tolerance.

**Golden set = 2 easy + 1 ambiguous per class, plus 2 most ambiguous overall.** Easy cases catch gross breakage; ambiguous ones (smallest top-1/top-2 margin) are the most sensitive to subtle drift in weights or preprocessing.

**One preprocessing module shared by training and serving.** The Colab notebook clones the repo and imports `app/classifier/preprocessing.py`, removing a whole class of training/serving skew.

**Casbin policy read from the database on every permission check.** Caching the enforcer would save one small query per request but would delay role changes until a cache expiry or restart. The brief requires a role change to apply on the next page load, so correctness wins; the policy table has one row per permission and per user.

**Roles granted through invitations, not self-selected.** Registration is open (email + password), but a new account has no permissions until an admin invites that email with a role. This keeps registration simple while making every grant an audited admin action. The last admin cannot be demoted, so the system can't lock itself out.

**One async data layer for api and workers.** fastapi-users needs an async session, so repositories are async. Workers run them on a single long-lived event loop (not `asyncio.run` per job) so the async engine's connection pool survives between jobs.

**Enqueue after commit; idempotent jobs.** The ingest worker commits the batch before queueing jobs, so a job never points at a row that does not exist. If the worker crashes after writing a prediction but before acknowledging the job, the retry sees the existing prediction and does nothing.

**SFTP files are claimed by rename and only once their size is stable.** A file is renamed to `*.processing` before it is read and deleted only after its batch is committed. A crash at any point leaves the file either unclaimed or claimed, and claimed files are resumed on the next poll. Requiring the same size on two consecutive polls avoids reading a scan that is still uploading.

**Unreadable scans fail fast; other errors retry.** A file that isn't a decodable image is marked failed immediately. Transient errors (database, MinIO) are retried twice with backoff, and the document is marked failed only after the last attempt.

**Cache invalidation lives in services, by namespace.** Routers declare what is cached; services know what a write changes. Deleting whole namespaces (`batches`, `predictions`, `me`) is coarse but impossible to get subtly wrong, and the cached views are cheap to rebuild.

**The api verifies model artifacts without loading PyTorch.** `verify_artifacts` needs only hashlib and the model card, so the api enforces the same refuse-to-start rule as the worker without holding a model in memory it never uses.

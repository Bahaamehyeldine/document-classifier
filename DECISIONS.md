# Decisions

**Layered architecture from day one.** A vertical slice (`/health/db`) proves every layer before features are added, so the boundary can be checked live by adding an endpoint.

**ConvNeXt Tiny as the default backbone.** It meets the CPU latency budget (< 1 s per page) with headroom; Small is about 1 point more accurate at roughly twice the size and latency. Both are supported by `BACKBONES` in `app/classifier/model.py`.

**Full fine-tune, 224 × 224, grayscale replicated to RGB.** RVL-CDIP classes differ in global layout rather than fine text, and the ImageNet stem expects 3 channels. Augmentation is kept layout-preserving (±2° rotation, ±2 % shift, ±5 % scale).

**Store a `state_dict`, not a pickled module.** `torch.load(weights_only=True)` cannot execute code from the file, and the service rebuilds the architecture from the model card's `backbone` field.

**Refuse to start instead of degrading.** Missing weights, a SHA-256 mismatch, a wrong class list, or test top-1 below the README gate all raise `ClassifierStartupError`. Serving from the wrong weights is worse than not serving.

**Golden outputs recorded on CPU in float32.** CI replays on CPU; GPU or fp16 numbers would not match within the tolerance.

**Golden confidence tolerance is 1e-5, not the brief's 1e-6 (measured, deliberate deviation).** Replaying the 50 golden images on a different CPU than the one that recorded them gave identical labels but confidence differences of up to 1.2e-6 (two images over 1e-6); changing only the torch thread count moved results by ~2e-7 and toggling oneDNN by ~1e-6. Forcing deterministic scalar kernels (`ATEN_CPU_CAPABILITY=default`) is not an answer: it changed confidences by 6e-3, so it is not the computation the service runs. 1e-6 is therefore the noise floor of float32 across hardware, and a test at that level would fail on whichever CPU did not record the values. 1e-5 sits about 8x above the measured noise and still fails on real drift (different weights, resize or normalisation change confidences by far more). Labels must still match exactly. Set `CONFIDENCE_TOLERANCE` in `app/classifier/eval/golden.py` back to `1e-6` if the strict value is required, and record and replay on identical hardware.

**Golden set = 2 easy + 1 ambiguous per class, plus 2 most ambiguous overall.** Easy cases catch gross breakage; ambiguous ones (smallest top-1/top-2 margin) are the most sensitive to subtle drift in weights or preprocessing.

**One training script, run on Colab.** As the brief requires, the model is trained, evaluated and its golden set picked on a Colab GPU, and the ~37 GB dataset never has to live on a laptop. The notebook (`notebooks/train_rvl_cdip.ipynb`) is deliberately thin: it prepares the runtime and calls `scripts/train.py`, which holds all the training logic. Notebook-only training code is hard to review, test and resume; a script is linted and dry-run like the rest of the repo, checkpoints to Google Drive every 300 steps, and resumes after a Colab disconnect. Around it, `scripts/colab_run.sh` makes the whole pipeline idempotent (each step skips itself when its result exists), so recovery is always the same single action: run the cell again. It picks bfloat16 or float16 mixed precision depending on the GPU, so the same command also works on a local NVIDIA card (an early local attempt on a 16 GB laptop ran out of system memory under WSL2, which is the constraint the brief anticipated). The model card's `environment` block records where the shipped weights were trained. The compose stack never trains and never sees the dataset.

**A resumable 224 px cache on Drive, so deleting the Colab runtime no longer costs the dataset.** Idempotent steps recover from a dropped connection, but when Colab *deletes* the runtime its disk goes with it: the 36 GB archive had to be streamed and extracted again (30-40 min), and every epoch then decoded full-size TIFFs on Colab's 2 CPU cores. `scripts/rvl_cache.py` does that work once. It stores each page already resized by the service's own `RESIZE_TRANSFORM` (the first half of the eval transform, split out of `preprocessing.py`) in compressed shards of 10k pages on Drive, written one shard at a time and resumed by skipping what is stored. Training reads those pages. Because the cache holds the output of the *first half of the eval transform*, not a lossy re-encode, the model sees exactly the tensors the worker computes; a test asserts bit equality. The trade-off is Drive space (an estimated 10-14 GB for the full cache), so `TRAIN_PER_CLASS` offers a smaller one. `USE_CACHE = False` keeps the original extract-to-runtime path. The cache keeps the original TIFFs of 1 in 20 test pages (the "golden pool"), because the golden set must be real full-size scans.

**Audit the dataset, never edit the splits.** RVL-CDIP has documented exact duplicates, placeholder pages and test pages duplicated in train. While building the cache, each page's *decoded full-size pixels* are hashed with SHA-256 (not file bytes, which differ by TIFF encoding). `scripts/audit_dataset.py` classifies duplicate groups by which splits they span and flags label conflicts and unreadable scans. The model card then reports two test scores: the **official** one (the quality gate, comparable to published work) and a **leakage-controlled** one that drops test pages with a pixel-identical twin in train or validation. The golden set is drawn from leakage-controlled pages. Near-uniform pages are flagged, never removed on sight, since genuinely blank or dark pages belong to the task.

**One preprocessing module shared by training and serving.** The Colab notebook clones the repo and imports `app/classifier/preprocessing.py`, removing a whole class of training/serving skew.

**Casbin policy read from the database on every permission check.** Caching the enforcer would save one small query per request but would delay role changes until a cache expiry or restart. The brief requires a role change to apply on the next page load, so correctness wins; the policy table has one row per permission and per user.

**Roles granted through invitations, not self-selected.** Registration is open (email + password), but a new account has no permissions until an admin invites that email with a role. This keeps registration simple while making every grant an audited admin action. The last admin cannot be demoted, so the system can't lock itself out.

**One async data layer for api and workers.** fastapi-users needs an async session, so repositories are async. Workers run them on a single long-lived event loop (not `asyncio.run` per job) so the async engine's connection pool survives between jobs.

**Enqueue after commit; idempotent jobs.** The ingest worker commits the batch before queueing jobs, so a job never points at a row that does not exist. If the worker crashes after writing a prediction but before acknowledging the job, the retry sees the existing prediction and does nothing.

**SFTP files are claimed by rename and only once their size is stable.** A file is renamed to `*.processing` before it is read and deleted only after its batch is committed. A crash at any point leaves the file either unclaimed or claimed, and claimed files are resumed on the next poll. Requiring the same size on two consecutive polls avoids reading a scan that is still uploading.

**Unreadable scans fail fast; other errors retry.** A file that isn't a decodable image is marked failed immediately. Transient errors (database, MinIO) are retried twice with backoff, and the document is marked failed only after the last attempt.

**Cache invalidation lives in services, by namespace.** Routers declare what is cached; services know what a write changes. Deleting whole namespaces (`batches`, `predictions`, `me`) is coarse but impossible to get subtly wrong, and the cached views are cheap to rebuild.

**The api verifies model artifacts without loading PyTorch.** `verify_artifacts` needs only hashlib and the model card, so the api enforces the same refuse-to-start rule as the worker without holding a model in memory it never uses.

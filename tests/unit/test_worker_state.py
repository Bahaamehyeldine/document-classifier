"""A worker started with `python -m app.workers.inference` must be able to run its jobs.

`python -m` runs the file as `__main__`, while RQ imports the job function as
`app.workers.inference`: two separate module objects. The loaded model used to live in the
entry module only, so in a real worker every job failed with "Worker not initialised" while
all in-process tests passed. Only the compose smoke test caught it.
"""

import runpy

import pytest

from app.workers import inference
from app.workers.common import STATE


@pytest.fixture(autouse=True)
def _clean_state():
    yield
    STATE.ctx = STATE.classifier = None


@pytest.mark.filterwarnings("ignore::RuntimeWarning")  # runpy notes the module is already imported
def test_state_set_in_the_entry_module_reaches_the_module_rq_imports():
    # Executing the module under another name creates the second copy, like `python -m` does.
    entry = runpy.run_module("app.workers.inference", run_name="__entry__")
    assert entry["classify_document_job"] is not inference.classify_document_job

    ctx, classifier = object(), object()
    STATE.ctx, STATE.classifier = ctx, classifier  # what main() does after loading the model

    assert inference._loaded() == (ctx, classifier)
    assert entry["_loaded"]() == (ctx, classifier)


def test_a_job_fails_clearly_when_the_worker_was_never_initialised():
    with pytest.raises(RuntimeError, match="Worker not initialised"):
        inference.classify_document_job("00000000-0000-0000-0000-000000000000")

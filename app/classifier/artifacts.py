"""Model-artifact checks that need no PyTorch.

Both the api (at startup, without loading the model) and the inference worker
(before loading it) call `verify_artifacts`. Any failure raises
ClassifierStartupError and the process refuses to start.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from app.classifier.labels import RVL_CDIP_CLASSES

CLASSIFIER_DIR = Path(__file__).resolve().parent
WEIGHTS_PATH = CLASSIFIER_DIR / "models" / "classifier.pt"
MODEL_CARD_PATH = CLASSIFIER_DIR / "models" / "model_card.json"

# Keep in sync with the "Model quality gate" line in README.md.
MIN_TEST_TOP1 = 0.85

# Predictions below this top-1 confidence are open to reviewer relabeling.
REVIEW_CONFIDENCE_THRESHOLD = 0.7

SUPPORTED_BACKBONES = ("convnext_tiny", "convnext_small")


class ClassifierStartupError(RuntimeError):
    """Raised when the classifier (or a service depending on it) must not start."""


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def verify_artifacts(
    weights_path: Path = WEIGHTS_PATH,
    card_path: Path = MODEL_CARD_PATH,
    min_test_top1: float = MIN_TEST_TOP1,
) -> dict:
    """Return the model card if the weights are present, intact and good enough."""
    if not weights_path.is_file():
        raise ClassifierStartupError(f"Classifier weights missing: {weights_path}")
    if not card_path.is_file():
        raise ClassifierStartupError(f"Model card missing: {card_path}")

    card = json.loads(card_path.read_text())

    expected_sha = card.get("sha256")
    actual_sha = sha256_file(weights_path)
    if actual_sha != expected_sha:
        raise ClassifierStartupError(
            f"Weights SHA-256 mismatch: model card says {expected_sha}, file is {actual_sha}"
        )

    test_top1 = card.get("metrics", {}).get("test", {}).get("top1")
    if test_top1 is None or test_top1 < min_test_top1:
        raise ClassifierStartupError(
            f"Model card test top-1 {test_top1} is below the committed threshold {min_test_top1}"
        )

    if list(card.get("classes", [])) != list(RVL_CDIP_CLASSES):
        raise ClassifierStartupError("Model card class list does not match RVL_CDIP_CLASSES")

    if card.get("backbone") not in SUPPORTED_BACKBONES:
        raise ClassifierStartupError(
            f"Unknown backbone {card.get('backbone')!r}; "
            f"expected one of {sorted(SUPPORTED_BACKBONES)}"
        )
    return card

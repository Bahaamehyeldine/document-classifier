"""Golden-set replay test.

Runs the committed classifier on the 50 golden TIFFs and compares with the
outputs recorded on Colab (on CPU, float32) when the model was trained.
Pass = byte-identical labels and top-1 confidence within 1e-6.
Any drift in weights, preprocessing or library versions fails CI.

Run: pytest app/classifier/eval/golden.py
"""

import json
from pathlib import Path

import pytest

from app.classifier.model import load_classifier

EVAL_DIR = Path(__file__).resolve().parent
GOLDEN_IMAGES = EVAL_DIR / "golden_images"
GOLDEN_EXPECTED = EVAL_DIR / "golden_expected.json"
CONFIDENCE_TOLERANCE = 1e-6


def _cases():
    if not GOLDEN_EXPECTED.is_file():
        return []
    return json.loads(GOLDEN_EXPECTED.read_text())["images"]


@pytest.fixture(scope="module")
def classifier():
    return load_classifier()


def test_golden_set_is_complete():
    cases = _cases()
    assert len(cases) == 50, f"expected 50 golden images, found {len(cases)}"
    for case in cases:
        assert (GOLDEN_IMAGES / case["file"]).is_file(), (
            f"missing golden image {case['file']}"
        )


@pytest.mark.parametrize("case", _cases(), ids=lambda c: c["file"])
def test_golden_prediction_matches(classifier, case):
    pred = classifier.predict_path(GOLDEN_IMAGES / case["file"])
    assert pred.label == case["expected_label"]
    assert abs(pred.confidence - case["expected_confidence"]) <= CONFIDENCE_TOLERANCE, (
        f"{case['file']}: confidence {pred.confidence!r} vs expected {case['expected_confidence']!r}"
    )

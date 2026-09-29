"""Refuse-to-start rules and inference for app.classifier.model (no real weights needed)."""

import json

import pytest
import torch
from PIL import Image

from app.classifier.labels import RVL_CDIP_CLASSES
from app.classifier.model import (
    MIN_TEST_TOP1,
    ClassifierStartupError,
    build_model,
    load_classifier,
    render_overlay,
    sha256_file,
)


@pytest.fixture(scope="module")
def weights(tmp_path_factory):
    torch.manual_seed(0)
    path = tmp_path_factory.mktemp("w") / "classifier.pt"
    torch.save(build_model("convnext_tiny").state_dict(), path)
    return path


def write_card(path, weights, **overrides):
    card = {
        "backbone": "convnext_tiny",
        "classes": list(RVL_CDIP_CLASSES),
        "sha256": sha256_file(weights),
        "metrics": {"test": {"top1": MIN_TEST_TOP1 + 0.05, "top5": 0.98}},
    }
    card.update(overrides)
    path.write_text(json.dumps(card))
    return path


def test_loads_and_predicts(weights, tmp_path):
    clf = load_classifier(weights, write_card(tmp_path / "card.json", weights))
    pred = clf.predict_image(Image.new("L", (600, 800), 255))
    assert pred.label in RVL_CDIP_CLASSES
    assert len(pred.top5) == 5
    assert 0.0 <= pred.confidence <= 1.0
    assert pred.top5[0] == (pred.label, pred.confidence)


def test_refuses_when_weights_missing(tmp_path, weights):
    card = write_card(tmp_path / "card.json", weights)
    with pytest.raises(ClassifierStartupError, match="weights missing"):
        load_classifier(tmp_path / "nope.pt", card)


def test_refuses_on_sha_mismatch(tmp_path, weights):
    card = write_card(tmp_path / "card.json", weights, sha256="0" * 64)
    with pytest.raises(ClassifierStartupError, match="SHA-256 mismatch"):
        load_classifier(weights, card)


def test_refuses_below_quality_gate(tmp_path, weights):
    card = write_card(
        tmp_path / "card.json",
        weights,
        metrics={"test": {"top1": MIN_TEST_TOP1 - 0.01}},
    )
    with pytest.raises(ClassifierStartupError, match="below the committed threshold"):
        load_classifier(weights, card)


def test_refuses_on_wrong_class_list(tmp_path, weights):
    card = write_card(tmp_path / "card.json", weights, classes=list(reversed(RVL_CDIP_CLASSES)))
    with pytest.raises(ClassifierStartupError, match="class list"):
        load_classifier(weights, card)


def test_refuses_unknown_backbone(tmp_path, weights):
    card = write_card(tmp_path / "card.json", weights, backbone="resnet50")
    with pytest.raises(ClassifierStartupError, match="Unknown backbone"):
        load_classifier(weights, card)


def test_overlay_marks_low_confidence(weights, tmp_path):
    clf = load_classifier(weights, write_card(tmp_path / "card.json", weights))
    page = Image.new("L", (600, 800), 255)
    pred = clf.predict_image(page)
    overlay = render_overlay(page, pred)
    assert overlay.size == page.size and overlay.mode == "RGB"
    assert overlay.getpixel((2, 2)) != (255, 255, 255)  # label banner drawn


def test_readme_states_the_same_quality_gate():
    from pathlib import Path

    readme = (Path(__file__).resolve().parents[2] / "README.md").read_text()
    assert f"test top-1 ≥ {MIN_TEST_TOP1:.2f}" in readme

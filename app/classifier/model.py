"""Load the fine-tuned ConvNeXt classifier and run inference.

The worker (and the api, for its startup check) call `load_classifier()`.
It refuses to return a model unless:
  * the weights file exists,
  * its SHA-256 matches the model card,
  * the model card's full-test top-1 is at least MIN_TEST_TOP1
    (the threshold committed in the README).
Any failure raises ClassifierStartupError, so the process exits instead of
serving predictions from the wrong weights.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision import models

from app.classifier.labels import NUM_CLASSES, RVL_CDIP_CLASSES
from app.classifier.preprocessing import load_image, to_tensor

CLASSIFIER_DIR = Path(__file__).resolve().parent
WEIGHTS_PATH = CLASSIFIER_DIR / "models" / "classifier.pt"
MODEL_CARD_PATH = CLASSIFIER_DIR / "models" / "model_card.json"

# Keep in sync with the "Model quality gate" line in README.md.
MIN_TEST_TOP1 = 0.85

# Predictions below this top-1 confidence are open to reviewer relabeling.
REVIEW_CONFIDENCE_THRESHOLD = 0.7

BACKBONES = {
    "convnext_tiny": models.convnext_tiny,
    "convnext_small": models.convnext_small,
}


class ClassifierStartupError(RuntimeError):
    """Raised when the classifier must not start."""


@dataclass(frozen=True)
class Prediction:
    label: str
    label_index: int
    confidence: float
    top5: list[tuple[str, float]]

    @property
    def needs_review(self) -> bool:
        return self.confidence < REVIEW_CONFIDENCE_THRESHOLD


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def build_model(backbone: str) -> torch.nn.Module:
    """ConvNeXt with its last layer replaced by a 16-way head (no pretrained download)."""
    if backbone not in BACKBONES:
        raise ClassifierStartupError(
            f"Unknown backbone {backbone!r}; expected one of {sorted(BACKBONES)}"
        )
    model = BACKBONES[backbone](weights=None)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = torch.nn.Linear(in_features, NUM_CLASSES)
    return model


class DocumentClassifier:
    def __init__(self, model: torch.nn.Module, card: dict):
        self.model = model.eval()
        self.card = card

    @torch.inference_mode()
    def predict_tensor(self, batch: torch.Tensor) -> list[Prediction]:
        probs = torch.softmax(self.model(batch).float(), dim=1)
        top_p, top_i = probs.topk(5, dim=1)
        out = []
        for p_row, i_row in zip(top_p.tolist(), top_i.tolist()):
            top5 = [(RVL_CDIP_CLASSES[i], p) for i, p in zip(i_row, p_row)]
            out.append(
                Prediction(
                    label=top5[0][0],
                    label_index=i_row[0],
                    confidence=p_row[0],
                    top5=top5,
                )
            )
        return out

    def predict_image(self, img: Image.Image) -> Prediction:
        return self.predict_tensor(to_tensor(img).unsqueeze(0))[0]

    def predict_path(self, path: str | Path) -> Prediction:
        return self.predict_image(load_image(path))


def load_classifier(
    weights_path: Path = WEIGHTS_PATH,
    card_path: Path = MODEL_CARD_PATH,
    min_test_top1: float = MIN_TEST_TOP1,
) -> DocumentClassifier:
    """Verify weights against the model card, then load them. Refuses to start on any mismatch."""
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
        raise ClassifierStartupError(
            "Model card class list does not match RVL_CDIP_CLASSES"
        )

    model = build_model(card.get("backbone", ""))
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    torch.set_grad_enabled(False)
    return DocumentClassifier(model, card)


def render_overlay(img: Image.Image, pred: Prediction) -> Image.Image:
    """Annotated copy of the page with the predicted label and confidence, for MinIO."""
    canvas = img.convert("RGB")
    draw = ImageDraw.Draw(canvas)
    font_size = max(16, canvas.width // 30)
    try:
        font = ImageFont.load_default(size=font_size)
    except TypeError:  # Pillow < 10.1
        font = ImageFont.load_default()
    text = f"{pred.label}  {pred.confidence:.1%}" + (
        "  [needs review]" if pred.needs_review else ""
    )
    colour = (200, 30, 30) if pred.needs_review else (20, 120, 40)
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    pad = font_size // 3
    draw.rectangle((0, 0, right - left + 2 * pad, bottom - top + 2 * pad), fill=colour)
    draw.text((pad, pad - top), text, fill=(255, 255, 255), font=font)
    return canvas

"""Image preprocessing shared by training (Colab) and inference (worker).

The Colab notebook copies this exact transform. If you change it here, retrain:
the golden-set replay test will fail otherwise, which is the point.
"""

from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms import v2

INPUT_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

EVAL_TRANSFORM = v2.Compose(
    [
        v2.ToImage(),
        v2.Resize(
            (INPUT_SIZE, INPUT_SIZE),
            interpolation=v2.InterpolationMode.BILINEAR,
            antialias=True,
        ),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
)


def load_image(path: str | Path) -> Image.Image:
    """Open a scanned page (usually a grayscale TIFF) as a 3-channel image.

    Classifies visual layout only: no OCR. Grayscale is replicated to RGB so
    the ImageNet-pretrained stem sees the channel count it expects.
    """
    with Image.open(path) as img:
        img.seek(0)  # multi-page TIFFs: first page only
        return img.convert("L").convert("RGB")


def to_tensor(img: Image.Image) -> torch.Tensor:
    """PIL image -> normalized (3, 224, 224) float tensor."""
    return EVAL_TRANSFORM(img)

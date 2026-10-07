"""Image preprocessing shared by training (Colab) and inference (worker).

The Colab notebook imports this module. If you change it here, retrain:
the golden-set replay test will fail otherwise, which is the point.
"""

import io
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms import v2

INPUT_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# The two halves of the eval transform are separate so the training cache can store the
# result of the first (a uint8 224 x 224 page) and apply the second at load time. Their
# composition is exactly what the service runs, which a test asserts.
RESIZE_TRANSFORM = v2.Compose(
    [
        v2.ToImage(),
        v2.Resize(
            (INPUT_SIZE, INPUT_SIZE),
            interpolation=v2.InterpolationMode.BILINEAR,
            antialias=True,
        ),
    ]
)
NORMALIZE_TRANSFORM = v2.Compose(
    [
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
)
EVAL_TRANSFORM = v2.Compose([RESIZE_TRANSFORM, NORMALIZE_TRANSFORM])


def _as_rgb(img: Image.Image) -> Image.Image:
    img.seek(0)  # multi-page TIFFs: first page only
    return img.convert("L").convert("RGB")


def load_image(path: str | Path) -> Image.Image:
    """Open a scanned page (usually a grayscale TIFF) as a 3-channel image.

    Classifies visual layout only: no OCR. Grayscale is replicated to RGB so
    the ImageNet-pretrained stem sees the channel count it expects.
    """
    with Image.open(path) as img:
        return _as_rgb(img)


def load_image_bytes(data: bytes) -> Image.Image:
    """Same as load_image, for bytes fetched from blob storage."""
    with Image.open(io.BytesIO(data)) as img:
        return _as_rgb(img)


def to_tensor(img: Image.Image) -> torch.Tensor:
    """PIL image -> normalized (3, 224, 224) float tensor."""
    return EVAL_TRANSFORM(img)

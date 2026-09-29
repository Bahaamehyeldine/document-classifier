# Decisions

**Layered architecture from day one.** A vertical slice (`/health/db`) proves every layer before features are added, so the boundary can be checked live by adding an endpoint.

**ConvNeXt Tiny as the default backbone.** It meets the CPU latency budget (< 1 s per page) with headroom; Small is about 1 point more accurate at roughly twice the size and latency. Both are supported by `BACKBONES` in `app/classifier/model.py`.

**Full fine-tune, 224 × 224, grayscale replicated to RGB.** RVL-CDIP classes differ in global layout rather than fine text, and the ImageNet stem expects 3 channels. Augmentation is kept layout-preserving (±2° rotation, ±2 % shift, ±5 % scale).

**Store a `state_dict`, not a pickled module.** `torch.load(weights_only=True)` cannot execute code from the file, and the service rebuilds the architecture from the model card's `backbone` field.

**Refuse to start instead of degrading.** Missing weights, a SHA-256 mismatch, a wrong class list, or test top-1 below the README gate all raise `ClassifierStartupError`. Serving from the wrong weights is worse than not serving.

**Golden outputs recorded on CPU in float32.** CI replays on CPU; GPU or fp16 numbers would not match within the 1e-6 tolerance.

**Golden set = 2 easy + 1 ambiguous per class, plus 2 most ambiguous overall.** Easy cases catch gross breakage; ambiguous ones (smallest top-1/top-2 margin) are the most sensitive to subtle drift in weights or preprocessing.

**One preprocessing module shared by training and serving.** The Colab notebook clones the repo and imports `app/classifier/preprocessing.py`, removing a whole class of training/serving skew.

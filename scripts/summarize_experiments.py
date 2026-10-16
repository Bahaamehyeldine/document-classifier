"""Print the experiments written by `train.py --experiment` as one table (validation only).

python -m scripts.summarize_experiments ~/data/rvl-cdip-run/experiments
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

CHANCE = 1 / 16  # RVL-CDIP is balanced: 25,000 pages in each of 16 classes


def load(root: Path) -> list[dict]:
    return sorted(
        (json.loads(p.read_text()) for p in root.glob("*/experiment.json")),
        key=lambda r: r["val_top1"],
        reverse=True,
    )


def table(rows: list[dict]) -> str:
    head = (
        "| experiment | backbone | freeze | lr | seed | train imgs | val top-1 | val top-5 "
        "| trainable params | minutes |\n|---|---|---|---|---|---|---|---|---|---|"
    )
    lines = [
        f"| {r['name']} | {r['backbone']} | {r['freeze']} | {r['lr']:g} | {r['seed']} "
        f"| {r['train_images']:,} | **{r['val_top1']:.4f}** | {r['val_top5']:.4f} "
        f"| {r['trainable_parameters']:,} | {r['train_minutes']} |"
        for r in rows
    ]
    n_val = rows[0]["val_images"]
    foot = (
        f"\nValidation only ({n_val:,} fixed pages; the test split was not used). "
        f"Chance is {CHANCE:.4f}. Differences smaller than the gap between the repeated "
        "`full` runs (different seeds) are within noise."
    )
    return "\n".join([head, *lines]) + "\n" + foot


def main() -> int:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("experiments")
    rows = load(root)
    if not rows:
        print(f"No experiment.json files under {root}")
        return 1
    text = table(rows)
    print(text)
    (root / "summary.md").write_text(text + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

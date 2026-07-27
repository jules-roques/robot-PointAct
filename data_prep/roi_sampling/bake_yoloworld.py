"""Pre-bake a YOLO-World checkpoint with a fixed text prompt (login node, online).

YOLO-World's open-vocab prompting uses a CLIP text encoder that downloads on first
`set_classes`. Compute nodes are offline, so we bake the prompt's embeddings into a
checkpoint here (with internet) and ship that; inference then needs no downloads.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="yolov8s-world.pt", help="Base YOLO-World weights (auto-downloads).")
    ap.add_argument("--prompt", nargs="+", default=["drawer handle"])
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    from ultralytics import YOLOWorld

    model = YOLOWorld(args.base)
    model.set_classes(list(args.prompt))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(args.out))
    print(f"baked prompt {args.prompt} -> {args.out}")

    # Verify the baked checkpoint runs (no network needed for a fresh load + predict).
    reloaded = YOLOWorld(str(args.out))
    dummy = (np.random.rand(256, 256, 3) * 255).astype(np.uint8)
    res = reloaded.predict(dummy[..., ::-1], conf=0.001, verbose=False)
    print(f"verify: baked model predicts OK (boxes on random image: {len(res[0].boxes)})")


if __name__ == "__main__":
    main()

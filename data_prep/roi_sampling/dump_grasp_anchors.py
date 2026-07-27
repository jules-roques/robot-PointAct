"""Stage A2: dump each episode's grasp anchor (the target drawer's ground truth).

For every episode, the end-effector position at the frame the gripper first closes is the
handle the demonstration reached for. ``observation.state[0:3]`` is ``base_to_eef_pos`` —
already in the robot-base frame, the same frame as the stored point clouds — so the value
can be compared directly against detection anchors.

This runs in the training venv (needs pyarrow) and writes a small JSON that Stage B reads,
so the ROI preprocessing env needs no extra dependency and the anchors stay auditable.

Why not decode the instruction instead: "left"/"right" cannot be resolved geometrically on
this dataset — the robot parks at a different yaw in each kitchen, so left-drawer episodes
grasp at both positive and negative base-frame Y.

Usage:
    python data_prep/roi_sampling/dump_grasp_anchors.py \
        --dataset-dir $SCRATCH/.../OpenDrawer \
        --out         $SCRATCH/.../OpenDrawer/roi_meta/grasp_anchors.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from tqdm.auto import tqdm


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    d = args.dataset_dir.expanduser().resolve()
    info = json.loads((d / "meta" / "info.json").read_text())
    chunks = int(info.get("chunks_size", 1000))

    episodes, tasks = [], {}
    with open(d / "meta" / "episodes.jsonl") as f:
        for line in f:
            r = json.loads(line)
            ep = int(r["episode_index"])
            episodes.append(ep)
            tk = r.get("tasks") or [""]
            tasks[ep] = tk[0] if tk else ""

    anchors, missing = {}, 0
    for ep in tqdm(sorted(episodes), desc="episodes", unit="ep"):
        p = d / "data" / f"chunk-{ep // chunks:03d}" / f"episode_{ep:06d}.parquet"
        try:
            tb = pq.read_table(p, columns=["observation.state", "action"])
        except Exception:
            missing += 1
            continue
        S = np.array([np.asarray(x, np.float32) for x in tb.column("observation.state").to_pylist()])
        A = np.array([np.asarray(x, np.float32) for x in tb.column("action").to_pylist()])
        if S.ndim != 2 or S.shape[1] < 3 or A.ndim != 2 or A.shape[1] < 8:
            missing += 1
            continue
        idx = np.flatnonzero(A[:, 7] > 0.5)  # gripper_close
        if len(idx) == 0:
            missing += 1
            continue
        anchors[str(ep)] = [float(v) for v in S[idx[0], 0:3]]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "description": "eef position (robot-base frame) at first gripper close, per episode",
        "num_episodes": len(episodes),
        "num_with_anchor": len(anchors),
        "num_missing": missing,
        "anchors": anchors,
        "tasks": {str(k): v for k, v in tasks.items()},
    }, indent=2))
    print(f"grasp anchors: {len(anchors)}/{len(episodes)} episodes (missing {missing}) -> {args.out}")


if __name__ == "__main__":
    main()

"""Why is a MolmoMotion forward so expensive, and what would a full cache cost?

The Stage 4.0 gate measured 63.7 s per forward, roughly 80x MolmoPoint's 0.78 s, from a
model with *half* the parameters. That is not a paradox once you look at what is being
generated rather than how big the model is, and this script measures the decomposition
instead of arguing it.

Method: run the same real request at several ``future_horizon`` values and fit

    seconds_per_forward = a + b * F

``a`` is everything that does not scale with the horizon -- vision encoding of the three
history frames, prefill of the prompt, Python overhead. ``b * F`` is the autoregressive
decode, which is the part that should dominate if the token count is the explanation.
The fit also gives the honest projection for a whole-dataset cache at any horizon, which
a single measurement at F=30 cannot.

Usage:
    $SCRATCH/venvs/molmo_motion/bin/python -m data_prep.roi_sampling.profile_molmo_motion \
        --dataset-root $SCRATCH/datasets/robot_data/robocasa365/lerobot_point_lmdb \
        --model-dir $SCRATCH/models/MolmoMotion-4B-H3-F30 --task OpenDrawer
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from pointact.roi_sampling.geometry import base_to_camera, project_base_points
from pointact.roi_sampling.molmo_motion import HISTORY_SIZE, history_frame_indices

from data_prep.roi_sampling.gate_molmo_motion import (
    FPS,
    decode_video,
    load_calib,
    read_episode_states,
    read_meta,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", required=True, type=Path)
    ap.add_argument("--model-dir", required=True, type=Path)
    ap.add_argument("--task", default="OpenDrawer")
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--horizons", nargs="*", type=int, default=[5, 10, 20, 30])
    ap.add_argument("--n-forwards", type=int, default=3,
                    help="Requests per horizon. The first is discarded as warm-up.")
    ap.add_argument("--history-stride", type=int, default=1)
    ap.add_argument("--replan-stride", type=int, default=8,
                    help="Cadence a cache would be built at, for the projection.")
    ap.add_argument("--num-points", type=int, default=None,
                    help="Override config.num_points (default 8). This is decode-bound, so "
                         "asking for 1 point instead of 8 is the largest cost lever there is.")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    dataset_dir = args.dataset_root / args.task
    cams, _hw = load_calib(dataset_dir / "roi_meta" / "camera_calib.npz")
    cam = cams["left"]
    lengths, instructions = read_meta(dataset_dir / "meta")
    info = json.loads((dataset_dir / "meta" / "info.json").read_text())
    chunks = int(info.get("chunks_size", 1000))

    states = read_episode_states(dataset_dir, args.episode, chunks)
    video = (dataset_dir / "videos" / f"chunk-{args.episode // chunks:03d}"
             / "observation.images.left_image" / f"episode_{args.episode:06d}.mp4")
    frames = decode_video(video)
    instruction = instructions.get(args.episode, "")

    eef_base = states[:, :3]
    eef_cam = base_to_camera(eef_base, cam["base2cam"])

    from pointact.roi_sampling.molmo_motion import MolmoMotionForecaster
    fc = MolmoMotionForecaster(str(args.model_dir), num_points=args.num_points)
    print(f"num_points={fc.num_points}")

    # Query frames spread through the episode, so the timing is not dominated by one
    # unusually short or long generation.
    n_use = min(len(states), len(frames))
    cands = [t for t in range(0, n_use, args.replan_stride)
             if history_frame_indices(t, HISTORY_SIZE, args.history_stride) is not None]
    picks = [cands[int(r * (len(cands) - 1))]
             for r in np.linspace(0.2, 0.8, args.n_forwards)]

    rows = []
    for F in args.horizons:
        secs, chars = [], []
        for i, t0 in enumerate(picks):
            hist = history_frame_indices(t0, HISTORY_SIZE, args.history_stride)
            uv0, _z, _f = project_base_points(eef_base[t0:t0 + 1], cam["intrinsic"],
                                              cam["base2cam"])
            t_start = time.time()
            fc.forecast_point(
                history_frames=[frames[j] for j in hist],
                point_2d_at_t0=uv0[0],
                point_3d_history_cam=eef_cam[hist],
                action=instruction,
                future_horizon=F,
            )
            dt = time.time() - t_start
            # The first call pays cuDNN autotuning and lazy CUDA init, which is a
            # one-off and would otherwise inflate the smallest horizon most.
            if i == 0:
                continue
            secs.append(dt)
            chars.append(len(fc.last_future_text))
        rows.append({"future_horizon": F, "s_per_forward": float(np.mean(secs)),
                     "s_std": float(np.std(secs)), "text_chars": float(np.mean(chars))})
        print(f"F={F:<4} {rows[-1]['s_per_forward']:7.1f} s "
              f"(+/-{rows[-1]['s_std']:.1f})  text={rows[-1]['text_chars']:.0f} chars",
              flush=True)

    # Fit seconds = a + b * F. Two unknowns, so this needs at least two horizons.
    F = np.array([r["future_horizon"] for r in rows], dtype=float)
    S = np.array([r["s_per_forward"] for r in rows], dtype=float)
    b, a = np.polyfit(F, S, 1) if len(rows) >= 2 else (float("nan"), float("nan"))

    total_frames = int(info.get("total_frames", 0))
    n_forwards = total_frames // args.replan_stride
    print(f"\nfit: seconds = {a:.1f} + {b:.2f} * F")
    print(f"  horizon-independent (vision + prefill + overhead): {a:.1f} s")
    print(f"  per future step (autoregressive decode):           {b:.2f} s")
    if a + b * 30 > 0:
        print(f"  decode share at F=30: {100 * b * 30 / (a + b * 30):.0f}%")

    print(f"\n{args.task}: {total_frames} frames, stride {args.replan_stride} "
          f"-> {n_forwards} forwards (one view, one history stride)")
    print(f"{'F':<6}{'s/fwd':<10}{'GPU-hours':<12}{'GPU-days':<10}")
    for f in args.horizons:
        s = a + b * f
        h = s * n_forwards / 3600
        print(f"{f:<6}{s:<10.1f}{h:<12.1f}{h / 24:<10.1f}")
    print("\nFor reference, the MolmoPoint cache for this task was 15,822 forwards at "
          "0.774 s = 3.4 GPU-h.")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(
            {"task": args.task, "rows": rows, "fit_intercept_s": a, "fit_slope_s_per_step": b,
             "total_frames": total_frames, "replan_stride": args.replan_stride,
             "forwards": n_forwards}, indent=2))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

"""Render eval rollouts dumped by run_robocasa365_client.py into point-cloud animations.

Split from the client because the two halves need different environments: the client runs in
envs/robocasa365 (MuJoCo, python 3.11) which has neither plotly nor lmdb, while the pointact
env has both. The client writes raw arrays; this turns them into the same figures the training
animation uses, so eval and training artefacts are directly comparable.

    python experiments/13_robocasa365/render_rollouts.py <results/checkpoint-50000-viz>
"""

import argparse
import sys
import types
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from data_prep.roi_sampling.viz_sampling_episode import build_figure, oversampling_factor
from pointact.roi_sampling.geometry import eef_density_weights

#: Ablation sampling name -> the name viz_sampling_episode knows it by.
VIZ_METHOD = {"anchor": "oracle"}


def render(npz_path: Path, sigma: float, floor: float, num_frames: int) -> Path:
    data = np.load(npz_path, allow_pickle=False)
    sampling = str(data["sampling"])
    outcome = str(data["outcome"])
    trial = int(data["trial"])
    total = int(data["n_frames"])

    # Thin to a fixed frame count so file size does not track episode length; every sampled
    # point is still drawn, since point count is the thing being compared.
    stride = max(1, total // max(1, num_frames))
    indices = list(range(0, total, stride))

    frames_data = []
    for i in indices:
        pts = data[f"points_{i}"][:, :3]
        eef = data[f"eef_{i}"]
        # The server weighted the cloud around this anchor; recompute rather than transmit it.
        weights = (eef_density_weights(pts, eef, sigma, floor)
                   if sampling in ("eef", "anchor") else None)
        logw = np.log2(np.clip(oversampling_factor(weights, len(pts)), 1e-6, None))
        frames_data.append({
            "pts": pts, "colors": logw.astype(np.float32), "n_sel": len(pts),
            "anchor": eef if weights is not None else None, "frame": int(data[f"step_{i}"]),
            "near_frac": None, "n_handle_total": 0, "n_handle_sel": 0,
            "handle_recall": None, "n_cloud": len(pts), "n_frames": len(indices),
            "phase": outcome,
        })

    style = types.SimpleNamespace(
        color_by="weight", dark=True, frame_ms=120, point_size=2.0,
        near_radius=0.15, roi_radius=0.15, roi_radius_scale=1.0,
    )
    # The ablation calls the GT-handle arm "anchor"; viz_sampling_episode predates that name and
    # keys its label/colour tables on "oracle". Translate here rather than aliasing in the viz
    # module, which the training-time callback also drives and which already passes "oracle".
    # Without this, every anchor rollout died as KeyError: 'anchor' -- and because
    # eval_robocasa365.sh renders with `|| echo ...`, the eval still reported success.
    fig = build_figure(VIZ_METHOD.get(sampling, sampling), frames_data, trial,
                       f"eval rollout ({outcome})", style)
    out = npz_path.with_suffix(".html")
    # CDN plotly: inlining adds ~3.5MB per figure, and W&B renders these in a browser.
    fig.write_html(str(out), include_plotlyjs="cdn", auto_play=False)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dirs", nargs="+", type=Path, help="Directories holding rollout_*.npz")
    parser.add_argument("--sigma", type=float, default=0.08)
    parser.add_argument("--floor", type=float, default=0.05)
    parser.add_argument("--num-frames", type=int, default=20)
    args = parser.parse_args()

    total = 0
    for directory in args.dirs:
        for npz_path in sorted(directory.glob("rollout_*.npz")):
            try:
                out = render(npz_path, args.sigma, args.floor, args.num_frames)
                print(f"  {out}  ({out.stat().st_size / 1e6:.1f} MB)")
                total += 1
            except Exception as exc:  # noqa: BLE001 - one bad figure must not stop the rest
                print(f"  FAILED {npz_path.name}: {type(exc).__name__}: {exc}")
    print(f"\nrendered {total} rollout figure(s)")


if __name__ == "__main__":
    main()

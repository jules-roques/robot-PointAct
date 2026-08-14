"""Check a stage-5 run config's data path before it is given a GPU.

Every failure this looks for produces a *healthy-looking training curve*, which is why it is
worth a minute of CPU:

* an anchor cache that does not load -- every frame silently falls through to the fallback,
  so the arm trains the fallback under the anchor's name;
* the opposite, a fallback that never fires, which means the flag is inert;
* a molmo arm reading a per-view cache while its config says closest_gt, which trains on up
  to three centres per query and is then evaluated on one (or the reverse);
* an oracle npz whose episodes do not line up with the dataset's, which returns None for most
  frames and quietly becomes a uniform draw;
* a point budget that cannot be met, which makes the point-count coordinate a fiction.

Reports what the sampler ACTUALLY did over a sample of frames rather than that it ran.

    python experiments/13_robocasa365/smoke_stage5.py \
        experiments/13_robocasa365/runs/s5-tom-molmo-n8192-s0.yaml --frames 40
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# LeRobotDatasetMetadata resolves a revision against the Hub unless told not to, and these
# repo_ids ("TurnOnMicrowave") are local directories that do not exist there -- the failure is
# a 401, which reads as a credentials problem rather than a lookup that should never have
# happened. The training jobs set this in their slurm scripts; set it here too so the check
# runs the same way from a login node. Must precede the huggingface_hub import.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lerobot.constants import OBS_STATE  # noqa: E402

from pointact.constants import OBS_POINTS  # noqa: E402
from pointact.data.robot.multi_data import load_single_lerobot_dataset  # noqa: E402
from pointact.data.schema import LerobotConfig  # noqa: E402
from pointact.roi_sampling.geometry import eef_density_weights  # noqa: E402
from pointact.train.run_config import resolve_run_config  # noqa: E402

#: A draw is "concentrated" if it puts more of its budget near the anchor than a uniform draw
#: would. Below this ratio the arm is not doing what its name says, whatever the config holds.
MIN_CONCENTRATION = 1.5


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_config", type=Path)
    ap.add_argument("--frames", type=int, default=40, help="Frames to draw.")
    ap.add_argument("--radius", type=float, default=0.08,
                    help="Radius the concentration is measured in, = the sampling sigma.")
    args = ap.parse_args()

    meta, data, train = resolve_run_config(args.run_config)
    cfg = dict(data["lerobot_datasets"][0])
    arm = meta.get("sampling", "?")
    print(f"run={train.get('run_name')} task={meta.get('task')} arm={arm} "
          f"npoints={cfg.get('max_npoints')} steps={train.get('max_steps')}")
    for key in sorted(cfg):
        if any(k in key for k in ("oracle", "molmo", "eef")) and key != "eef_sampling_floor":
            print(f"  {key}: {cfg[key]}")

    # Built through the trainer's own loader rather than by calling the dataset class here:
    # it is what joins `root` to `repo_id` and derives delta_timestamps from the metadata's
    # fps, and a smoke check that constructs the dataset differently from training is
    # checking something other than training.
    ds = load_single_lerobot_dataset(0, [LerobotConfig(**cfg)], chunk_size=train["chunk_size"])
    print(f"  dataset: {ds.num_frames} frames, {ds.num_episodes} episodes")

    rng = np.random.default_rng(0)
    idxs = rng.choice(ds.num_frames, size=min(args.frames, ds.num_frames), replace=False)

    counts, concentrations = [], []
    n_anchor, n_fallback, n_centres = 0, 0, []
    for i in idxs.tolist():
        item = ds.hf_dataset[int(i)]
        ep, fr = int(item["episode_index"]), int(item["frame_index"])
        cloud = ds.load_point_cloud(ep, fr)

        # Which centres would this frame's sampler use? Same calls the dataset makes.
        centres = None
        if ds.molmo_sampling:
            centres = ds.load_molmo_anchors(ep, fr)
        elif ds.oracle_sampling and ds.oracle_gt == "geom":
            centres = ds.load_oracle_geom_anchor(ep, fr)
        elif ds.oracle_sampling:
            labels = ds.load_point_labels(ep, fr)
            cloud_c, labels_c = ds.filter_point_cloud_by_workspace(cloud, labels)
            centres = ds.oracle_anchor(cloud_c, labels_c)
        elif ds.eef_sampling:
            # The RAW state, not ds[i]'s: __getitem__ centres the cloud and may rotate the
            # state with it, so the processed copy is in a different frame from the raw cloud
            # this check measures against. augment_point_cloud reads it before either happens.
            centres = np.asarray(item[OBS_STATE], dtype=np.float64)[:3]

        if centres is None:
            n_fallback += 1
        else:
            n_anchor += 1
            n_centres.append(1 if np.asarray(centres).ndim == 1 else len(centres))

        counts.append(len(np.asarray(ds[int(i)][OBS_POINTS])))

        if centres is not None and arm != "uniform":
            # How much of the budget the density steers inside one sigma of the anchor,
            # against the share a uniform draw would put there. Computed from the weights
            # rather than by re-drawing: the draw is proportional to w, so comparing mass is
            # the same quantity without the sampling noise. Measured on the workspace-cropped
            # cloud, which is what the sampler sees, and before centring, which is what the
            # anchors are expressed in.
            full, _ = ds.filter_point_cloud_by_workspace(cloud, None)
            w = eef_density_weights(full[:, :3], centres, args.radius, 0.05)
            near = w > (0.05 + 0.95 * np.exp(-0.5))       # within 1 sigma of some centre
            if near.any():
                concentrations.append(float(w[near].sum() / w.sum()) / float(near.mean()))

    print(f"\n  points drawn: median {int(np.median(counts))} "
          f"(min {min(counts)}, max {max(counts)}) of a {cfg.get('max_npoints')} budget")
    if arm != "uniform":
        total = n_anchor + n_fallback
        print(f"  anchored frames: {n_anchor}/{total}"
              f"   fallback ({cfg.get('molmo_fallback', 'uniform')}): {n_fallback}/{total}")
        if n_centres:
            uniq = sorted(set(n_centres))
            print(f"  centres per anchored frame: {uniq}"
                  f"  (mean {np.mean(n_centres):.2f})")
        if concentrations:
            print(f"  budget inside 1 sigma vs uniform: {np.median(concentrations):.1f}x")

    problems = []
    if int(np.median(counts)) != int(cfg.get("max_npoints", 0)):
        problems.append(f"point budget not met: median {int(np.median(counts))} "
                        f"!= {cfg.get('max_npoints')}. The cloud may be smaller than the "
                        f"budget after voxelisation and the workspace crop.")
    if arm != "uniform" and n_anchor == 0:
        problems.append("NO frame got an anchor -- the cache or npz is not being read, and "
                        "this arm is training its fallback under another name.")
    if arm != "uniform" and n_fallback == n_anchor + n_fallback:
        problems.append("every frame fell back; the anchor source is empty.")
    if arm != "uniform" and n_anchor and not concentrations:
        # Silence here is not a pass. It means no frame had a single cloud point within one
        # sigma of its anchor, which is either an anchor in the wrong coordinate frame or a
        # target outside the workspace crop -- and the arm would train as a uniform draw.
        problems.append("no frame had any point within 1 sigma of its anchor -- the anchor "
                        "and the cloud are probably not in the same frame.")
    if concentrations and np.median(concentrations) < MIN_CONCENTRATION:
        problems.append(f"draw is barely concentrated "
                        f"({np.median(concentrations):.2f}x uniform) -- check sigma/floor.")
    if arm == "molmo" and cfg.get("molmo_view_select") == "closest_gt" and n_centres \
            and max(n_centres) > len(cfg.get("molmo_anchor_ids", [0])):
        problems.append(f"closest_gt should give one centre per queried id "
                        f"({len(cfg.get('molmo_anchor_ids', [0]))}), got up to "
                        f"{max(n_centres)} -- this looks like the per-view cache.")

    print()
    if problems:
        for p in problems:
            print(f"  PROBLEM: {p}")
        raise SystemExit(1)
    print("  OK")


if __name__ == "__main__":
    main()

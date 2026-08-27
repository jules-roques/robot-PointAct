"""Dump per-point feature cosine similarity at full resolution, before any downsampling.

The companion question to attention_saliency.py. That script asks "does the action attend
more to some points than others" and answers no. This one asks the prior question: does the
network's *representation* of the cloud distinguish anything at all at the resolution a warp
would operate on? A warp -- or any learned sampler placed at the input -- can only merge or
keep points on the basis of the features available where it sits, which is stage 0 of the PTv3
encoder, before the first GridPooling drops the cloud to a quarter of its size.

So: pick a handful of query points, and colour every other point by cosine similarity to the
query's feature. If the drawer handle's neighbourhood lights up and the floor does not, there
is structure to exploit even where the attention map is flat. If the handle looks like the
floor, there is nothing for the sampler to read.

WHICH TAPS
----------
`enc_channels[0] = 54`, `enc_depths[0] = 3`, and stage 0 is the only stage at full resolution
(`enc1` starts with a GridPoolingWithAction at stride 2). Every full-resolution tap is dumped:

    embed              the linear stem alone -- xyz+rgb projected, no context of any kind.
                       The control: whatever this map shows is available from raw colour and
                       position, so anything the later taps do NOT add is not learned structure.
    enc0.block0..2     serialized self-attention + MLP, the point stream only.
    enc0.ca_block0..2  cross-attention to the language context. This is where task identity
                       can first reach a point feature, so a handle that only becomes special
                       after a ca_block is special *because of the instruction*, which is a
                       different and much more interesting claim than a colour blob.

Point order is preserved throughout: the stem is per-point, and the attention blocks restore
input order with `feat[inverse]` before returning. So row i of every tap is point i of the
cloud the model was handed, and of the label array aligned to it.

QUERY POINTS
------------
  * eef -- the gripper, from the frame's own state. The trivially-available prior.
  * handle -- label 4 in the points_3views_labels LMDB (falling back to label 3, the drawer
    panel, when the handle is occluded), the same ground truth the oracle sampling arm uses.
  * three uniformly random points, as the null: if a random floor point's similarity map looks
    like the handle's, the map is measuring the encoder's global geometry, not the task.

`augment_pc_rot` is forced to 0 so the labels, which are stored against the unrotated cloud,
stay aligned with what the model sees. The random 0-20% dropout in augment_point_cloud is NOT
disabled -- it is what training actually feeds the model -- so the returned cloud is a subset
of the labelled one, recovered by nearest-neighbour matching with an assertion that every
match is exact to 0.1 mm.

    python experiments/13_robocasa365/feature_similarity.py \
        $SCRATCH/PointAct_exprs/robocasa365/ablation/od-none-s0/checkpoint-final-50000 \
        --frames 4 --out featsim.npz
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from lerobot.constants import OBS_STATE  # noqa: E402

from pointact.data.robot.multi_data import load_single_lerobot_dataset  # noqa: E402
from pointact.data.schema import LerobotConfig  # noqa: E402

from attention_saliency import build_batch  # noqa: E402

HANDLE_LABEL = 4
PANEL_LABEL = 3
LABEL_NAMES = ["background", "robot", "fixture", "panel", "handle"]


def find_ptv3(model):
    for mod in model.modules():
        if type(mod).__name__.startswith("PointTransformerV3CA"):
            return mod
    raise SystemExit("no PointTransformerV3CA* backbone found -- wrong model class?")


def instrument(model):
    """Capture point.feat at every full-resolution tap of encoder stage 0."""
    ptv3 = find_ptv3(model)
    taps: list[tuple[str, torch.nn.Module]] = [("embed", ptv3.embedding)]
    # Stage 0 has no `down` child (GridPooling is added only for s > 0), so every child here
    # runs at full resolution. Taking them in declaration order keeps the dump in depth order.
    for name, child in ptv3.enc.enc0.named_children():
        taps.append((f"enc0.{name}", child))

    feats: dict[str, torch.Tensor] = {}

    def make_hook(name):
        def hook(_module, _inp, out):
            # These modules take and return the same Point object, mutating it in place, but
            # every block REASSIGNS point.feat rather than writing into it, so the tensor
            # captured here is not overwritten by the next block. Detach anyway; float32
            # because the model runs in bf16 and cosine similarity in bf16 quantises to
            # about 3 decimal digits, which is coarser than the differences being measured.
            feats[name] = out.feat.detach().float()
        return hook

    for name, mod in taps:
        mod.register_forward_hook(make_hook(name))
    return [name for name, _ in taps], feats


def align_labels(cropped_xyz, cropped_labels, model_xyz, center):
    """Map the model's (subsampled, centred) cloud back onto the labelled one.

    augment_point_cloud draws int(len * U(0.8, 1.0)) points and does not report which, and
    centre subtraction is applied afterwards. With rotation augmentation off, xyz is otherwise
    untouched, so undoing the centre and matching nearest neighbours recovers the mapping
    exactly -- and the assertion below is what makes "exactly" a checked claim rather than an
    assumption. The cloud is voxelised at 1 cm, so a correct match is at machine precision and
    a wrong one is >= 1 cm away; there is no ambiguous middle to worry about.
    """
    from scipy.spatial import cKDTree

    tree = cKDTree(cropped_xyz)
    dist, idx = tree.query(model_xyz + center, k=1)
    if dist.max() > 1e-4:
        raise SystemExit(
            f"label alignment failed: max nearest-neighbour distance {dist.max():.2e} m. "
            "Is augment_pc_rot really 0?")
    return cropped_labels[idx]


def pick_queries(cloud_xyz, labels, eef, rng, n_random=3):
    """(name, point index) for each query point, in a fixed order."""
    queries: list[tuple[str, int]] = []

    queries.append(("eef", int(np.argmin(np.linalg.norm(cloud_xyz - eef, axis=1)))))

    for label, name in ((HANDLE_LABEL, "handle"), (PANEL_LABEL, "panel")):
        mask = labels == label
        if mask.any():
            centroid = cloud_xyz[mask].mean(axis=0)
            # Nearest labelled point to the centroid, not the nearest point overall: the
            # centroid of a C-shaped handle can land in the air between its prongs, and the
            # point closest to *that* may belong to the drawer front behind it.
            cand = np.flatnonzero(mask)
            queries.append((name, int(cand[np.argmin(
                np.linalg.norm(cloud_xyz[cand] - centroid, axis=1))])))
            break
    else:
        print("  WARNING: neither handle nor panel visible in this frame")

    for j in range(n_random):
        queries.append((f"random{j}", int(rng.integers(len(cloud_xyz)))))
    return queries


def pairwise_summary(feat, rng, n=2000):
    """Percentiles of the cosine similarity between random point PAIRS.

    The control for every map in this dump. If two points drawn at random already sit at 0.9
    cosine, a query map that peaks at 0.95 is not showing selectivity -- it is showing a
    representation whose directions barely vary, and a sampler reading it has nothing to go on.
    """
    idx = rng.choice(feat.shape[0], size=min(n, feat.shape[0]), replace=False)
    sub = torch.nn.functional.normalize(feat[idx], dim=-1)
    sim = (sub @ sub.T).cpu().numpy()
    iu = np.triu_indices(len(idx), k=1)
    vals = sim[iu]
    return np.percentile(vals, [1, 5, 25, 50, 75, 95, 99]).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--frames", type=int, default=4)
    ap.add_argument("--out", type=Path, default=Path("featsim.npz"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--label-dirname", default="points_3views_labels")
    args = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    from run_server import MODEL_MAP  # noqa: E402

    cfg = json.load(open(args.checkpoint / "config.json"))
    name = cfg["architectures"][0]
    model_class, _ = MODEL_MAP[name]
    print(f"loading {name} from {args.checkpoint}")
    model = model_class.from_pretrained(args.checkpoint, device_map={"": "cuda"}).eval()

    data_cfg = yaml.safe_load(open(args.checkpoint.parent / "data_config.yaml"))
    ds_cfg = dict(data_cfg["lerobot_datasets"][0])
    # The labels live in their own LMDB and the Stage 6 arm never opened it (it has no
    # sampler). Naming the directory is enough: it is opened lazily by load_point_labels and
    # changes nothing else about the dataset.
    ds_cfg["oracle_label_dirname"] = args.label_dirname
    # See align_labels: the labels are stored against the unrotated cloud.
    ds_cfg["augment_pc_rot"] = 0

    chunk = model.config.action_chunk_size
    max_action_dim = model.config.max_action_dim
    max_state_dim = model.config.max_state_dim
    ds = load_single_lerobot_dataset(0, [LerobotConfig(**ds_cfg)], chunk_size=chunk)
    print(f"dataset: {ds.num_frames} frames; sampling {args.frames}")

    # The state in the returned item has been normalised; the value wanted here is the
    # centred-but-unnormalised eef position, which exists only between center_point_cloud and
    # normalize_state_action. Grab it on the way past rather than trying to invert the norm.
    original_norm = ds.normalize_state_action

    def wrapped_norm(item, _o=original_norm):
        ds._viz_eef = item[OBS_STATE][:3].clone().numpy().astype(np.float64)
        return _o(item)

    ds.normalize_state_action = wrapped_norm

    tap_names, feats = instrument(model)
    print(f"taps ({len(tap_names)}): {', '.join(tap_names)}")

    rng = np.random.default_rng(args.seed)
    idxs = rng.choice(ds.num_frames, size=min(args.frames, ds.num_frames),
                      replace=False).tolist()

    out: dict[str, np.ndarray] = {}
    all_names: list[str] = []
    for f, idx in enumerate(idxs):
        feats.clear()
        item = ds.hf_dataset[idx]
        ep, fr = int(item["episode_index"]), int(item["frame_index"])
        # The labelled cloud, cropped exactly as the dataset crops it.
        cropped, cropped_labels = ds.filter_point_cloud_by_workspace(
            ds.load_point_cloud(ep, fr), ds.load_point_labels(ep, fr))

        batch = build_batch(ds, idx, max_action_dim, max_state_dim, "cuda")
        with torch.no_grad():
            model.compute_action(
                points=batch["points"], npoints_in_batch=batch["npoints_in_batch"],
                ctx_embeds=batch["ctx_embeds"], ctx_lens=batch["ctx_lens"],
                states=batch["states"])

        cloud = batch["points"].detach().float().cpu().numpy()
        xyz, rgb = cloud[:, :3].astype(np.float64), cloud[:, 3:6]
        center = np.asarray(batch["point_center"], dtype=np.float64)
        labels = align_labels(cropped[:, :3].astype(np.float64), cropped_labels, xyz, center)
        eef = ds._viz_eef

        queries = pick_queries(xyz, labels, eef, rng)
        counts = {LABEL_NAMES[k] if k < len(LABEL_NAMES) else str(k): int((labels == k).sum())
                  for k in np.unique(labels)}
        print(f"  frame {f} (idx {idx}, ep {ep} fr {fr}): {len(xyz)} points, labels {counts}")
        print(f"    queries: " + ", ".join(
            f"{n}@{i}({LABEL_NAMES[labels[i]]})" for n, i in queries))

        out[f"f{f}_coord"] = xyz.astype(np.float32)
        out[f"f{f}_rgb"] = rgb.astype(np.float32)
        out[f"f{f}_labels"] = labels.astype(np.uint8)
        out[f"f{f}_eef"] = eef.astype(np.float32)
        out[f"f{f}_center"] = center.astype(np.float32)
        out[f"f{f}_idx"] = np.array([idx, ep, fr])
        out[f"f{f}_query_idx"] = np.array([i for _, i in queries])
        # Per frame, not global: the second query is "handle" when the handle is visible and
        # "panel" when it is not, and a viewer that assumed one name for all frames would
        # silently mislabel the fallback frames.
        out[f"f{f}_query_names"] = np.array([n for n, _ in queries])
        all_names = [n for n, _ in queries]

        for tap in tap_names:
            feat = feats[tap]
            if feat.shape[0] != len(xyz):
                raise SystemExit(f"tap {tap} has {feat.shape[0]} rows for {len(xyz)} points "
                                 "-- that tap is not at full resolution")
            unit = torch.nn.functional.normalize(feat, dim=-1)
            qi = torch.tensor([i for _, i in queries], device=unit.device)
            # [N, Q] cosine similarity of every point to every query point.
            sims = (unit @ unit[qi].T).cpu().numpy().astype(np.float16)
            out[f"f{f}_{tap}_sim"] = sims
            out[f"f{f}_{tap}_pairwise"] = pairwise_summary(feat, rng)
            # Per-class mean similarity to each query: the number behind the picture.
            per_class = np.full((len(LABEL_NAMES), len(queries)), np.nan, dtype=np.float32)
            for k in range(len(LABEL_NAMES)):
                mask = labels == k
                if mask.any():
                    per_class[k] = sims[mask].astype(np.float32).mean(axis=0)
            out[f"f{f}_{tap}_per_class"] = per_class

    out["taps"] = np.array(tap_names)
    out["query_names"] = np.array(all_names)
    out["label_names"] = np.array(LABEL_NAMES)
    out["n_frames"] = np.array([len(idxs)])
    np.savez_compressed(args.out, **out)
    print(f"\nwrote {args.out} ({args.out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()

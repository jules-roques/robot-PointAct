"""Dump point-feature cosine similarity at every resolution the encoder passes through.

The companion question to attention_saliency.py. That script asks "does the action attend more
to some points than to others". This one asks the prior question: does the network's
*representation* distinguish anything a sampler could act on? Pick a query point, colour every
other point by cosine similarity to it, and look.

WHICH TAPS, AND WHY THESE ONES
------------------------------
The encoder is five stages, `enc_depths = (3, 3, 3, 12, 3)`, with a GridPooling at stride 2
opening each stage after the first. The interesting moment is the state of the cloud
*immediately before each pooling* -- that is the representation any merge-or-keep decision
would have to be made from, at each of the resolutions where such a decision exists. So the
taps are the outputs of `enc0` .. `enc4`: four "about to be halved" snapshots and the encoder's
final output, which is never pooled again.

    embed    the linear stem alone -- a 6->54 projection of (x, y, z, r, g, b), no context of
             any kind. The control: whatever it shows is available from raw colour and
             position, so anything the real taps do not add is not learned structure.
    stage 0  ~19K points, before the first pooling. Where an input-stage warp would act.
    stage 1  ~5.8K   stage 2  ~1.7K   stage 3  ~470   stage 4  ~136, the encoder's output.

LABELS THROUGH THE POOLING CHAIN
--------------------------------
Ground truth exists only at full resolution. GridPooling keeps `pooling_inverse` -- the cluster
index of every parent point -- so the labels are carried down *exactly* rather than matched to
a nearest neighbour. A pooled point is a voxel cell holding several stage-0 points and has no
single label, so both readings are kept: the majority class, for a display colour, and "does
this cell contain any handle point at all", which is the positive set a sampler would have to
keep and the only one that stays meaningful when a stage-4 cell is far bigger than the handle.

Query points are carried down the same chain, so the "handle" query at stage 4 is the cell that
literally contains the stage-0 handle point, not the nearest thing to where it used to be.

QUERY POINTS
------------
  * eef -- the gripper, from the frame's own state. The trivially-available prior.
  * handle -- label 4 (falling back to label 3, the drawer panel, when the handle is occluded),
    the same ground truth the oracle sampling arm uses.
  * three uniformly random points, as the null: if a random floor point's map looks like the
    handle's, the map is measuring the encoder's global geometry, not the task.

`augment_pc_rot` is forced to 0 so the labels, which are stored against the unrotated cloud,
stay aligned with what the model sees. The random 0-20% dropout in augment_point_cloud is NOT
disabled -- it is what training actually feeds the model -- so the returned cloud is a subset of
the labelled one, recovered by nearest-neighbour matching with an assertion that every match is
exact to 0.1 mm.

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

from attention_saliency import (align_labels, build_batch, propagate_labels,  # noqa: E402
                                track_pooling)

HANDLE_LABEL = 4
PANEL_LABEL = 3
LABEL_NAMES = ["background", "robot", "fixture", "panel", "handle"]


def find_ptv3(model):
    for mod in model.modules():
        if type(mod).__name__.startswith("PointTransformerV3CA"):
            return mod
    raise SystemExit("no PointTransformerV3CA* backbone found -- wrong model class?")


def instrument(model):
    """Capture point.feat and point.coord immediately before each pooling, plus the stem."""
    ptv3 = find_ptv3(model)
    taps: list[tuple[str, int, torch.nn.Module]] = [("embed", 0, ptv3.embedding)]
    for name, child in ptv3.enc.named_children():
        # enc{s}'s output is the cloud as it stands just before enc{s+1}'s GridPooling --
        # and for the last stage, the encoder's own output.
        taps.append((f"stage{name[3:]}", int(name[3:]), child))

    grabbed: dict[str, dict] = {}

    def make_hook(name):
        def hook(_module, _inp, out):
            # These modules mutate one Point in place, but every block REASSIGNS point.feat,
            # and GridPooling builds a fresh Point rather than editing its parent -- so what is
            # captured here survives the stages that follow. float32 because the model runs in
            # bf16, which quantises cosine similarity to about three decimal digits, coarser
            # than the differences being measured.
            grabbed[name] = dict(feat=out.feat.detach().float(),
                                 coord=out.coord.detach().float().cpu().numpy())
        return hook

    for name, _, mod in taps:
        mod.register_forward_hook(make_hook(name))
    return [(n, s) for n, s, _ in taps], grabbed


def pick_queries(cloud_xyz, labels, eef, rng, n_random=3):
    """(name, point index) for each query point at full resolution, in a fixed order."""
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


def counterfactual_contexts(ds, true_task: str, swap_task: str | None):
    """(label, embeds) for each alternative instruction to re-run the same frame under.

    Two of them, deliberately different in kind. The NEAR one is the other instruction in this
    task's own cache -- on OpenDrawer that is "Open the left drawer." against "Open the right
    drawer.", the minimal semantic change the policy must actually resolve, and the one whose
    answer decides whether a sampler could ever be steered. The FAR one is another task's
    instruction entirely, which bounds the effect from above: if even that does not move the
    point features, nothing will.
    """
    # The control comes first and is the frame's OWN instruction, re-run unchanged. Two
    # forwards of the same input must agree bit for bit in eval mode, so anything below 1.0000
    # here is nondeterminism, and every other number in this block would be measuring that
    # instead of the instruction.
    alts = [("control: " + true_task, ds.text_context[true_task])]
    others = sorted(k for k in ds.text_context if k != true_task)
    if others:
        alts.append(("near: " + others[0], ds.text_context[others[0]]))
    if swap_task:
        path = Path(ds.root).parent / swap_task / "text_context" / "qwen2.5-vl-3b.pt"
        if path.exists():
            d = torch.load(path, map_location="cpu", weights_only=True)
            k = sorted(d)[0]
            alts.append(("far: " + k, d[k]))
        else:
            print(f"  (no swap cache at {path}; far counterfactual skipped)")
    return alts


def pairwise_summary(feat, rng, n=2000):
    """Percentiles of the cosine similarity between random point PAIRS.

    The control for every map in this dump. If two points drawn at random already sit at 0.9
    cosine, a query map that peaks at 0.95 is not showing selectivity -- it is showing a
    representation whose directions barely vary, and a sampler reading it has nothing to go on.
    """
    idx = rng.choice(feat.shape[0], size=min(n, feat.shape[0]), replace=False)
    sub = torch.nn.functional.normalize(feat[idx], dim=-1)
    sim = (sub @ sub.T).cpu().numpy()
    vals = sim[np.triu_indices(len(idx), k=1)]
    return np.percentile(vals, [1, 5, 25, 50, 75, 95, 99]).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--frames", type=int, default=4)
    ap.add_argument("--out", type=Path, default=Path("featsim.npz"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--label-dirname", default="points_3views_labels")
    # The decisive test of whether the instruction reaches the point features at all: re-run
    # the identical cloud under a different instruction and see whether the features move.
    ap.add_argument("--swap-task", default="TurnOnMicrowave",
                    help="sibling dataset whose instruction is used as the far counterfactual; "
                         "the near one is always the other instruction in this task's own cache")
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

    # The frame's instruction string, needed to know which OTHER instruction is the
    # counterfactual. It is resolved inside __getitem__ by select_task_text and never returned,
    # so take it where it is used.
    original_lookup = ds.lookup_text_context

    def wrapped_lookup(task, _o=original_lookup):
        ds._viz_task = task
        return _o(task)

    ds.lookup_text_context = wrapped_lookup

    taps, grabbed = instrument(model)
    inverses = track_pooling(model)
    print(f"taps ({len(taps)}): {', '.join(n for n, _ in taps)}")

    rng = np.random.default_rng(args.seed)
    idxs = rng.choice(ds.num_frames, size=min(args.frames, ds.num_frames),
                      replace=False).tolist()

    out: dict[str, np.ndarray] = {}
    for f, idx in enumerate(idxs):
        grabbed.clear(); inverses.clear()
        item = ds.hf_dataset[idx]
        ep, fr = int(item["episode_index"]), int(item["frame_index"])
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
        centre = np.asarray(batch["point_center"], dtype=np.float64)
        base = align_labels(cropped[:, :3].astype(np.float64), cropped_labels, xyz, centre)
        counts = propagate_labels(base, inverses, len(LABEL_NAMES))
        eef = ds._viz_eef

        queries = pick_queries(xyz, base, eef, rng)
        qidx0 = [i for _, i in queries]
        # Carry the queries down the same chain the labels take, so "the handle" at stage 4 is
        # the cell that actually contains the stage-0 handle point.
        qidx = [qidx0]
        for inv in inverses:
            qidx.append([int(inv[i]) for i in qidx[-1]])

        cls = {LABEL_NAMES[k]: int((base == k).sum()) for k in np.unique(base)}
        print(f"  frame {f} (idx {idx}, ep {ep} fr {fr}): {len(xyz)} points, labels {cls}")
        print("    queries: " + ", ".join(
            f"{n}@{i}({LABEL_NAMES[base[i]]})" for n, i in queries))
        print(f"    task: {ds._viz_task!r}")
        print("    sizes: " + ", ".join(f"{n}={len(grabbed[n]['coord'])}" for n, _ in taps))

        out[f"f{f}_rgb"] = rgb.astype(np.float32)
        out[f"f{f}_eef"] = eef.astype(np.float32)
        out[f"f{f}_center"] = centre.astype(np.float32)
        out[f"f{f}_idx"] = np.array([idx, ep, fr])
        out[f"f{f}_query_names"] = np.array([n for n, _ in queries])

        for tap, stage in taps:
            g = grabbed[tap]
            cnt = counts[stage]
            if len(cnt) != len(g["coord"]):
                raise SystemExit(f"label chain gives {len(cnt)} points at {tap}, "
                                 f"the tap has {len(g['coord'])}")
            qi = qidx[stage]
            unit = torch.nn.functional.normalize(g["feat"], dim=-1)
            sims = (unit @ unit[torch.tensor(qi, device=unit.device)].T).cpu().numpy()
            out[f"f{f}_{tap}_coord"] = g["coord"].astype(np.float32)
            out[f"f{f}_{tap}_label"] = cnt.argmax(axis=1).astype(np.uint8)
            out[f"f{f}_{tap}_has_handle"] = (cnt[:, HANDLE_LABEL] > 0).astype(np.uint8)
            out[f"f{f}_{tap}_has_robot"] = (cnt[:, 1] > 0).astype(np.uint8)
            out[f"f{f}_{tap}_qidx"] = np.array(qi)
            out[f"f{f}_{tap}_sim"] = sims.astype(np.float16)
            out[f"f{f}_{tap}_pairwise"] = pairwise_summary(g["feat"], rng)

        # --- counterfactual instructions, same cloud ---------------------------------
        true_feat = {tap: grabbed[tap]["feat"] for tap, _ in taps}
        alts = counterfactual_contexts(ds, ds._viz_task, args.swap_task)
        for a, (alabel, aembed) in enumerate(alts):
            grabbed.clear()
            with torch.no_grad():
                model.compute_action(
                    points=batch["points"], npoints_in_batch=batch["npoints_in_batch"],
                    ctx_embeds=aembed.unsqueeze(0).to("cuda"),
                    ctx_lens=torch.LongTensor([len(aembed)]).to("cuda"),
                    states=batch["states"])
            out[f"f{f}_cf{a}_label"] = np.array(alabel)
            for tap, _ in taps:
                # Per-point cosine between the two runs' features. 1.0 to the last bit means
                # the instruction never reached this tap; anything less is the size of the
                # channel, measured rather than argued about.
                cs = torch.nn.functional.cosine_similarity(
                    true_feat[tap], grabbed[tap]["feat"], dim=-1).cpu().numpy()
                out[f"f{f}_cf{a}_{tap}_cos"] = cs.astype(np.float32)
            print(f"    counterfactual [{alabel}]: " + "  ".join(
                f"{tap} {out[f'f{f}_cf{a}_{tap}_cos'].mean():.4f}" for tap, _ in taps))
        out[f"f{f}_n_cf"] = np.array([len(alts)])

    out["taps"] = np.array([n for n, _ in taps])
    out["tap_stage"] = np.array([s for _, s in taps])
    out["label_names"] = np.array(LABEL_NAMES)
    out["n_frames"] = np.array([len(idxs)])
    np.savez_compressed(args.out, **out)
    print(f"\nwrote {args.out} ({args.out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()

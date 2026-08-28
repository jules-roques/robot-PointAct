"""Dump per-point attention-to-action saliency from a trained PointAct checkpoint.

The Stage 6 question: does any point in the cloud draw systematically more attention from the
action tokens than its neighbours? If yes, that map is the saliency a Learning-to-Zoom-style
warp would use at the pooling step, and it costs nothing to compute because the model already
computes it. If the map is flat, there is nothing to warp toward and the idea is dead.

WHAT IS ACTUALLY MEASURED, and why it is not quite EViT's recipe
---------------------------------------------------------------
`SerializedAttentionWithAction` prepends the action tokens to every serialized patch, so one
patch attends over `[A actions | K points]` as a SINGLE softmax. A point token's row therefore
spends a fraction of its probability mass on the action keys, and that fraction is a per-point
scalar in [0, 1]: "how much this point cares about the action".

EViT reads the opposite direction -- the class token's row over patches -- and that direction
IS available here, contrary to what an earlier version of this file claimed. The action tokens
are replicated per patch (`action_qkv.repeat_interleave(repeat_size)`), so action->point is a
separate softmax in each patch; but the replicas are recombined by a PLAIN MEAN
(model_ca_action.py:143-148), which makes point j's coefficient in action token a exactly
(1/P) * attn_p[a, j] -- one global linear map with a constant factor, hence comparable across
the whole cloud. Both directions are therefore dumped, in two forms each:

  * softmax mass (`ap_*` for action->point, `total`/`state`/`steps` for point->action).
    Point->action carries a per-patch OFFSET, because each point's denominator includes its
    own patch's K point keys; serialised patches are spatially compact, so that offset paints
    as spatial bands that are easy to mistake for structure. `patch` is stored so between- and
    within-patch variance can be separated rather than guessed at.
  * raw logits (`a2p_*` for action->point, `logit_*` for point->action), which have no
    denominator at all and so no offset. a2p is the cleanest of the four: q_a is literally the
    same vector in every patch, so q_a . k_i depends on nothing but point i.

One asymmetry survives in every logit here and is worth naming: RoPE rotates the POINT q/k by
coordinate (`self.rope(q, k, rope_coord)`) and the action qkv never passes through it, so a
positional phase gradient rides along with whatever semantic signal exists. Correlate any map
against the coordinate axes before calling it task structure.

The A action tokens are NOT interchangeable, which is why they are never summed blindly:
  * index 0 is the STATE embedding (the gripper pose) -- genuinely frame-dependent;
  * indices 1..T are learned position embeddings of the chunk step, identical across frames at
    the input and only made frame-specific by the layers below.
So "mass on the state token" and "mass on step t" are different questions, and a saliency that
turns out to be all state mass is just the eef prior rediscovered, not a new signal. Both are
dumped separately, per step, so the two can be told apart.

Run under `enable_flash=False`: the flash kernel never materialises the attention matrix. The
non-flash branch sets patch_size = min(min points, patch_size_max), which for a ~19K cloud is
the same 1024 training used, so this reads the trained behaviour rather than a different one.

    python experiments/13_robocasa365/attention_saliency.py \
        $SCRATCH/PointAct_exprs/robocasa365/ablation/od-none-s0/checkpoint-20000 \
        --frames 8 --out saliency.npz
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

from lerobot.constants import OBS_STATE  # noqa: E402

from pointact.constants import OBS_POINTS  # noqa: E402
from pointact.data.robot.multi_data import load_single_lerobot_dataset  # noqa: E402
from pointact.data.schema import LerobotConfig  # noqa: E402


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


def track_pooling(model):
    """Record each GridPooling's parent->child map, so labels can be carried down exactly.

    `pooling_inverse` is the `cluster` tensor torch.unique returns when the stage builds its
    voxel grid: entry i is the index of the pooled point that stage-0 point i was merged into.
    Chaining the four of them gives an exact ancestry for every point at every stage -- strictly
    better than matching a pooled centroid to its nearest labelled neighbour, which is a guess
    that gets worse at each stage as the cells grow.
    """
    inverses: list = []
    mods = [m for m in model.modules() if type(m).__name__ == "GridPoolingWithAction"]
    if len(mods) != 4:
        raise SystemExit(f"expected 4 pooling stages, found {len(mods)}")
    for mod in mods:
        mod.register_forward_hook(
            lambda _m, _i, out: inverses.append(out.pooling_inverse.detach().cpu().numpy()))
    return inverses


def propagate_labels(base, inverses, n_classes=5):
    """Per-stage label counts, from stage-0 ground truth down through the pooling chain.

    Returns one [n_points, n_classes] count matrix per stage, stage 0 first. A pooled point is
    a voxel cell holding several stage-0 points, so it has no single label: keep the counts and
    let the caller decide. `argmax` is the honest choice for a display colour; `counts[:, 4] > 0`
    -- does this cell contain any handle point at all -- is the honest positive set for scoring,
    because that is exactly the set a sampler would have to keep.
    """
    counts = np.zeros((len(base), n_classes), dtype=np.int64)
    counts[np.arange(len(base)), base.astype(int)] = 1
    out = [counts]
    for inv in inverses:
        m = int(inv.max()) + 1
        counts = np.stack([np.bincount(inv, weights=counts[:, c], minlength=m)
                           for c in range(n_classes)], axis=1).astype(np.int64)
        out.append(counts)
    return out


def instrument(model):
    """Record point->action attention mass at every serialized-attention layer.

    Hooks `self.softmax` rather than replacing `forward`: the attention matrix is a local
    variable inside forward, but it is exactly that module's output, so a forward hook on it
    reads the real thing with no copy of the model code to drift out of sync.
    """
    records: list[dict] = []
    mods = [m for m in model.modules() if type(m).__name__ == "SerializedAttentionWithAction"]
    if not mods:
        raise SystemExit("no SerializedAttentionWithAction layers found -- wrong model class?")

    for layer, mod in enumerate(mods):
        # The flash and non-flash branches are configured differently at construction
        # (model.py:188-196), so flipping the flag alone leaves the non-flash path missing
        # two attributes it needs:
        #   * patch_size_max is only set in the non-flash branch. Seed it from the flash
        #     branch's patch_size so the non-flash path derives the SAME 1024 training used
        #     -- otherwise this would read attention at a different window size.
        #   * attn_drop is a bare float under flash but is CALLED as a module here.
        # enable_rpe / upcast_* are asserted False whenever flash is on, so the non-flash
        # path skips the RPE entirely and this reads the same attention training computed.
        mod.patch_size_max = mod.patch_size
        if not isinstance(mod.attn_drop, torch.nn.Module):
            mod.attn_drop = torch.nn.Dropout(0.0)
        mod.enable_flash = False

        # `pad` is what maps padded/serialised rows back to input point indices, and it is
        # produced inside forward. Wrap the method that makes it instead of recomputing it
        # afterwards (recomputing risks reading a cached value from a later layer).
        original = mod.get_padding_and_inverse

        def wrapped(point, _m=mod, _o=original):
            out = _o(point)
            _m._viz_pad, _m._viz_point = out[0], point
            return out

        mod.get_padding_and_inverse = wrapped

        def hook(_module, _inp, out, _m=mod, _layer=layer):
            # _inp[0] is the PRE-softmax logit tensor, same shape as `out`.
            # out: [patches, heads, A + K, A + K], each row a softmax over [actions | points].
            point = _m._viz_point
            A = point.action_feat.size(1)
            P, H = out.shape[0], out.shape[1]
            K = out.shape[-1] - A

            # --- raw logits: the only patch-INDEPENDENT quantity here -------------------
            # q_i . k_a does not depend on which patch point i landed in, because the action
            # keys are one tensor repeat_interleave'd into every patch -- identical k_a
            # everywhere. The softmax mass is not patch-independent (its denominator is the
            # patch's own point keys), so a patch that barely attends to the action at all
            # still hands its points a full unit of mass to divide up. These do not.
            lg = _inp[0][:, :, A:, :A].float()                  # [P, H, K, A]
            logit_max = lg.max(dim=-1).values.mean(dim=1)       # [P, K]
            logit_mean = lg.mean(dim=-1).mean(dim=1)            # [P, K]

            # --- the action stream: what the action tokens are actually built FROM ---------
            # Each action token's row is a softmax within its patch, but the replicas are
            # combined by a PLAIN MEAN over patches (model_ca_action.py:143-148), so the
            # coefficient of point j in action token a is (1/P) * attn_p[a, j] -- one global
            # linear map, with 1/P a constant that drops out of any ranking. A patch whose
            # action rows spend their mass on the other action tokens rather than on points
            # correctly yields small values for all of its points.
            ap = out[:, :, :A, A:].float().mean(dim=1)          # [P, A, K] mean over heads
            ap_sum = ap.sum(dim=1)                              # [P, K] over all action tokens
            ap_max = ap.max(dim=1).values                       # [P, K] peak over tokens
            ap_state = ap[:, 0, :]                              # [P, K] the gripper-pose token

            # --- action -> point, RAW LOGITS ------------------------------------------
            # The mirror of `lg` above, and the block the action is literally assembled
            # from: row a, column i is q_a . k_i * scale, one row per action token and one
            # column per point. It is patch-independent for a stronger reason than the
            # point->action logits are: the action QUERIES are one tensor
            # repeat_interleave'd into every patch, so q_a is the same vector everywhere,
            # and k_i is point i's own key. Point i's value therefore does not depend on
            # which patch it landed in at all -- no per-patch softmax denominator, no
            # band artefact, nothing to renormalise away afterwards.
            #
            # Taking the MAX over action tokens rather than the sum asks "is this point
            # important to ANY moment of the chunk", which is the question a warp cares
            # about; a point that matters only at the instant of contact would be averaged
            # into the floor by a mean over 17 tokens.
            a2p = _inp[0][:, :, :A, A:].float()                 # [P, H, A, K]
            a2p_max = a2p.max(dim=2).values.mean(dim=1)         # [P, K] max over actions
            a2p_mean = a2p.mean(dim=2).mean(dim=1)              # [P, K]
            # Mean-over-heads is exactly what hides ONE selective head among H flat ones,
            # so keep the head-wise peak too rather than concluding "flat" from an average.
            a2p_max_head = a2p.amax(dim=(1, 2))                 # [P, K] max over H and A
            # Which action token peaks. 0 is the state (gripper-pose) embedding, 1..T are
            # the chunk steps -- a map that is all token 0 is the eef prior rediscovered,
            # not a new signal, and that has to be visible rather than assumed.
            a2p_argmax = a2p.mean(dim=1).argmax(dim=1).to(torch.float32)   # [P, K]

            # Rows = point queries, columns = action keys.
            pah = out[:, :, A:, :A].float()                     # [P, H, K, A]
            # Mean over heads, as EViT does...
            pa = pah.mean(dim=1)                                # [P, K, A]
            # ...but keep the heads too. Averaging is exactly what would hide ONE selective
            # head among H flat ones, and "the map is flat" is not a safe conclusion to draw
            # from an average alone.
            tot_h = pah.sum(dim=-1)                             # [P, H, K]
            total = pa.sum(dim=-1)                              # [P, K]
            state = pa[..., 0]                                  # [P, K]  gripper pose token
            steps = pa[..., 1:]                                 # [P, K, T] chunk-step tokens

            # Undo the serialisation: row (p, j) is padded position p*K + j, and `pad` says
            # which input point that is. Padding repeats points, so reduce with amax rather
            # than letting the last write win.
            order = point.serialized_order[_m.order_index][_m._viz_pad]
            n = point.coord.shape[0]

            def scatter(flat):
                dst = torch.zeros(n, dtype=flat.dtype, device=flat.device)
                return dst.scatter_reduce(0, order, flat, reduce="amax", include_self=False)

            # Which patch each point landed in. The action mass carries a per-patch OFFSET --
            # every row is normalised over its own patch's K point keys, so a patch with
            # peaky internal attention gives all its points a higher action mass for reasons
            # that have nothing to do with the task. Serialised patches are spatially
            # compact, so that offset paints as spatial bands and is easy to mistake for
            # structure. Store the id so between-patch and within-patch variance can be
            # separated instead of guessed at.
            pid = (torch.arange(P, device=out.device)
                   .repeat_interleave(K).to(torch.float32))

            rec = {
                "logit_max": scatter(logit_max.reshape(-1)).cpu().numpy(),
                "logit_mean": scatter(logit_mean.reshape(-1)).cpu().numpy(),
                "ap_sum": scatter(ap_sum.reshape(-1)).cpu().numpy(),
                "ap_max": scatter(ap_max.reshape(-1)).cpu().numpy(),
                "ap_state": scatter(ap_state.reshape(-1)).cpu().numpy(),
                "a2p_max": scatter(a2p_max.reshape(-1)).cpu().numpy(),
                "a2p_mean": scatter(a2p_mean.reshape(-1)).cpu().numpy(),
                "a2p_max_head": scatter(a2p_max_head.reshape(-1)).cpu().numpy(),
                "a2p_argmax": scatter(a2p_argmax.reshape(-1)).cpu().numpy(),
                "patch": scatter(pid).cpu().numpy(),
                "layer": _layer,
                "npoints": n,
                "n_actions": A,
                "patch_size": K,
                "n_patches": P,
                "coord": point.coord.detach().float().cpu().numpy(),
                "total": scatter(total.reshape(-1)).cpu().numpy(),
                "state": scatter(state.reshape(-1)).cpu().numpy(),
                # Per chunk step, so "critical at one moment" can be told from "diffusely
                # relevant" -- the mean over steps cannot distinguish those two.
                "per_head": np.stack(
                    [scatter(tot_h[:, h].reshape(-1)).cpu().numpy()
                     for h in range(H)], axis=-1),
                "steps": np.stack(
                    [scatter(steps[..., t].reshape(-1)).cpu().numpy()
                     for t in range(steps.shape[-1])], axis=-1),
            }
            records.append(rec)

        mod.softmax.register_forward_hook(hook)

    return records, mods


def build_batch(ds, idx: int, max_action_dim: int, max_state_dim: int, device):
    """One frame, shaped the way the collator shapes it for a cached-context run."""
    from pointact.data.dataset import collect_robot_tensors

    item = ds[idx]
    _, actions, states, _, points, _ = collect_robot_tensors(
        item, max_action_dim, max_state_dim)
    return {
        "points": points.to(device),
        "npoints_in_batch": torch.LongTensor([points.size(0)]).to(device),
        "ctx_embeds": item["ctx_embeds"].unsqueeze(0).to(device),
        "ctx_lens": torch.LongTensor([len(item["ctx_embeds"])]).to(device),
        "states": states.unsqueeze(0).to(device) if states is not None else None,
        "raw_state": np.asarray(item[OBS_STATE], dtype=np.float64),
        "point_center": np.asarray(item[f"{OBS_POINTS}.center"], dtype=np.float64),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--out", type=Path, default=Path("saliency.npz"))
    ap.add_argument("--seed", type=int, default=0)
    # Off by default: the arm being read here never opens the label LMDB during training,
    # and switching it on forces augment_pc_rot to 0 (see align_labels), which is a change to
    # the model's input -- small, and arguably the cleaner reading, but a change all the same.
    ap.add_argument("--labels", default=None,
                    help="label LMDB dirname under root, e.g. points_3views_labels. Adds "
                         "per-point ground truth so the saliency can be scored against the "
                         "target instead of only described.")
    args = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    from run_server import MODEL_MAP  # noqa: E402

    cfg = json.load(open(args.checkpoint / "config.json"))
    name = cfg["architectures"][0]
    model_class, _ = MODEL_MAP[name]
    print(f"loading {name} from {args.checkpoint}")
    model = model_class.from_pretrained(args.checkpoint, device_map={"": "cuda"}).eval()

    # The run's own archived data config, so the frames are the ones it trained on.
    data_cfg = yaml.safe_load(open(args.checkpoint.parent / "data_config.yaml"))
    ds_cfg = dict(data_cfg["lerobot_datasets"][0])
    if args.labels:
        ds_cfg["oracle_label_dirname"] = args.labels
        ds_cfg["augment_pc_rot"] = 0
    # From the model config, never from a guessed default: the checkpoint ships
    # training_args.BIN, not .json, so an `if exists()` fallback silently takes over and
    # feeds the state encoder the wrong width (max_state_dim is 64 here, not 32 -- the
    # mismatch surfaces as a bmm shape error deep in the action head).
    chunk = model.config.action_chunk_size
    max_action_dim = model.config.max_action_dim
    max_state_dim = model.config.max_state_dim
    print(f"config: chunk={chunk} max_action_dim={max_action_dim} "
          f"max_state_dim={max_state_dim} -> A = 1 + {chunk} action tokens")
    ds = load_single_lerobot_dataset(0, [LerobotConfig(**ds_cfg)], chunk_size=chunk)
    print(f"dataset: {ds.num_frames} frames; sampling {args.frames}")

    records, mods = instrument(model)
    inverses = track_pooling(model) if args.labels else []
    stage_of = sum([[s] * d for s, d in enumerate(model.config.ptv3_enc_depths)], [])
    print(f"instrumented {len(mods)} attention layers (flash disabled)")

    rng = np.random.default_rng(args.seed)
    idxs = rng.choice(ds.num_frames, size=min(args.frames, ds.num_frames),
                      replace=False).tolist()

    out: dict[str, np.ndarray] = {}
    for f, idx in enumerate(idxs):
        records.clear()
        inverses.clear()
        row = ds.hf_dataset[idx]
        ep, fr = int(row["episode_index"]), int(row["frame_index"])
        labelled = None
        if args.labels:
            labelled = ds.filter_point_cloud_by_workspace(
                ds.load_point_cloud(ep, fr), ds.load_point_labels(ep, fr))
        batch = build_batch(ds, idx, max_action_dim, max_state_dim, "cuda")
        with torch.no_grad():
            model.compute_action(
                points=batch["points"], npoints_in_batch=batch["npoints_in_batch"],
                ctx_embeds=batch["ctx_embeds"], ctx_lens=batch["ctx_lens"],
                states=batch["states"])
        print(f"  frame {f} (dataset idx {idx}): {len(records)} layers, "
              f"{records[0]['npoints']} points, A={records[0]['n_actions']}, "
              f"K={records[0]['patch_size']}, patches={records[0]['n_patches']}")
        for rec in records:
            for key in ("coord", "total", "state", "steps", "per_head", "patch",
                        "logit_max", "logit_mean", "ap_sum", "ap_max", "ap_state",
                        "a2p_max", "a2p_mean", "a2p_max_head", "a2p_argmax"):
                out[f"f{f}_l{rec['layer']}_{key}"] = rec[key]
            out[f"f{f}_l{rec['layer']}_meta"] = np.array(
                [rec["npoints"], rec["n_actions"], rec["patch_size"], rec["n_patches"]])
        if labelled is not None:
            cropped, cropped_labels = labelled
            centre = np.asarray(batch["point_center"], dtype=np.float64)
            base = align_labels(cropped[:, :3].astype(np.float64), cropped_labels,
                                records[0]["coord"].astype(np.float64), centre)
            per_stage = propagate_labels(base, inverses)
            for rec in records:
                cnt = per_stage[stage_of[rec["layer"]]]
                if len(cnt) != rec["npoints"]:
                    raise SystemExit(f"label chain gives {len(cnt)} points at layer "
                                     f"{rec['layer']}, attention saw {rec['npoints']}")
                out[f"f{f}_l{rec['layer']}_labels"] = cnt.argmax(axis=1).astype(np.uint8)
                # "this cell contains at least one handle point" -- the set a sampler would
                # have to keep, and the only positive set that stays meaningful once a stage-4
                # point covers a cell far larger than the handle itself.
                out[f"f{f}_l{rec['layer']}_has_handle"] = (cnt[:, 4] > 0).astype(np.uint8)
        out[f"f{f}_state"] = batch["raw_state"]
        out[f"f{f}_center"] = batch["point_center"]
        out[f"f{f}_idx"] = np.array([idx])

    out["n_frames"] = np.array([len(idxs)])
    out["n_layers"] = np.array([len(mods)])
    np.savez_compressed(args.out, **out)
    print(f"\nwrote {args.out} ({args.out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()

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

EViT reads the opposite direction -- the class token's row over patches -- because a ViT has
one CLS token and one global softmax, so those values are comparable across all tokens. That
does not transfer here: the action tokens are REPLICATED per patch
(`action_qkv.repeat_interleave(repeat_size)`), so action->point is a different softmax in every
patch, over a different key set. Point->action is individually normalised per point and is the
comparable direction. It is still not perfect -- each point's denominator includes its own
patch's K points -- so patch-relative rank is reported alongside the raw mass.

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
            # out: [patches, heads, A + K, A + K], each row a softmax over [actions | points].
            point = _m._viz_point
            A = point.action_feat.size(1)
            P, H = out.shape[0], out.shape[1]
            K = out.shape[-1] - A

            # Rows = point queries, columns = action keys. Mean over heads, as EViT does.
            pa = out[:, :, A:, :A].float().mean(dim=1)          # [P, K, A]
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

            rec = {
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
    print(f"instrumented {len(mods)} attention layers (flash disabled)")

    rng = np.random.default_rng(args.seed)
    idxs = rng.choice(ds.num_frames, size=min(args.frames, ds.num_frames),
                      replace=False).tolist()

    out: dict[str, np.ndarray] = {}
    for f, idx in enumerate(idxs):
        records.clear()
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
            for key in ("coord", "total", "state", "steps"):
                out[f"f{f}_l{rec['layer']}_{key}"] = rec[key]
            out[f"f{f}_l{rec['layer']}_meta"] = np.array(
                [rec["npoints"], rec["n_actions"], rec["patch_size"], rec["n_patches"]])
        out[f"f{f}_state"] = batch["raw_state"]
        out[f"f{f}_center"] = batch["point_center"]
        out[f"f{f}_idx"] = np.array([idx])

    out["n_frames"] = np.array([len(idxs)])
    out["n_layers"] = np.array([len(mods)])
    np.savez_compressed(args.out, **out)
    print(f"\nwrote {args.out} ({args.out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()

"""Would this run config's PTv3 checkpoint actually initialise the backbone?

`scripts/train.py:maybe_load_ptv3_checkpoint` loads with ``strict=False`` and drops every
tensor whose name or shape disagrees. A backbone whose architecture arguments do not match its
checkpoint therefore trains from random init, reports it in one INFO line, and produces a loss
curve that looks entirely normal -- the switch from Concerto to Utonia is exactly this shape of
change, because the two differ in channel widths and head counts as well as in the backend.

This builds the backbone the run config asks for, loads the checkpoint the same way training
does, and reports the match rate. Seconds on CPU, against ~20 H100-hours for the run.

    python experiments/13_robocasa365/check_ptv3_init.py \
        experiments/13_robocasa365/runs/s5-tom-eef-n8192-s0.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pointact.model.vla_pointact.action_head_3d.ptv3_backbone import (  # noqa: E402
    get_ptv3_model_cls,
)
from pointact.train.run_config import resolve_run_config  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_config", type=Path)
    ap.add_argument("--min-match", type=float, default=0.9,
                    help="Fail below this fraction of the model's tensors initialised.")
    args = ap.parse_args()

    _meta, _data, train = resolve_run_config(args.run_config)
    backend = train.get("ptv3_backend", "concerto")
    ckpt_file = Path(train["ptv3_init_ckpt_file"])
    print(f"run={train.get('run_name')} backend={backend}")
    print(f"  ckpt={ckpt_file}")
    print(f"  enc_channels={train.get('ptv3_enc_channels')}")
    print(f"  enc_num_head={train.get('ptv3_enc_num_head')}")
    print(f"  enc_depths={train.get('ptv3_enc_depths')}")
    if not ckpt_file.exists():
        raise SystemExit(f"no checkpoint at {ckpt_file}")

    # Same class the trainer builds, with the run's own architecture arguments.
    cls = get_ptv3_model_cls(backend, with_action=False)
    patch = train.get("ptv3_patch_size", 1024)
    n_stages = len(train["ptv3_enc_depths"])
    model = cls(
        in_channels=train.get("ptv3_input_channels", 6),
        enc_depths=tuple(train["ptv3_enc_depths"]),
        enc_channels=tuple(train["ptv3_enc_channels"]),
        enc_num_head=tuple(train["ptv3_enc_num_head"]),
        enc_patch_size=tuple([patch] * n_stages),
        enc_mode=train.get("ptv3_enc_mode", True),
    )
    target_state = model.state_dict()

    checkpoint = torch.load(ckpt_file, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)

    matched, mismatched, absent = [], [], []
    for name, value in state_dict.items():
        if name not in target_state:
            absent.append(name)
        elif value.shape == target_state[name].shape:
            matched.append(name)
        else:
            mismatched.append((name, tuple(value.shape), tuple(target_state[name].shape)))

    frac = len(matched) / max(1, len(target_state))
    print(f"\n  checkpoint tensors : {len(state_dict)}")
    print(f"  model tensors      : {len(target_state)}")
    print(f"  initialised        : {len(matched)} ({frac:.1%} of the model)")
    print(f"  shape mismatches   : {len(mismatched)}")
    print(f"  not in the model   : {len(absent)}")
    for name, s, t in mismatched[:5]:
        print(f"    {name}: ckpt{s} vs model{t}")

    if frac < args.min_match:
        raise SystemExit(
            f"\nPROBLEM: only {frac:.1%} of the backbone would be initialised. The "
            f"architecture arguments and the checkpoint disagree -- check ptv3_backend, "
            f"ptv3_enc_channels and ptv3_enc_num_head against the reference script for this "
            f"backbone.")
    print("\n  OK")


if __name__ == "__main__":
    main()

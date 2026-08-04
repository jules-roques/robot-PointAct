# Python environments

This repo needs **three mutually incompatible Python environments**. They are separate
because their dependency constraints genuinely conflict, not by accident — do not try to
merge them.

| Env | Location | Used for | Why separate |
|---|---|---|---|
| Training / root | `.venv` | Training and evaluating PointAct. torch 2.7 (cu126). | The main environment; `pointact` is installed into it. |
| Simulator | `envs/robocasa365/.venv` | RoboCasa365 / MuJoCo rollouts. Python 3.11. | MuJoCo and the RoboCasa stack pin versions that conflict with the training env. |
| Pointing | `envs/molmo/.venv` | MolmoPoint-8B detection for ROI-guided sampling (`data_prep/roi_sampling/build_molmo_cache.py`). | MolmoPoint ships custom modelling code that requires `transformers==4.57.1`; the training env is on 5.x, and pinning it back would break the trainer. |

## Running things

The simulator env is driven through `uv` without re-syncing:

```bash
uv run --project envs/robocasa365 --no-sync <command>
```

`--no-sync` matters: a sync can re-resolve and break a working environment.

## Running worktree code inside an existing env

The environments have `pointact` installed, so a git worktree's copy of the source is
shadowed by the installed package. To make the worktree's code win, `cd` into the
worktree and prepend its path to `PYTHONPATH`, while pointing `uv` at the *main*
checkout so it finds the existing `.venv`:

```bash
cd "$WORKTREE"
PYTHONPATH="$WORKTREE" uv run --project "$MAIN_CHECKOUT" --no-sync python -m ...
```

Without the `PYTHONPATH` prefix you will run the installed copy and see none of your
changes — with no error to tell you so.

## Adding a dependency

Add it to the environment that actually needs it. If a new dependency forces a torch
version change, that is a signal it belongs in a fourth isolated env, not in the root
one — the root env's torch build is constrained by the oldest GPU you need to train on.

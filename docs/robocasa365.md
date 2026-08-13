# RoboCasa365 integration — end-to-end guide

How to go from a fresh checkout to a trained PointAct policy with a success rate on a
RoboCasa365 task. The worked example throughout is **OpenDrawer**; other tasks differ only in
the task name and the registry date component.

The pipeline has five stages:

```
 1. environments      two uv projects: pointact (training) + robocasa365 (simulator)
 2. assets + demos    kitchen assets and the source demonstrations, from Hugging Face
 3. dataset           replay -> convert -> normalization stats
 4. training          frozen Qwen2.5-VL + trainable PTv3 point-action expert
 5. evaluation        policy server + simulator client, success rate over N trials
```

Reference pages for each part: [`envs/robocasa365/README.md`](../envs/robocasa365/README.md)
(simulator env + downloads), [`data_prep/robocasa365_to_lerobot/README.md`](../data_prep/robocasa365_to_lerobot/README.md)
(dataset construction), [`experiments/13_robocasa365/README.md`](../experiments/13_robocasa365/README.md)
(training + evaluation).

> **Cluster specifics.** The `.slurm` files carry `--account` / `--partition` / `--qos` values
> for the two clusters this was developed on (CLEPS and Jean Zay). Override them on the
> `sbatch` command line for your own allocation; see [`docs/clusters/`](clusters/).

---

## 1. Environments

Two separate [uv](https://docs.astral.sh/uv/) projects, because their dependency sets are
genuinely incompatible: RoboCasa365 pins `numpy==2.2.5` / `mujoco==3.3.1` on Python 3.11, while
the training stack is held at `numpy==1.26.4` by `open3d==0.18.0`.

| Project | Selected with | Used for |
| --- | --- | --- |
| root `pointact` | `uv run ...` | training, conversion, stats, policy server |
| `envs/robocasa365` | `uv run --project envs/robocasa365 ...` | replay, evaluation client |

No activation step — `--project` selects the environment.

```bash
# The simulator is a git submodule pinned to the robocasa365_release branch.
git submodule update --init envs/robocasa365/robocasa

# Training env (needs a GPU node for the cuda extra).
uv sync --extra train --extra cuda

# Simulator env. robosuite is a git dependency, so this first sync needs network access;
# afterwards the uv cache serves offline compute nodes.
uv sync --project envs/robocasa365
```

See [`INSTALLATION.md`](../INSTALLATION.md) for the training environment in detail and
[`docs/envs.md`](envs.md) for which interpreter to use where — picking the wrong one produces
confusing failures rather than clean errors.

### Where the data will live

RoboCasa reads a single base path for everything it downloads. Set it once:

```bash
uv run --project envs/robocasa365 python -m robocasa.scripts.setup_macros
```

then edit `DATASET_BASE_PATH` in `envs/robocasa365/robocasa/robocasa/macros_private.py`:

```python
DATASET_BASE_PATH = "/path/to/scratch/datasets/robot_data/robocasa365"
```

This value has **no environment-variable override** — `robocasa/macros.py` reads only the
module attribute. Left at `None`, datasets land in a `datasets/` directory beside the package.

The rest of this guide refers to that directory as `$ROBOCASA_DATASET_ROOT`; the SLURM scripts
read exactly that variable, so export it to the same value to keep the two consistent:

```bash
export ROBOCASA_DATASET_ROOT=/path/to/scratch/datasets/robot_data/robocasa365
```

Keep `HF_HOME` on large storage too — downloads are staged in the Hugging Face cache before
extraction, so transient usage is roughly twice the final size:

```bash
export HF_HOME=$SCRATCH/.cache/huggingface
```

---

## 2. Assets and demonstrations

### 2a. Kitchen assets (~5 GB)

Required to instantiate any environment, independent of the demonstration data:

```bash
uv run --project envs/robocasa365 python -m robocasa.scripts.download_kitchen_assets
```

This covers objaverse objects, AI-generated objects, and both texture packs. Two gaps in that
script must be filled in by hand, and **both fail late** — as a `FileNotFoundError` on some
`model.xml` deep inside a replay — rather than at download time:

**Gap 1 — the lightwheel packs.** Upstream requests monolithic `objects_lightwheel.zip` /
`fixtures_lightwheel.zip` files that no longer exist; the NVIDIA repo now stores per-item zips.
Use the wrapper in this repo:

```bash
uv run --project envs/robocasa365 \
    data_prep/robocasa365_to_lerobot/download_lightwheel_assets.py --groups objects

# The fixtures group MUST be run with --overwrite:
uv run --project envs/robocasa365 \
    data_prep/robocasa365_to_lerobot/download_lightwheel_assets.py --groups fixtures --overwrite
```

The `--overwrite` is not optional. The downloader skips any pack whose destination directory
already exists, and several fixture categories — including `cabinets` and `handles`, i.e.
exactly the `CabinetDoorPanel*` and drawer-handle assets OpenDrawer needs — are already present
in the checkout. Without `--overwrite` those packs are silently skipped.

**Gap 2 — the base fixtures pack.** `fixtures.zip` exists in the public `robocasa/robocasa-assets`
repo but has **no entry** in robocasa's `DOWNLOAD_ASSET_REGISTRY` (which only registers the
*lightwheel* fixtures), so the documented path never fetches it:

```bash
uv run --project envs/robocasa365 python - <<'PY'
import zipfile
from pathlib import Path
import robocasa
from huggingface_hub import hf_hub_download

# The zip is rooted at "fixtures/", so extract into models/assets (NOT models/assets/fixtures,
# which would give you models/assets/fixtures/fixtures).
dest = Path(robocasa.__path__[0]) / "models" / "assets"
path = hf_hub_download(repo_id="robocasa/robocasa-assets", repo_type="dataset",
                       filename="fixtures.zip")
with zipfile.ZipFile(path) as z:
    z.extractall(dest)
print("extracted to", dest)
PY
```

> Run downloads on a compute node if your login nodes kill long transfers (CLEPS does), with
> `HF_HUB_OFFLINE` unset. The whole asset step is a few minutes.

### 2b. Demonstrations (gated)

The demonstration dataset is **gated**. Before downloading:

1. Accept the terms at
   <https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-Manipulation-Kitchen-Demos>
   with a Hugging Face account.
2. Authenticate: `uv run --project envs/robocasa365 hf auth login`.

Access is per-dataset — being logged in is not enough until you have accepted *this* repo's
terms. Otherwise the download returns 401 (anonymous) or 404 (authenticated, not granted).

Do **not** use `robocasa.scripts.download_datasets`: on the release branch it hardcodes a stale
repo id (`nvidia/PhysicalAI-Robotics-Kitchen-Sim-Demos`) that no longer exists. Use the wrapper
here, which reuses robocasa's own registry and destination logic and only overrides the repo id,
leaving the pinned submodule unmodified:

```bash
uv run --project envs/robocasa365 \
    data_prep/robocasa365_to_lerobot/download_demos.py \
    --tasks OpenDrawer --split target --source human
```

| Flag | Default | Notes |
| --- | --- | --- |
| `--tasks` | all registered tasks | Names from `ATOMIC_TASK_DATASETS` / `COMPOSITE_TASK_DATASETS` |
| `--split` | `target` | `target` = 500 demos/task, `pretrain` = 100 demos/task |
| `--source` | `human` | `mimicgen` is ~10k demos/task |
| `--repo-id` | current demos repo | Override if the dataset is renamed again |
| `--overwrite` | — | Re-download even if the target already exists |

Data lands in `$ROBOCASA_DATASET_ROOT/v1.0/<split>/atomic/<task>/<date>/lerobot`. **The date
component is per task** and comes from `robocasa/utils/dataset_registry.py` — do not assume the
`20250816` used in the OpenDrawer examples generalises to other tasks.

---

## 3. Building the dataset

Replay re-executes the recorded actions in simulation to capture observations and depth, which
is what makes the fused point clouds possible; conversion then packs them into a LeRobot dataset
plus a point-cloud LMDB. Full details in
[`data_prep/robocasa365_to_lerobot/README.md`](../data_prep/robocasa365_to_lerobot/README.md).

```bash
task=OpenDrawer
input_dir=$ROBOCASA_DATASET_ROOT/v1.0/target/atomic/$task/20250816

# 3a. Replay in simulation (robocasa365 env, GPU node, NOT H100 — see below).
export MUJOCO_GL=egl
uv run --project envs/robocasa365 data_prep/robocasa365_to_lerobot/replay.py \
    --input-dir $input_dir/lerobot --cache-dir $input_dir/replay_cache --resume

# 3b. Convert to LeRobot + points_3views LMDB (pointact env, CPU only).
uv run data_prep/robocasa365_to_lerobot/convert.py \
    --cache-dir $input_dir/replay_cache \
    --output-dir $ROBOCASA_DATASET_ROOT/lerobot_point_lmdb/$task \
    --repo-id $task --image-writer-threads 12

# 3c. Normalization statistics. Both extra flags are mandatory here (see below).
uv run data_prep/prepare_robot_state_action_stats.py \
    --dataset_dirs $ROBOCASA_DATASET_ROOT/lerobot_point_lmdb/$task \
    --output_file  $ROBOCASA_DATASET_ROOT/lerobot_point_lmdb/$task/robot_state_action_stats/rot6d.json \
    --point_cloud_dir points_3views \
    --state_xyz_slice 0 3 --action_xyz_slice 0 3 \
    --state_rotation_slice 3 7 --action_rotation_slice 3 7 \
    --rotation_type quat --target_rotation_type rot6d \
    --replace_zero_std
```

As SLURM jobs: `experiments/13_robocasa365/data_prep_replay.slurm` and
`data_prep_convert.slurm` (which also runs stage 3c). Jean Zay variants live in
`data_prep/robocasa365_to_lerobot/{replay,convert}.slurm`; the replay one supports job-array
chunking, which is safe in parallel because the per-episode caches are disjoint.

Three things that are easy to get wrong:

- **Not H100.** The RoboCasa365 simulator does not run correctly on H100. Replay needs only
  MuJoCo/EGL rendering, so a V100 is plenty.
- **`--replace_zero_std` is required.** OpenDrawer is a fixed-base task, so the state's base-pose
  dims are constant and have zero standard deviation. PointAct's state normalization
  (`(state - mean) / std`) has no zero-std guard, so without this flag those dims produce NaN and
  training diverges immediately.
- **`--point_cloud_dir points_3views` is required** whenever an `--*_xyz_slice` is given, because
  the position statistics must be computed in the point-cloud frame.

Replay does not succeed on every episode — `replay_summary.jsonl` records per-episode outcomes.
On OpenDrawer, 496 of 514 episodes replayed successfully, and only those were converted. Note
that `convert.py` numbers output episodes densely by position, so when replays fail the
**converted episode index is not the source episode index**.

Finally, the data config resolves `root` relative to the training working directory, following
the Libero convention. Create the symlink once, at the repository root:

```bash
ln -s $SCRATCH/datasets/robot_data robot_data
```

---

## 4. Training

Frozen vision tower + LLM + merger; the trainable part is the PTv3 point-action expert. Training
has no simulator dependency, so it runs in the root `pointact` env and can use H100.

Needs two pretrained backbones — see [`INSTALLATION.md`](../INSTALLATION.md):

- VLM: `$SCRATCH/models/Qwen2.5-VL-3B-Instruct` (on Jean Zay, picked up from `$DSDIR` automatically)
- PTv3: `$SCRATCH/models/Pointcept-Concerto/concerto_large.pth`

```bash
mkdir -p logs/robocasa365          # the SLURM scripts write here, relative to the submit dir

# Directly:
bash experiments/13_robocasa365/train_pointact_concerto.sh

# Or as a job (submit from the repository root):
sbatch experiments/13_robocasa365/train.slurm            # CLEPS
sbatch experiments/13_robocasa365/train_jeanzay.slurm    # Jean Zay
```

Effective batch size 128 (4 GPUs x 32), 20 epochs. The paper's 20-50K-step budget was for
10-task suites; a naive 1/10 cut badly undertrains a single task here — 5 epochs evaluates at
8%, 20 epochs at 60%. The jobs auto-resume from the run's `output_dir`, so resubmitting the same
script continues past a walltime limit.

Outputs land under `$SCRATCH/PointAct_exprs/robocasa365/pointact/...`.

---

## 5. Evaluation

A policy server (pointact env, holds the model) and a simulator client (robocasa365 env, runs
MuJoCo/EGL) talk over ZMQ on the **same GPU**. `eval_robocasa365.sh` manages both lifecycles;
the `.slurm` files just carry the per-cluster directives.

```bash
# Full 50-trial success rate:
sbatch --export=ALL,CKPT_STEP=final-48750 experiments/13_robocasa365/eval.slurm

# Smoke test (3 trials + videos):
sbatch --time=01:00:00 \
       --export=ALL,CKPT_STEP=1000,NUM_TRIALS=3,OPTS="--args.save_video --args.verbose" \
       experiments/13_robocasa365/eval.slurm
```

Point at a specific run with `--export=ALL,CKPT_DIR=...,CKPT_STEP=...`. Results (per-trial log,
success rate, optional videos) are written to `<run_dir>/results/checkpoint-<step>/`.

**Use an A100.** Both the Qwen VLM and the PTv3 backbone use FlashAttention, which needs
Ampere or newer, so V100 fails; H100 is avoided because the simulator misbehaves there. A100 is
the intersection.

Two details that materially affect the numbers:

- **The client sends a fused 3-view point cloud** (left + right + wrist), matching the
  `points_3views` training data. Left to itself the server would build the cloud from
  `select_video_keys` alone — a single view — which starves the PTv3 backbone and collapses the
  policy. This is a silent quality failure, not an error.
- **Success is counted from the simulator's `info["success"]`, not `done`.** robosuite also sets
  `done` at the horizon timeout, which would inflate the rate.

### Reference result

The Concerto OpenDrawer checkpoint (20 epochs, left+right VLM views) scores **60% (30/50)** on
this evaluation. Use it as the sanity check that a fresh setup is wired correctly.

---

## Keeping results

`$SCRATCH` is purged after ~30 days of no access on both clusters, so anything worth keeping
must be copied off it:

- **Training curves** — logged to Weights & Biases in offline mode (compute nodes have no
  guaranteed outbound access). Upload from a login node: `wandb sync $SCRATCH/wandb/offline-run-*`.
  Set `WANDB_ENTITY` to your own team, or `WANDB_MODE=disabled` to skip logging entirely.
- **Final checkpoint + config** — `bash experiments/13_robocasa365/archive_run.sh <run_dir>`
  tars them to `$STORE` (Jean Zay) or `$HOME/archives` (CLEPS). One tar per run, to conserve
  inode quota.
- **Intermediate checkpoints** exist for resume and are regenerable — let the purge take them.

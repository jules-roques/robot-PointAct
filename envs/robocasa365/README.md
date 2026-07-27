# RoboCasa365 Environment

Standalone [uv](https://docs.astral.sh/uv/) project providing the RoboCasa365 simulator. It is
kept separate from the root `pointact` project because the two cannot share a dependency
resolution: RoboCasa365 pins `numpy==2.2.5`, `mujoco==3.3.1` and `scipy==1.15.3` and targets
Python 3.11, while the training stack is held at `numpy==1.26.4` by `open3d==0.18.0`.

This environment is used by two things:

- `data_prep/robocasa365_to_lerobot/replay.py` — replays source episodes to build the dataset.
- the evaluation client, which talks to the PointAct policy server over ZMQ.

The root `pointact` package is installed here as an editable path dependency, but only its
lightweight base dependencies come with it — no torch, no transformers, no lerobot from the
training side. That covers `pointact.utils.{server_client,rotation,depth}` and
`pointact.robot_envs.robocasa365_utils`, which is everything the simulator side imports.

## Setup

### 1. Check out RoboCasa

If the repository was cloned without `--recurse-submodules`, populate it:

```bash
git submodule update --init envs/robocasa365/robocasa
```

The pinned revision (`robocasa365_release` branch) is enforced by the submodule; the working
copy must not drift from it. `pointact/robot_envs/robocasa365_utils/environments.py` depends
on the release-branch registry API. Move the pointer deliberately:

```bash
git -C envs/robocasa365/robocasa fetch
git -C envs/robocasa365/robocasa checkout <new-rev>
git add envs/robocasa365/robocasa   # records the new pointer in the superproject
```

### 2. Sync

`robosuite` is a git dependency, so the first sync needs network access. Compute nodes have
none — run this once on a login node, after which the uv cache serves compute nodes offline:

```bash
uv sync --project envs/robocasa365
```

### 3. Configure paths

`setup_macros.py` copies `robocasa/macros.py` to `robocasa/macros_private.py` in the
checkout. Edit `DATASET_BASE_PATH` in that file afterwards:

```bash
uv run --project envs/robocasa365 python -m robocasa.scripts.setup_macros
# then set in envs/robocasa365/robocasa/robocasa/macros_private.py:
#   DATASET_BASE_PATH = "<SCRATCH>/datasets/robot_data/robocasa365"
```

`DATASET_BASE_PATH` has no environment-variable override — `robocasa/macros.py` reads only
the module attribute. Left at `None`, datasets default to a `datasets/` directory beside the
package. The value above is what makes downloaded data land where
`data_prep/robocasa365_to_lerobot/README.md` expects it.

### 4. Download kitchen assets

Required to instantiate any environment (~5GB), independent of demonstration data:

```bash
uv run --project envs/robocasa365 python -m robocasa.scripts.download_kitchen_assets
```

The `objaverse`, `aigen_objs`, textures and generative-texture packs come from the public
`robocasa/robocasa-assets` repo. The two `lightwheel` packs point at NVIDIA repos and 404 on
the release branch (renamed/unavailable upstream); they are skipped and are not needed unless
a task uses lightwheel assets.

## Datasets

Demonstrations are LeRobot datasets pulled from Hugging Face as per-task tar files, extracted,
then deleted. **Do not use `robocasa.scripts.download_datasets`**: on the release branch it
hardcodes a stale repo id (`nvidia/PhysicalAI-Robotics-Kitchen-Sim-Demos`) that no longer
exists. Use the wrapper in this repo, which reuses robocasa's registry and destination-path
logic with the current repo id `nvidia/PhysicalAI-Robotics-Manipulation-Kitchen-Demos`:

```bash
uv run --project envs/robocasa365 \
  data_prep/robocasa365_to_lerobot/download_demos.py --tasks OpenDrawer --split target --source human
```

The dataset is **gated**. Accept its terms at
<https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-Manipulation-Kitchen-Demos> with a
Hugging Face account, create a token, and authenticate (`uv run --project envs/robocasa365 hf
auth login`) first — otherwise access returns 401 (anonymous) or 404 (authenticated but not
granted). Access is per-dataset: being logged in is not enough until you accept this repo's
terms.

| Flag | Default | Notes |
| --- | --- | --- |
| `--tasks` | all tasks | Names from `ATOMIC_TASK_DATASETS` / `COMPOSITE_TASK_DATASETS` |
| `--split` | `target` | `pretrain` (100 demos/task) or `target` (500 demos/task) |
| `--source` | `human` | `mimicgen` is ~10k demos/task |
| `--repo-id` | current demos repo | Override if the dataset is renamed again |
| `--overwrite` | — | Re-download even if the target already exists |

For task `T` and split `S` the registry yields `v1.0/<S>/atomic/<T>/<date>`, and data lands in
`DATASET_BASE_PATH/v1.0/<S>/atomic/<T>/<date>/lerobot` — the `input_dir` used by
`data_prep/robocasa365_to_lerobot/README.md`. The date component is per task and comes from
`robocasa/utils/dataset_registry.py`; do not assume the one in that README generalises.

Each tar is fetched into the Hugging Face cache before extraction, so transient usage is
roughly twice the final size — keep `HF_HOME` on `$SCRATCH` (set in step 4).

## Usage

No activation needed — `--project` selects the environment:

```bash
# Dataset replay
uv run --project envs/robocasa365 data_prep/robocasa365_to_lerobot/replay.py <args>
```

Run the PointAct policy server separately, from the repository root, in the training
environment (`uv sync --extra train --extra cuda`).

## Notes

- `robosuite` is pinned to a commit in `pyproject.toml` and requires the **master** branch,
  not a tagged release. Bump it, and the robocasa checkout, deliberately:
  `pointact/robot_envs/robocasa365_utils/environments.py` depends on `TASK_SET_REGISTRY`,
  `get_task_horizon` and `create_env`, which are not stable across revisions.
- Per `data_prep/robocasa365_to_lerobot/README.md`, the RoboCasa365 simulator does not run
  properly on H100 GPUs.
- Importing robocasa also emits a "No private macro file found" warning from **robosuite**.
  It is unrelated to `DATASET_BASE_PATH` and can be ignored. Do not run robosuite's own
  `setup_macros.py`: robosuite is not an editable checkout, so the file would be written into
  `.venv/` and lost on the next sync.

# Installation

Environments are managed with [uv](https://docs.astral.sh/uv/). The root project provides the `pointact` environment for model training, checkpoint loading, data preprocessing, and the policy server; each simulator has its own uv project under `envs/`, used for environment rollouts and evaluation clients.

## PointAct Environment

```bash
git clone https://github.com/cshizhe/PointAct.git
cd PointAct

# Python 3.10: the flash-attn wheel in the `cuda` extra is cp310-only.
uv python pin 3.10

# Base dependencies only cover the client core (see pyproject.toml).
# Training, the policy server and data preprocessing need both extras.
uv sync --extra train --extra cuda
```

`torchcodec` and `av` link against FFmpeg shared libraries at runtime; they are not provided by the environment:

```bash
module load cuda/12.8.0
module load ffmpeg/6.1.1
```

`spconv-cu120`, `torch-scatter` and `flash-attn` are declared in the `cuda` extra as prebuilt wheels, so no CUDA toolkit or compiler is required at install time.

Run this after syncing:

```bash
uv run python - <<'PY'
import torch
import flash_attn
import spconv.pytorch as spconv
import torch_scatter
import open3d
import pointact

print("torch", torch.__version__)
print("cuda available", torch.cuda.is_available())
print("pointact import ok")
PY
```

## Pretrained Backbones

Download the required checkpoint, place it under your preferred storage path, and update the script argument accordingly.

On offline clusters, download models before setting `TRANSFORMERS_OFFLINE=1`.

```bash
# Point transformer v3: concerto
uv run hf download --repo-type model Pointcept/Concerto \
  --local-dir $SCRATCH/datasets/pretrained/Pointcept-Concerto

# Point transformer v3: utonia
uv run hf download --repo-type model Pointcept/Utonia \
  --local-dir $SCRATCH/datasets/pretrained/Pointcept-Utonia

# Qwen 2.5
uv run hf download --repo-type model Qwen/Qwen2.5-VL-3B-Instruct

# Pi0, Pi05
uv run hf download --repo-type model lerobot/pi0_base
uv run hf download --repo-type model lerobot/pi05_base
```

## Simulator Environments

Do not install simulator packages into the `pointact` environment. Keep one environment per simulator, then use server-client evaluation when needed: the PointAct policy server runs in `pointact`, and the simulator client runs in the simulator environment.

Simulator projects are separate uv projects with their own lockfile. They depend on the root project as a path dependency, which contributes only its base dependencies, and are selected with `--project` instead of activation:

```bash
uv sync --project envs/robocasa365
uv run --project envs/robocasa365 <script> <args>
```

| Simulator / platform | Environment | Installation and experiment notes |
| --- | --- | --- |
| RoboCASA365 | `envs/robocasa365` (uv) | [envs/robocasa365/README.md](envs/robocasa365/README.md) |
| LIBERO | `libero` (conda) | [experiments/2_libero/README.md](experiments/2_libero/README.md) |
| RLBench | `rlbench` (conda) | [experiments/10_rlbench/README.md](experiments/10_rlbench/README.md) |

LIBERO and RLBench have not been migrated to uv; follow the conda instructions in their respective READMEs.

# MolmoMotion environment

Isolated environment for the trajectory-forecasting pass behind the Stage 4 anchor arm
(`data_prep/roi_sampling/gate_molmo_motion.py`).

Separate from `envs/molmo` on purpose. Both are Molmo checkpoints, but MolmoPoint loads its
modelling code out of the model repo with `trust_remote_code=True` against
`transformers==4.57.1` exactly, while MolmoMotion ships a real pip package carrying its own
model code and only requires `transformers<5`. Merging them would pin the looser environment
to the stricter one's exact transformers for nothing.

`pointact` is a path dependency so the geometry is shared rather than copied — the
base↔camera conversions the forecaster round-trips through must be literally the same code
the sampler uses, or a Gaussian centre can be silently displaced.

## Build (login node — this one downloads)

```bash
export UV_PROJECT_ENVIRONMENT=$SCRATCH/venvs/molmo_motion
uv sync --project envs/molmo_motion
```

The env lives on `$SCRATCH`, not `$WORK`: a torch env is tens of thousands of files and
`$WORK` is close to its 500k inode quota (`docs/clusters/jean-zay.md`).

`molmo-motion` declares `torchcodec` and `decord` for its own video loading. This pipeline
decodes with `av` and never calls either, so if they fail to resolve or import against the
pinned torch 2.7.0, install the package with `--no-deps` and keep the explicit pins above —
nothing here needs them.

## Weights

```bash
sbatch experiments/13_robocasa365/download_molmo_motion.slurm
```
Fetches `allenai/MolmoMotion-4B-H3-F30` to `$SCRATCH/models/`. **H3, not H1**: the H1
checkpoint sees a single frame and so cannot infer the gripper's velocity from the images,
which is most of what makes the forecast useful here.

Compute nodes have no internet, so this must complete before any job starts.

## Run

```bash
uv run --project envs/molmo_motion --no-sync python \
    -m data_prep.roi_sampling.gate_molmo_motion --help
```

SLURM jobs reference `$SCRATCH/venvs/molmo_motion/bin/python` directly instead, since
`uv run` looks for a `.venv` inside the project directory.

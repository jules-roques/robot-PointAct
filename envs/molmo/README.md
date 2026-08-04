# MolmoPoint environment

Isolated environment for the pointing pass that builds the ROI anchor cache
(`data_prep/roi_sampling/build_molmo_cache.py`).

Separate from the training env for one reason: MolmoPoint ships its modelling code inside
the model repo and loads it with `trust_remote_code=True`, written against
`transformers==4.57.1`. The training stack is on transformers 5.x, and pinning it back
would break the trainer.

`pointact` is a path dependency so the geometry and cache-format modules are shared rather
than copied — the record layout must not be able to drift between writer and reader.

## Build (login node — this one downloads)

```bash
uv sync --project envs/molmo
```

## Weights

```bash
hf download allenai/MolmoPoint-8B --local-dir $SCRATCH/models/MolmoPoint-8B
```
~36 GB on the hub (F32), ~18 GB resident in bf16. Compute nodes have no internet, so this
must be complete before any job starts; `build_molmo_cache.py` checks and fails loudly
rather than reaching for the network.

## Run

```bash
uv run --project envs/molmo --no-sync python -m data_prep.roi_sampling.build_molmo_cache --help
```

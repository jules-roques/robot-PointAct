# CLEPS (Inria)

Primary development cluster as of 2026-07-27. Chosen over Jean Zay for day-to-day work
because dev jobs there can have internet access; Jean Zay is kept for large runs.

> **This page is a stub.** Nothing below the checklist has been verified on CLEPS yet.
> Do not copy SLURM flags from `jean-zay.md` — the account and partition names there are
> IDRIS-specific and will not work here. Fill each item in as you confirm it, and delete
> this banner once the page is real.

## To determine

- [ ] Account / partition / QoS flags for each GPU tier available.
- [ ] Which GPU generations are offered. This decides a lot: the model needs **Ampere or
      newer** for FlashAttention, and **RoboCasa365 simulation misbehaves on H100** — so
      evaluation wants A100-class hardware.
- [ ] Do compute nodes have internet? If yes, the model-weight pre-baking dance required
      on Jean Zay is unnecessary here.
- [ ] Scratch storage path, quota, and purge policy — and therefore whether
      `archive_run.sh` needs a CLEPS-specific destination.
- [ ] Where the RoboCasa365 datasets and LMDB point caches will live.
- [ ] Module system: are `module load` names the same? Is there a CUDA module to load at
      all, or does the uv-managed torch bring its own?
- [ ] Max walltime per partition, and whether requeue/checkpointing is needed for a full
      training run.

## Environment setup

The three uv environments in `docs/envs.md` must be rebuilt here — they are not portable
from Jean Zay. The ROI env's `ultralytics` torch build drops Volta support, which only
matters if CLEPS has V100-class nodes.

## Migrating from Jean Zay

Code arrives via `git clone`. Datasets, LMDB caches and model weights do not — they live
on Jean Zay scratch and must be re-derived or copied deliberately.

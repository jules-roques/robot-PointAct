# Working in this repo

PointAct — a 3D-aware vision-language-action policy. This fork adds a RoboCasa365
integration and experiments on top of the upstream implementation.

## Orientation

| Path | What it is |
|---|---|
| `pointact/` | The policy: VLM backbone + point-action expert. |
| `data_prep/` | Offline dataset construction (LMDB point caches). |
| `experiments/13_robocasa365/` | RoboCasa365 training/eval entry points and SLURM jobs. |
| `envs/robocasa365/` | Isolated simulator environment (see `docs/envs.md`). |
| `docs/clusters/` | Per-cluster infra notes — **read before submitting jobs**. |

## Before you run anything

This repo is developed across more than one HPC cluster, and the clusters differ in
ways that silently break jobs (GPU generation, account/partition names, whether compute
nodes have internet). Nothing cluster-specific belongs in shared code — it goes in
`docs/clusters/<cluster>.md` and in per-machine config.

Read `docs/clusters/jean-zay.md` or `docs/clusters/cleps.md` for the machine you are on
before writing or submitting a SLURM script. If you are on a cluster with no page yet,
write one as you learn it rather than leaving the knowledge in a session transcript.

Read `docs/envs.md` before running any Python. There are several interpreters here and
picking the wrong one produces confusing failures rather than clean errors.

## Conventions

- **Never hardcode a cluster path, account, or partition in `pointact/` or `data_prep/`.**
  Those take paths as arguments. Cluster specifics live in `experiments/*/*.slurm` and
  in `docs/clusters/`.
- **Large artifacts live on scratch storage, never in the repo.** Datasets, LMDB caches
  and checkpoints are referenced by path. Scratch is periodically purged on some
  clusters — see the cluster page for the archiving story.
- **Per-machine settings go in `.claude/settings.local.json`** (gitignored), not in
  `.claude/settings.json` (shared and committed).
- **Point clouds are in the robot-base frame** unless a function name says otherwise.
  Camera calibration is not stored in the RoboCasa datasets; it is recovered from a
  simulator reset.

## Experiments

Each `experiments/NN_name/` directory is self-contained: a README describing the
experiment, data configs, shell entry points, and SLURM wrappers. When adding an
experiment, follow that layout instead of adding top-level scripts.

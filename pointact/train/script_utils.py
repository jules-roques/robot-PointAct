import hashlib
import os
import sys
from pathlib import Path

import yaml
from accelerate.utils import broadcast_object_list
from transformers import HfArgumentParser

from pointact.train.pipeline_config import TrainPipelineConfig
from pointact.train.run_config import group_from_meta, resolve_run_config, tags_from_meta


def parse_training_args(logger=None) -> TrainPipelineConfig:
    parser = HfArgumentParser(TrainPipelineConfig)
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        (training_args,) = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    elif len(sys.argv) >= 2 and sys.argv[1].endswith(".yaml"):
        training_args = _parse_run_yaml(parser, os.path.abspath(sys.argv[1]), sys.argv[2:], logger)
    else:
        (training_args,) = parser.parse_args_into_dataclasses()

    training_args.output_dir = broadcast_object_list([training_args.output_dir])[0]
    if logger is not None:
        logger.info(f"set output-dir to {training_args.output_dir}")

    derive_gradient_accumulation(training_args, logger)
    configure_wandb_identity(training_args)
    return training_args


def derive_gradient_accumulation(training_args: TrainPipelineConfig, logger=None) -> None:
    """Turn `effective_batch` into gradient_accumulation_steps for the GPUs actually granted.

    Ablation arms are only comparable at equal effective batch: 2 GPUs x 32 is not equivalent
    to 4 x 32, it halves the batch and doubles the optimiser steps. Deriving the accumulation
    here rather than in the launcher means the invariant holds however the job was submitted,
    including a dev-QoS fallback from 4 GPUs to 2, and needs no command-line arguments -- the
    run yaml stays the single source of truth.
    """
    if training_args.effective_batch is None:
        return

    world_size = max(1, training_args.world_size)
    per_step = world_size * training_args.per_device_train_batch_size
    accum, remainder = divmod(training_args.effective_batch, per_step)
    if remainder or accum < 1:
        raise ValueError(
            f"effective_batch={training_args.effective_batch} is not reachable with "
            f"{world_size} process(es) x per_device_train_batch_size="
            f"{training_args.per_device_train_batch_size} (= {per_step} per step). "
            "Adjust per_device_train_batch_size in the run yaml, or request a GPU count that "
            "divides the effective batch."
        )

    training_args.gradient_accumulation_steps = accum
    if logger is not None:
        logger.info(
            f"effective batch {training_args.effective_batch} = {world_size} proc x "
            f"{training_args.per_device_train_batch_size} x accum {accum}"
        )


def _parse_run_yaml(
    parser: HfArgumentParser,
    path: str,
    overrides: list[str] | None = None,
    logger=None,
) -> TrainPipelineConfig:
    """Build TrainPipelineConfig from a run yaml, with optional CLI overrides on top.

    The yaml is folded in as argparse defaults, so anything given on the command line wins.
    That is what lets the launcher derive gradient accumulation from the GPUs SLURM actually
    granted while everything scientific stays in the (version-controlled) yaml.
    """
    meta, data, train = resolve_run_config(path)
    parser.set_defaults(**train)
    (training_args,) = parser.parse_args_into_dataclasses(args=list(overrides or []))
    training_args.run_config_path = path

    # The data block is written beside the checkpoints rather than merged into the training
    # args: the data layer already knows how to read a DataConfig yaml, and archiving the
    # resolved copy is what lets an eval job recover the exact sampling mode months later.
    if data:
        output_dir = Path(training_args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        data_path = output_dir / "data_config.yaml"
        with data_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(data, handle, sort_keys=False)
        training_args.data_path = str(data_path)

    training_args.run_meta = meta
    training_args.run_data_config = data
    if logger is not None:
        logger.info(f"loaded run config {path} -> run_name={training_args.run_name}")
    return training_args


def save_resolved_run_config(training_args: TrainPipelineConfig) -> Path | None:
    """Write the fully-resolved run config -- meta, every train argument, and data -- as one yaml.

    `training_args.json` already records the train arguments, but as HF's flattened dump rather
    than something you can feed back to `scripts/train.py`. This writes the round-trippable
    form: re-running it reproduces the run. It is also the artifact uploaded to W&B, so a run's
    page carries the config that produced it rather than only the flattened columns.
    """
    meta = getattr(training_args, "run_meta", None)
    data = getattr(training_args, "run_data_config", None)
    if meta is None and data is None:
        return None

    # dataclasses.asdict chokes on TrainingArguments' non-serialisable members (accelerator
    # state, devices); to_dict() is HF's own JSON-safe projection.
    train_block = {
        key: value
        for key, value in training_args.to_dict().items()
        # Derived at startup from the granted GPU count -- recording it would make the file
        # look reusable on a different topology when it is not. effective_batch is the input.
        if key != "gradient_accumulation_steps"
    }

    document = {"meta": meta or {}, "train": train_block, "data": data or {}}
    path = Path(training_args.output_dir) / "run_config.resolved.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(document, handle, sort_keys=False, default_flow_style=False)
    return path


def configure_wandb_identity(training_args: TrainPipelineConfig) -> None:
    """Pin the W&B run id and set group/job_type/tags via the env vars wandb itself reads.

    Two problems this solves:

    1. Every resume creates a fresh run. HF's WandbCallback calls `wandb.init()` with no id,
       so a training that is preempted, requeued or chained across short slices shows up as
       several identically-named runs. Deriving the id from the output dir and persisting it
       means all of them land in one run.
    2. Runs were identified only by a long concatenated name. `group`/`job_type`/`tags` are
       read straight from the environment by wandb, so no code beyond this is needed; the
       ablation coordinates themselves ride along as the `exp_*` training args.
    """
    if "wandb" not in (training_args.report_to or []):
        return

    output_dir = Path(training_args.output_dir)
    id_file = output_dir / "wandb_run_id.txt"
    run_id = os.environ.get("WANDB_RUN_ID")
    if not run_id and id_file.exists():
        run_id = id_file.read_text(encoding="utf-8").strip()
    if not run_id:
        # Deterministic in the output dir, so a resume that races before the file is written
        # still recovers the same id instead of forking a second run.
        run_id = hashlib.sha1(str(output_dir.resolve()).encode()).hexdigest()[:16]

    os.environ["WANDB_RUN_ID"] = run_id
    os.environ["WANDB_RESUME"] = "allow"
    os.environ.setdefault("WANDB_JOB_TYPE", "train")

    meta = getattr(training_args, "run_meta", None) or {}
    if meta:
        os.environ.setdefault("WANDB_RUN_GROUP", group_from_meta(meta))
        os.environ.setdefault("WANDB_TAGS", ",".join(tags_from_meta(meta)))

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        id_file.write_text(run_id + "\n", encoding="utf-8")
    except OSError:
        # Losing the file only costs the deterministic fallback above; never fail the run.
        pass


def log_trainable_parameters(model, logger) -> None:
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainable_pct = trainable_params / total_params * 100 if total_params else 0
    logger.warning(
        f"{total_params=}, {trainable_params=}, [{trainable_pct}%]",
        main_process_only=True,
    )


def has_resume_checkpoint(output_dir: str | Path) -> bool:
    return bool(list(Path(output_dir).glob("checkpoint-*")))


def train_or_resume(
    trainer,
    training_args: TrainPipelineConfig,
    logger,
    resume_from_checkpoint: bool | None = None,
) -> None:
    output_dir = Path(training_args.output_dir)

    # save training args to output dir for future reference
    if trainer.accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "training_args.json", "w") as outf:
            outf.write(training_args.to_json_string())
        save_resolved_run_config(training_args)

    if resume_from_checkpoint is None:
        resume_from_checkpoint = has_resume_checkpoint(output_dir)

    if resume_from_checkpoint:
        logger.info("resume from checkpoint")
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

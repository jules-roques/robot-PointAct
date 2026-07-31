"""One yaml per run: ablation coordinates, data config and training args in a single file.

Motivation: the training entry points had grown to ~40 CLI flags duplicated across a script
per ablation arm, with the run name assembled by string-concatenating a subset of them. That
made runs hard to reproduce (the yaml and the flags lived apart) and produced W&B names like
`data-robocasa365-opendrawer-point-rot6d-image.leftview_ck16_lr5e-5_gpu4_bs32_epoch50`.

A run yaml has three blocks:

    extends: _base.yaml       # optional, chainable; relative to the including file
    meta:   {...}             # ablation coordinates -> run name, W&B grouping, exp_* fields
    data:   {...}             # a DataConfig document, inlined
    train:  {...}             # TrainPipelineConfig fields, as in HfArgumentParser

`meta` is the single source of truth for identity: the run name, output dir and W&B
group/tags are all derived from it, so they cannot drift apart from the config that produced
them. The resolved `data` block is written next to the checkpoints, which also gives the eval
job a reliable place to read the sampling mode and text-context path from.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml

#: Fields of `meta` that become `exp_*` columns in the W&B run config. Grouping the runs
#: table by these reproduces the ablation grid, which is what lets run names stay short.
META_TO_EXP_FIELD = {
    "task": "exp_task",
    "sampling": "exp_sampling",
    "npoints": "exp_npoints",
    "context": "exp_context",
    "seed": "exp_seed",
    "stage": "exp_stage",
}

#: Short tokens used to build run names, e.g. OpenDrawer/eef/4096 -> od-eef-n4096-s0.
TASK_ABBREV = {
    "OpenDrawer": "od",
    "PickPlaceCounterToStove": "ppcs",
    "TurnOnMicrowave": "tom",
}


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursive dict merge; scalars and lists in `override` win outright."""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _expand_env(value: Any) -> Any:
    """Expand $VAR / ${VAR} in every string, recursively.

    Cluster paths must not be baked into shared config (see CLAUDE.md), so run yamls refer to
    `$SCRATCH`, `$DSDIR` and friends and the environment supplies the machine-specific part.
    """
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, dict):
        return {key: _expand_env(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_env(item) for item in value]
    return value


def load_run_yaml(path: str | Path) -> dict[str, Any]:
    """Read a run yaml and resolve its `extends:` chain."""
    path = Path(path).resolve()
    with path.open(encoding="utf-8") as handle:
        document = yaml.safe_load(handle) or {}

    parent_ref = document.pop("extends", None)
    if parent_ref is None:
        return _expand_env(document)

    parent = load_run_yaml((path.parent / parent_ref).resolve())
    return _deep_merge(parent, _expand_env(document))


def run_name_from_meta(meta: dict[str, Any]) -> str:
    """Short, unique, human-readable handle, e.g. `od-eef-n4096-s0`.

    Everything needed to *identify* a run lives in the W&B config columns; the name only has
    to be short enough to read in a legend and unique enough to index an output directory.
    """
    task = meta.get("task", "task")
    parts = [TASK_ABBREV.get(task, task.lower())]
    if meta.get("sampling"):
        parts.append(str(meta["sampling"]))
    if meta.get("npoints"):
        parts.append(f"n{meta['npoints']}")
    if meta.get("context") and meta["context"] != "text_cache":
        # text_cache is the default for this grid, so only the exception is worth the chars.
        parts.append(str(meta["context"]))
    parts.append(f"s{meta.get('seed', 0)}")
    return "-".join(parts)


def group_from_meta(meta: dict[str, Any]) -> str:
    """W&B group: the arm identity, shared by its seeds and by its eval runs."""
    return "/".join(
        str(meta.get(key)) for key in ("task", "sampling", "npoints") if meta.get(key) is not None
    )


def tags_from_meta(meta: dict[str, Any]) -> list[str]:
    tags = [str(meta[key]) for key in ("task", "sampling", "context", "stage") if meta.get(key)]
    if meta.get("npoints"):
        tags.append(f"n{meta['npoints']}")
    return tags


def resolve_run_config(path: str | Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Return (meta, data, train) with `meta` folded into the train args.

    The caller is responsible for writing `data` somewhere on disk and pointing
    `train["data_path"]` at it -- kept out of here so this stays free of side effects.
    """
    document = load_run_yaml(path)

    meta = document.get("meta") or {}
    data = document.get("data") or {}
    train = dict(document.get("train") or {})

    unknown = set(document) - {"meta", "data", "train"}
    if unknown:
        raise ValueError(
            f"unknown top-level key(s) in {path}: {sorted(unknown)}. A run yaml holds only "
            "extends / meta / data / train."
        )

    for meta_key, field_name in META_TO_EXP_FIELD.items():
        if meta.get(meta_key) is not None:
            train.setdefault(field_name, meta[meta_key])

    # `meta` drives identity, but an explicit train.run_name still wins if someone sets one.
    if meta and "run_name" not in train:
        train["run_name"] = run_name_from_meta(meta)

    # Deterministic output dir. TrainPipelineConfig otherwise defaults it to
    # {output_base}/{timestamp}-{run_name}, and a timestamp means a requeued job cannot find
    # its own checkpoints -- it would silently restart from scratch instead of resuming.
    if "output_dir" not in train and train.get("run_name"):
        output_base = train.get("output_base", "outputs")
        train["output_dir"] = f"{output_base}/{train['run_name']}"

    # Keep the model's context source and the data block's cache in step: forgetting one of
    # the two is the easiest way to silently train the wrong architecture.
    if meta.get("context") and "context_source" not in train:
        train["context_source"] = meta["context"]

    return meta, data, train

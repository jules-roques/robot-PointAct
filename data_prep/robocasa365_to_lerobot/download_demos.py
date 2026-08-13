"""Download RoboCasa365 LeRobot demonstrations from Hugging Face.

The upstream `robocasa.scripts.download_datasets` hardcodes a stale dataset repo id
(`nvidia/PhysicalAI-Robotics-Kitchen-Sim-Demos`) that no longer resolves — the data was
renamed to `nvidia/PhysicalAI-Robotics-Manipulation-Kitchen-Demos`. This wrapper reuses
robocasa's own registry and destination-path logic and only overrides the repo id, so the
pinned robocasa submodule stays unmodified.

The dataset is gated: accept the terms at
https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-Manipulation-Kitchen-Demos
and authenticate (`hf auth login`) first.

Run in the robocasa365 environment, e.g.:

    uv run --project envs/robocasa365 \
        data_prep/robocasa365_to_lerobot/download_demos.py --tasks OpenDrawer
"""

from __future__ import annotations

import argparse
import os
import tarfile
from pathlib import Path

import robocasa
import robocasa.macros as macros
from huggingface_hub import hf_hub_download
from robocasa.utils.dataset_registry import ATOMIC_TASK_DATASETS, COMPOSITE_TASK_DATASETS
from robocasa.utils.dataset_registry_utils import get_ds_meta

DEFAULT_REPO_ID = "nvidia/PhysicalAI-Robotics-Manipulation-Kitchen-Demos"


def _base_datasets_path() -> Path:
    if macros.DATASET_BASE_PATH is not None:
        return Path(macros.DATASET_BASE_PATH) / "v1.0"
    return Path(robocasa.__path__[0]).parent / "datasets" / "v1.0"


def download_task(task: str, split: str, source: str, repo_id: str, overwrite: bool) -> None:
    src = "mg" if source == "mimicgen" else source
    ds_meta = get_ds_meta(task=task, source=src, split=split)
    ds_path = ds_meta["path"] if ds_meta is not None else None
    if ds_path is None:
        print(f"[skip] no {source}/{split} dataset registered for {task}")
        return

    ds_path = Path(ds_path)
    if ds_path.exists() and not overwrite:
        print(f"[skip] already present: {ds_path}")
        return

    rel = ds_path.relative_to(_base_datasets_path())
    tar_filename = str(rel.parent / f"{rel.name}.tar")

    print(f"[get ] {repo_id}::{tar_filename}")
    tar_path = hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=tar_filename)

    extract_dir = ds_path.parent
    os.makedirs(extract_dir, exist_ok=True)
    print(f"[tar ] extracting to {extract_dir}")
    with tarfile.open(tar_path, "r") as tar:
        tar.extractall(path=extract_dir)
    os.remove(tar_path)  # keep the HF cache from doubling on disk
    print(f"[done] {ds_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", default=None, help="Defaults to all registered tasks")
    parser.add_argument("--split", nargs="+", default=["target"], choices=["pretrain", "target"])
    parser.add_argument("--source", nargs="+", default=["human"], choices=["human", "mimicgen"])
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    tasks = args.tasks
    if tasks is None:
        tasks = list(ATOMIC_TASK_DATASETS.keys()) + list(COMPOSITE_TASK_DATASETS.keys())

    for task in tasks:
        for split in args.split:
            for source in args.source:
                download_task(task, split, source, args.repo_id, args.overwrite)


if __name__ == "__main__":
    main()

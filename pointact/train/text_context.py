"""Make sure a run's cached text context exists before the dataloader needs it.

`context_source=text_cache` replaces the live VLM forward with per-instruction hidden states
read from a small `.pt`. Building it is a one-off, but it used to be a manual prerequisite:
forget it and the run dies at step 0, after queueing. This closes that gap -- if the cache is
missing or does not cover every instruction in the dataset, it is built at startup.

Deliberately at startup on one process, not lazily in the dataset: the dataset is constructed
in each of N dataloader workers on each of M ranks, and having any of them load a 3B model
would be catastrophic. Here it happens once, before the workers fork, and everyone else waits.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch

from data_prep.cache_text_context import build_cache, read_instructions


def _dataset_entries(training_args) -> list[tuple[Path, str]]:
    """(dataset_dir, cache_path) for every lerobot dataset that wants a text-context cache."""
    from pointact.data.dataset import SupervisedDataset

    entries = []
    for config in SupervisedDataset.load_data_config(training_args).lerobot_datasets:
        if not config.text_context_file:
            continue
        dataset_dir = Path(config.root or "") / config.repo_id
        entries.append((dataset_dir, dataset_dir / config.text_context_file))
    return entries


def _missing_instructions(dataset_dir: Path, cache_path: Path) -> list[str]:
    """Instructions the dataset uses that the cache does not cover (all of them if absent)."""
    instructions = read_instructions(dataset_dir)
    if not cache_path.exists():
        return instructions
    try:
        cached = torch.load(cache_path, map_location="cpu", weights_only=True)
    except Exception:  # noqa: BLE001 - a corrupt or truncated cache should just be rebuilt
        return instructions
    return [text for text in instructions if text not in cached]


def ensure_text_context(training_args, logger=None) -> None:
    """Build any missing text-context cache, once, before training starts.

    No-op unless `context_source` is a cached one, `text_context_autobuild` is set, and
    something is actually missing -- so a run with a pre-built cache pays nothing.
    """
    if training_args.context_source == "vlm" or not training_args.text_context_autobuild:
        return

    def _log(message: str) -> None:
        if logger is not None:
            logger.info(message, main_process_only=True)
        else:
            print(message)

    is_main = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", 0))) == 0
    if is_main:
        for dataset_dir, cache_path in _dataset_entries(training_args):
            missing = _missing_instructions(dataset_dir, cache_path)
            if not missing:
                continue
            _log(
                f"text context: {len(missing)} instruction(s) missing from {cache_path} "
                f"-- building (loads the VLM once; ~1 min)"
            )
            # Rebuild the whole cache rather than patching: it is tiny, and one file written
            # by one code path is easier to reason about than an incrementally grown one.
            cache = build_cache(
                read_instructions(dataset_dir),
                training_args.vlm_name_or_path,
                "cuda" if torch.cuda.is_available() else "cpu",
                torch.bfloat16,
            )
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            # Atomic rename: several runs of the same grid can start at once and find the same
            # cache missing. Each writes its own temp file and renames, so a reader never sees
            # a partially written .pt -- last writer wins, and every version is equivalent.
            tmp_path = cache_path.with_suffix(f".tmp{os.getpid()}")
            torch.save(cache, tmp_path)
            os.replace(tmp_path, cache_path)
            _log(f"text context: wrote {cache_path}")

    # Ranks that skipped the build must not race ahead and open a half-written file.
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()

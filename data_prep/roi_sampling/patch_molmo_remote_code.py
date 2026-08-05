"""Fix an upstream bug in MolmoPoint's remote modelling code. Idempotent.

``modeling_molmo_point.py`` combines two boolean tensors with Python's ``and``:

    should_embed = (input_patch_ids >= 0) and (input_patch_ids < (bounds.patch_end-1))

``and`` calls ``__bool__`` on the left operand, which raises
``RuntimeError: Boolean value of Tensor with more than one value is ambiguous`` for
anything but a single element. So the model only runs one pointing request at a time; any
batching crashes in the first forward.

At one element it does not crash, but it is still wrong: ``and`` returns the *second*
operand when the first is truthy, silently dropping the ``>= 0`` guard. Elementwise ``&``
is correct in both cases and identical to the current behaviour for a single element, so
this is a strict fix rather than a behaviour change.

Run after downloading the weights, before any cache build. Also clears the copy
``transformers`` keeps under ``$HF_HOME/modules``, which is generated from the model
directory and would otherwise keep serving the unpatched version.

    python -m data_prep.roi_sampling.patch_molmo_remote_code --model-dir $SCRATCH/models/MolmoPoint-8B
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

TARGET = "modeling_molmo_point.py"
BAD = "should_embed = (input_patch_ids >= 0) and (input_patch_ids < (bounds.patch_end-1))"
GOOD = "should_embed = (input_patch_ids >= 0) & (input_patch_ids < (bounds.patch_end-1))"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True, type=Path)
    ap.add_argument("--hf-modules", type=Path, default=None,
                    help="Defaults to $HF_HOME/modules/transformers_modules")
    args = ap.parse_args()

    path = args.model_dir.expanduser().resolve() / TARGET
    if not path.exists():
        raise SystemExit(f"no {TARGET} in {args.model_dir} -- is the download complete?")

    src = path.read_text()
    if GOOD in src:
        # Nothing to invalidate, so leave the module cache alone. Clearing it here
        # unconditionally is a race: transformers regenerates that directory on first load,
        # and with several cache-build jobs starting together one job would rmtree it while
        # another was midway through importing from it, giving
        # "ModuleNotFoundError: No module named
        # transformers_modules.MolmoPoint_hyphen_8B.image_processing_molmo2".
        print(f"already patched: {path} (module cache left alone)")
        return
    if BAD in src:
        path.write_text(src.replace(BAD, GOOD))
        print(f"patched: {path}")
    else:
        raise SystemExit(
            f"neither the buggy nor the patched line found in {path}.\n"
            f"The checkpoint's remote code has changed; re-check whether the bug is still "
            f"present before assuming batching works.")

    # transformers copies remote code into its own module cache on first load and reuses it
    # regardless of the source file's mtime, so the patch is invisible until this is gone.
    modules = args.hf_modules or Path(
        os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")
    ) / "modules" / "transformers_modules"
    removed = 0
    if modules.exists():
        for d in modules.iterdir():
            if "Molmo" in d.name:
                shutil.rmtree(d, ignore_errors=True)
                removed += 1
    print(f"cleared {removed} cached module dir(s) under {modules}")


if __name__ == "__main__":
    main()

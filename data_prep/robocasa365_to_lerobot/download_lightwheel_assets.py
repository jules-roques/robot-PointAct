"""Download RoboCasa365 'lightwheel' object and fixture assets from Hugging Face.

The upstream `robocasa.scripts.download_kitchen_assets` requests monolithic
`objects_lightwheel.zip` / `fixtures_lightwheel.zip` files that no longer exist: the public
repo `nvidia/PhysicalAI-Kitchen-Assets` now stores per-item zips under `objects_lightwheel/`
and `fixtures_lightwheel/`. Without these, kitchen scenes fail to load (e.g. OpenDrawer needs
`objects/lightwheel/utensil_rack/...`). This wrapper fetches every per-item zip and extracts
it into the layout robocasa expects:

    objects_lightwheel/<name>.zip   -> <assets>/objects/lightwheel/<name>/...
    fixtures_lightwheel/<name>.zip  -> <assets>/fixtures/<name>/...

where <assets> is `robocasa/models/assets` in the active robocasa checkout. Extraction is
in place; redirect the heavy directories to large storage beforehand (see the env README).

Run in the robocasa365 environment:

    uv run --project envs/robocasa365 \
        data_prep/robocasa365_to_lerobot/download_lightwheel_assets.py
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

import robocasa
from huggingface_hub import HfApi, hf_hub_download

DEFAULT_REPO_ID = "nvidia/PhysicalAI-Kitchen-Assets"


def _assets_root() -> Path:
    return Path(robocasa.__path__[0]) / "models" / "assets"


def download_group(repo_id: str, prefix: str, dest: Path, files: list[str], overwrite: bool) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    group = sorted(f for f in files if f.startswith(prefix + "/") and f.endswith(".zip"))
    print(f"[{prefix}] {len(group)} archives -> {dest}")
    refetched = 0
    for i, fname in enumerate(group, 1):
        item = Path(fname).stem
        marker = dest / item
        # An existing directory is not an extracted asset. These live on $SCRATCH (the
        # assets/ subdirectories are symlinks to large storage), and its access-time purge
        # deletes the .obj/.mtl files while leaving every directory standing -- on
        # 2026-09-21, 643 of 922 lightwheel item dirs were empty shells. MuJoCo then fails
        # at scene load with "Error opening file ...FlowerVase012.obj", one asset at a time,
        # halfway into a replay. Test for a file, not for the directory.
        if not overwrite and marker.is_dir() and any(marker.rglob("*.obj")):
            print(f"  ({i}/{len(group)}) skip {item} (present)")
            continue
        if marker.is_dir():
            refetched += 1
        print(f"  ({i}/{len(group)}) {item}")
        zip_path = hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=fname)
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(path=dest)
    if refetched:
        print(f"[{prefix}] refetched {refetched} item(s) whose directory existed but was "
              f"empty -- the signature of the $SCRATCH purge, not of a partial download.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--groups", nargs="+", default=["objects", "fixtures"],
                        choices=["objects", "fixtures"])
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    api = HfApi()
    files = api.list_repo_files(args.repo_id, repo_type="dataset")
    assets = _assets_root()

    if "objects" in args.groups:
        download_group(args.repo_id, "objects_lightwheel",
                       assets / "objects" / "lightwheel", files, args.overwrite)
    if "fixtures" in args.groups:
        download_group(args.repo_id, "fixtures_lightwheel",
                       assets / "fixtures", files, args.overwrite)


if __name__ == "__main__":
    main()

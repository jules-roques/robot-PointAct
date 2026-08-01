"""Attach a run's point-cloud animation to its existing W&B run, after the fact.

Needed because the viz is logged once at step 0 by SamplingVizCallback, so a run that has
already started cannot pick up a corrected figure. The nine stage-A runs logged a version whose
display subsample capped every point count at 2000 points, making 2048/4096/8192 look
identical; this pushes the regenerated ones.

Works by resuming the run: every run pins its W&B id in output_dir/wandb_run_id.txt, so
`wandb.init(id=..., resume="allow")` reopens the same cloud run rather than creating a
duplicate. Run it from somewhere with outbound network -- a Jean Zay login node, not a compute
node -- and only after the offline run has been synced, or there is nothing to resume into.

    python experiments/13_robocasa365/attach_viz_to_wandb.py \
        $SCRATCH/PointAct_exprs/robocasa365/ablation/od-eef-n4096-s0
"""

import argparse
import os
import sys
from pathlib import Path

import wandb


def attach(run_dir: Path, project: str, entity: str, key: str, delete_stale: bool) -> bool:
    run_id_file = run_dir / "wandb_run_id.txt"
    if not run_id_file.exists():
        print(f"  skip {run_dir.name}: no wandb_run_id.txt")
        return False

    html_files = sorted((run_dir / "viz").glob("train_ep_ep0000_*.html"))
    if not html_files:
        print(f"  skip {run_dir.name}: no viz/train_ep_ep0000_*.html")
        return False

    run_id = run_id_file.read_text().strip()
    html = html_files[0]
    size_mb = html.stat().st_size / 1e6

    if delete_stale:
        # The corrected figure lands at a later step, so the media slider would otherwise show
        # the misleading capped one first. Removing the old file leaves that history entry
        # dangling (it renders as missing), which is preferable to it rendering as plausible.
        try:
            api_run = wandb.Api().run(f"{entity}/{project}/{run_id}")
            for remote in api_run.files():
                if key.split("/")[-1] in remote.name and remote.name.endswith(".html"):
                    remote.delete()
                    print(f"  deleted stale {remote.name}")
        except Exception as exc:  # noqa: BLE001 - deletion is a nicety, never the point
            print(f"  could not delete stale media ({type(exc).__name__}: {exc})")

    run = wandb.init(
        project=project, entity=entity, id=run_id, resume="allow",
        # Nothing here should look like a new training run to the runs table.
        settings=wandb.Settings(_disable_stats=True, _disable_meta=True),
    )
    run.log({key: wandb.Html(html.read_text(encoding="utf-8"), inject=False)})
    run.summary["viz/episode0_points"] = int(run_dir.name.split("-n")[1].split("-")[0])
    run.finish()
    print(f"  {run_dir.name}: attached {html.name} ({size_mb:.1f} MB) to run {run_id}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--project", default=os.environ.get("WANDB_PROJECT", "pointact-robocasa365"))
    parser.add_argument("--entity", default=os.environ.get("WANDB_ENTITY", "diffusion4robots"))
    parser.add_argument("--key", default="data/episode0_sampling")
    parser.add_argument("--delete-stale", action="store_true",
                        help="Remove the previously logged (capped) HTML from the run's files.")
    args = parser.parse_args()

    if os.environ.get("WANDB_MODE") == "offline":
        sys.exit("WANDB_MODE=offline: this needs to reach the API. Run it from a login node.")

    done = sum(attach(d, args.project, args.entity, args.key, args.delete_stale)
               for d in args.run_dirs)
    print(f"\nattached {done}/{len(args.run_dirs)} run(s)")


if __name__ == "__main__":
    main()

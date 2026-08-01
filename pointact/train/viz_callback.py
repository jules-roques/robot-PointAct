"""Log an interactive point-cloud animation of training episode 0 to the run's W&B page.

The point of having it inside the training run: each arm draws its points differently, and the
animation is the only artefact that shows *what the network was actually fed* rather than a
number derived from it. Colour is the per-point sampling weight as a multiple of the uniform
draw, so a uniform arm reads as one neutral tone and concentration reads as warmth.

Implementation note: this shells out to data_prep/roi_sampling/viz_sampling_episode.py rather
than importing it. That script reads the point LMDB directly and is a validated, already-used
tool; wrapping it in a library API would be a bigger change than the feature warrants, and the
cost here is a single subprocess at step 0.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from transformers import TrainerCallback

VIZ_SCRIPT = "data_prep/roi_sampling/viz_sampling_episode.py"


def _sampling_method(dataset_cfg: dict) -> str:
    """Map a data config onto the viz script's --method name."""
    if dataset_cfg.get("oracle_sampling"):
        return "oracle"
    if dataset_cfg.get("eef_sampling"):
        return "eef"
    if dataset_cfg.get("roi_point_cloud_dirname"):
        return "roi"
    return "uniform"


def build_episode_html(
    dataset_dir: Path,
    dataset_cfg: dict,
    out_dir: Path,
    episode: int = 0,
    num_frames: int = 20,
    display_points: int = 0,
) -> Path | None:
    """Render one episode under this run's sampling strategy; return the HTML path.

    `display_points=0` disables the renderer's display subsample, so every point the network
    was actually given is drawn. This matters: capping the display at a fixed count makes a
    2048-point run and an 8192-point run render identically, which is exactly the comparison
    the artefact exists to show. Size is controlled with the frame count instead, which costs
    temporal resolution rather than the thing being measured -- and the files legitimately
    differ in size across the sweep, because they carry different amounts of data.

    `--plotlyjs cdn` rather than inline: plotly.js alone is ~3.5 MB, which would otherwise
    dominate. W&B renders the page in the viewer's browser, which can fetch it.
    """
    method = _sampling_method(dataset_cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        VIZ_SCRIPT,
        "--dataset-dir", str(dataset_dir),
        "--episode", str(episode),
        "--method", method,
        "--out-dir", str(out_dir),
        "--out-prefix", "train_ep",
        "--num-frames", str(num_frames),
        "--display-points", str(display_points),
        "--max-npoints", str(dataset_cfg.get("max_npoints", 4096)),
        "--plotlyjs", "cdn",
        "--dark",
    ]
    if method == "eef":
        command += [
            "--eef-sigma", str(dataset_cfg.get("eef_sampling_sigma", 0.08)),
            "--eef-floor", str(dataset_cfg.get("eef_sampling_floor", 0.05)),
        ]
    elif method == "oracle":
        command += [
            "--labels-dirname", str(dataset_cfg.get("oracle_label_dirname", "points_3views_labels")),
            "--oracle-sigma", str(dataset_cfg.get("oracle_sampling_sigma", 0.08)),
            "--oracle-floor", str(dataset_cfg.get("oracle_sampling_floor", 0.05)),
        ]

    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        # Never fail a training run over a visualisation.
        print(f"[viz] skipped episode-{episode} animation: {result.stderr.strip()[-500:]}")
        return None

    produced = sorted(out_dir.glob(f"train_ep_ep{episode:04d}_{method}.html"))
    return produced[0] if produced else None


def log_training_episode(training_args, dataset_cfg: dict, dataset_dir: Path) -> None:
    """Log the episode-0 animation to the active W&B run. Call once, from rank 0 only."""
    try:
        import wandb
    except ImportError:
        return
    if wandb.run is None:
        return

    html_path = build_episode_html(
        dataset_dir=dataset_dir,
        dataset_cfg=dataset_cfg,
        out_dir=Path(training_args.output_dir) / "viz",
    )
    if html_path is None:
        return

    size_mb = html_path.stat().st_size / 1e6
    wandb.log(
        {"data/episode0_sampling": wandb.Html(html_path.read_text(encoding="utf-8"), inject=False)},
        step=0,
    )
    print(f"[viz] logged {html_path.name} to W&B ({size_mb:.1f} MB)")


class SamplingVizCallback(TrainerCallback):
    """Logs the episode-0 sampling animation once, at the start of training.

    Must be a callback rather than a call before `trainer.train()`: HF's WandbCallback creates
    the run in its own `on_train_begin`, so `wandb.run` does not exist until training starts.
    Registering this afterwards puts it later in the callback order, so the run is live by the
    time this fires.
    """

    def __init__(self) -> None:
        self._done = False

    @staticmethod
    def _upload_run_config(args) -> None:
        """Attach the resolved run yaml to the W&B run.

        HF already logs every training argument as flat config columns, which is what makes
        the runs table groupable -- but not something you can re-run. The yaml is, so the run
        page carries the exact file that produced it.
        """
        try:
            import wandb

            if wandb.run is None:
                return
            resolved = Path(args.output_dir) / "run_config.resolved.yaml"
            if resolved.exists():
                wandb.save(str(resolved), base_path=str(resolved.parent), policy="now")
                print(f"[config] uploaded {resolved.name} to W&B")
        except Exception as exc:  # noqa: BLE001 - never fail training over bookkeeping
            print(f"[config] could not upload run config: {type(exc).__name__}: {exc}")

    def on_train_begin(self, args, state, control, **kwargs):
        # A resumed run has already logged this, and re-logging would land at a step behind
        # the run's current one.
        if self._done or not state.is_world_process_zero or state.global_step > 0:
            return
        self._done = True

        if "wandb" not in (args.report_to or []):
            return

        self._upload_run_config(args)

        try:
            import yaml

            with open(args.data_path, encoding="utf-8") as handle:
                entry = yaml.safe_load(handle)["lerobot_datasets"][0]
            dataset_dir = Path(entry.get("root") or "") / entry["repo_id"]
            log_training_episode(args, entry, dataset_dir)
        except Exception as exc:  # noqa: BLE001 - visualisation must not break training
            print(f"[viz] skipped episode-0 animation: {type(exc).__name__}: {exc}")

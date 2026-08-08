"""Ground-truth 3D position of a task's target (and its distractors), per frame.

Runs in the **robocasa365 env** (MuJoCo/EGL). Replays each episode with its recorded
actions -- no rendering, no point clouds -- and records where the manipulated part actually
is, in the robot-base frame the point clouds use. This is the reference the MolmoPoint
anchor is scored against (``eval_molmo_accuracy.py``).

Why not the segmentation labels the oracle arm uses:

* OpenDrawer has them (``points_3views_labels``, label 4 = handle) and they are the right
  reference *there*, because the oracle arm's Gaussian is centred on that same centroid.
  This script's OpenDrawer output is a cross-check on it, not a replacement.
* TurnOnMicrowave has none and cannot cheaply get them. The labeller finds the task's target
  by probing ``env.drawer/.door/.cab/.obj/...``; on this task the only attribute it matches is
  ``.obj`` -- the food item placed *inside* the microwave, not the button being pressed. And a
  start button is a couple of pixels: a segmentation centroid of it would be dominated by
  quantisation. ``sim.data.geom_xpos`` of the button geom is exact and needs no rendering.

Distractors matter as much as the target. RoboCASA's microwave carries a stop button a few
centimetres from the start button, so "pointed at the wrong button" and "pointed nowhere near
the microwave" are different failures with different fixes; scoring against the target alone
cannot tell them apart. Every geom this script knows about for a task is dumped, and the
evaluator decides which is the target.

Output (npz, ``<task>/roi_meta/target_positions.npz``)::

    episode (N,) int64, frame (N,) int64, <geom-set name> (N, 3) float64 in the base frame,
    plus ``geom_sets`` (json) and ``lengths`` for auditing.

Usage (see target_positions_jeanzay.slurm)::

    python -m data_prep.roi_sampling.dump_target_positions \
        --task TurnOnMicrowave \
        --source-dir $SCRATCH/.../TurnOnMicrowave/20250813/lerobot \
        --out        $SCRATCH/.../lerobot_point_lmdb/TurnOnMicrowave/roi_meta/target_positions.npz
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import time
from pathlib import Path

import numpy as np

from data_prep.robocasa365_to_lerobot.replay import (
    convert_env_action_to_dataset_action,
    load_source_actions,
    reorder_source_action_to_env,
)
from pointact.robot_envs.robocasa365_utils.environments import RoboCasa365Env

#: task -> {output name: how to find the geoms whose mean xpos is that position}.
#:
#: ``geom`` names an exact geom (after the fixture's naming prefix); ``body_contains`` takes
#: every geom hanging off a body of that fixture whose name contains the substring, which is
#: how ``environments._build_geom_label_lut`` assigns the handle/door labels -- so the
#: OpenDrawer entry below reproduces the oracle arm's target from geometry instead of from
#: rendered labels.
TARGET_GEOMS = {
    "TurnOnMicrowave": {
        "fixture_attr": "microwave",
        "sets": {
            "start_button": {"geom": "start_button"},
            "stop_button": {"geom": "stop_button"},
            "microwave": {"body_contains": ""},
        },
    },
    "OpenDrawer": {
        "fixture_attr": "drawer",
        "sets": {
            "handle": {"body_contains": "handle"},
            "drawer_panel": {"body_contains": "drawer"},
        },
    },
}


def resolve_geom_ids(env, fixture, spec: dict) -> list[int]:
    """Geom ids matching one entry of a task's ``sets``.

    ``geom`` is looked up through the fixture's ``naming_prefix`` (that is how RoboCASA's own
    success check finds the button); ``body_contains`` walks every geom and keeps those whose
    body belongs to the fixture and contains the substring -- empty substring means the whole
    fixture.
    """
    model = env.env.sim.model
    if "geom" in spec:
        name = f"{fixture.naming_prefix}{spec['geom']}"
        try:
            return [int(model.geom_name2id(name))]
        except Exception:
            return []

    want = spec["body_contains"]
    fixture_name = fixture.name
    out = []
    for gid in range(int(model.ngeom)):
        body = model.body_id2name(int(model.geom_bodyid[gid])) or ""
        if fixture_name in body and want in body:
            out.append(gid)
    return out


def world_to_base(points_world: np.ndarray, base_pos, base_quat_xyzw) -> np.ndarray:
    """(N, 3) world points expressed in the robot-base frame the point clouds use."""
    from scipy.spatial.transform import Rotation as R

    rot = R.from_quat(np.asarray(base_quat_xyzw, dtype=np.float64).reshape(4)).as_matrix()
    p = np.asarray(points_world, dtype=np.float64).reshape(-1, 3)
    return (p - np.asarray(base_pos, dtype=np.float64).reshape(1, 3)) @ rot


def infer_env_name(source_dir: Path) -> str:
    meta = json.loads((source_dir / "extras" / "dataset_meta.json").read_text())
    env_name = (
        meta.get("env")
        or meta.get("env_info", {}).get("env_name")
        or meta.get("env_args", {}).get("env_name")
    )
    assert env_name, f"Could not infer env_name from {source_dir}"
    return str(env_name)


def dump_episode(env, source_dir: Path, episode_index: int, sets: dict, fixture_attr: str,
                 max_frames: int | None, reset_only: bool = False) -> dict | None:
    """Per-frame base-frame position of every geom set, for one episode.

    Steps the recorded actions by default, because the OpenDrawer handle travels with the
    drawer and a reset-only position would be wrong for most of the episode. Frame indexing
    then matches ``replay.py`` exactly -- one row per action stepped, plus the terminal frame
    when the episode ends early -- so rows line up with the point-cloud LMDB keys.

    ``reset_only`` emits a single row instead, for targets that do not move: a microwave's
    start button travels 0.1 cm over a whole episode, and stepping 133 actions to learn that
    costs about 30x what the reset alone does. The evaluator broadcasts a one-row episode to
    every frame. Verify staticness with a full-replay probe before using it on a new task --
    the printed per-episode travel is exactly that check.
    """
    episode_dir = source_dir / "extras" / f"episode_{episode_index:06d}"
    if not episode_dir.exists():
        return None
    obs, _ = env.reset(initial_state_dir=episode_dir, step_after_reset=False)

    fixture = getattr(env.env, fixture_attr, None)
    if fixture is None:
        raise RuntimeError(
            f"env has no attribute '{fixture_attr}'; TARGET_GEOMS is out of step with the task")
    gids = {name: resolve_geom_ids(env, fixture, spec) for name, spec in sets.items()}
    missing = [n for n, ids in gids.items() if not ids]
    if missing:
        raise RuntimeError(f"no geoms matched {missing} on {fixture.name} "
                           f"(prefix '{fixture.naming_prefix}')")

    rows = {name: [] for name in sets}

    def record():
        # The robot base is fixed for the episode, but read it per frame anyway: it is one
        # array lookup and it makes this correct if a task ever moves the base.
        base_pos = np.asarray(obs["state.base_position"], dtype=np.float64).reshape(3)
        base_quat = np.asarray(obs["state.base_rotation"], dtype=np.float64).reshape(4)
        for name, ids in gids.items():
            xw = env.env.sim.data.geom_xpos[ids].reshape(-1, 3)
            rows[name].append(world_to_base(xw, base_pos, base_quat).mean(axis=0))

    if reset_only:
        record()
        return {name: np.stack(v, axis=0) for name, v in rows.items()}

    actions = load_source_actions(source_dir, episode_index)["actions_lerobot"]
    action_envs = np.stack([reorder_source_action_to_env(a) for a in actions], axis=0)
    if max_frames is not None:
        action_envs = action_envs[:max_frames]

    for action_env in action_envs:
        record()
        obs, _reward, done, info = env.step(convert_env_action_to_dataset_action(action_env))
        if bool(done or info.get("success", False)):
            record()  # replay.py keeps the terminal observation as one extra frame
            break

    return {name: np.stack(v, axis=0) for name, v in rows.items()}


def summarize(payload: dict, names: list[str]) -> None:
    """Two read-outs the evaluator's claims depend on.

    Separation between geom sets is the scale a "pointed at the wrong button" claim has to be
    measured against: if start and stop sit 4 cm apart, an 8 cm sampling sigma cannot
    distinguish them and the distinction is not worth drawing.

    Travel *within* an episode says whether a per-frame reference buys anything over a
    per-episode one -- and on a task whose target does move, it is the check that this replay
    is following the demonstration rather than sitting at the reset pose.
    """
    for a in range(len(names)):
        for b in range(a + 1, len(names)):
            d = np.linalg.norm(payload[names[a]] - payload[names[b]], axis=1)
            print(f"  |{names[a]} - {names[b]}|: median {np.median(d) * 100:.1f} cm "
                  f"(min {d.min() * 100:.1f}, max {d.max() * 100:.1f})")

    ep_all = payload["episode"]
    for name in names:
        moves = []
        for ep in np.unique(ep_all):
            arr = payload[name][ep_all == ep]
            moves.append(np.linalg.norm(arr - arr[0], axis=1).max())
        moves = np.asarray(moves)
        print(f"  {name}: travels {np.median(moves) * 100:.1f} cm within an episode "
              f"(median over episodes; max {moves.max() * 100:.1f})")


def merge_shards(shards: list[Path], out: Path) -> None:
    """Concatenate shard dumps into one npz, episode-ordered."""
    loaded = [np.load(p, allow_pickle=True) for p in shards]
    names = json.loads(str(loaded[0]["geom_sets"]))
    for z, p in zip(loaded, shards):
        if json.loads(str(z["geom_sets"])) != names:
            raise SystemExit(f"{p} has different geom sets; refusing to merge")

    payload = {
        "episode": np.concatenate([z["episode"] for z in loaded]),
        "frame": np.concatenate([z["frame"] for z in loaded]),
        "geom_sets": json.dumps(names),
        "task": str(loaded[0]["task"]),
    }
    for name in names:
        payload[name] = np.concatenate([z[name] for z in loaded], axis=0)

    order = np.lexsort((payload["frame"], payload["episode"]))
    for key in ["episode", "frame", *names]:
        payload[key] = payload[key][order]
    lengths = {str(int(ep)): int((payload["episode"] == ep).sum())
               for ep in np.unique(payload["episode"])}
    payload["lengths"] = json.dumps(lengths)

    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **payload)
    print(f"merged {len(shards)} shards -> {out}: {len(payload['episode'])} rows over "
          f"{len(lengths)} episodes")
    summarize(payload, names)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=sorted(TARGET_GEOMS))
    ap.add_argument("--source-dir", type=Path, default=None,
                    help="The '<...>/lerobot' dir holding extras/episode_XXXXXX.")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--episodes", nargs="*", type=int, default=None)
    ap.add_argument("--episode-range", nargs=2, type=int, default=None,
                    metavar=("START", "STOP"), help="Half-open [START, STOP); for array jobs.")
    ap.add_argument("--max-episodes", type=int, default=None)
    ap.add_argument("--max-frames", type=int, default=None, help="Smoke-test cap per episode.")
    ap.add_argument("--reset-only", action="store_true",
                    help="One row per episode instead of one per frame. Only for targets a "
                         "full-replay probe has shown do not move (a microwave button does "
                         "not; a drawer handle does).")
    ap.add_argument("--merge", nargs="*", type=Path, default=None,
                    help="Concatenate these shard npz files into --out and exit. No simulator.")
    ap.add_argument("--split", default="target")
    ap.add_argument("--seed", type=int, default=7)
    # Nothing here reads an image, but robosuite renders every camera in `camera_names` on
    # every step regardless (`use_camera_obs` is fixed inside robocasa's create_env and cannot
    # be overridden through kwargs). One 64px camera instead of three 256px ones is ~50x fewer
    # pixels per step, which is the difference between this job taking an hour and a day.
    ap.add_argument("--image-resolution", type=int, default=64)
    ap.add_argument("--camera", default="robot0_agentview_left")
    args = ap.parse_args()

    if args.merge:
        merge_shards([p.expanduser().resolve() for p in args.merge], args.out)
        return

    if args.source_dir is None:
        raise SystemExit("--source-dir is required unless --merge is given")
    source_dir = args.source_dir.expanduser().resolve()
    cfg = TARGET_GEOMS[args.task]

    available = sorted(
        int(re.search(r"episode_(\d+)", os.path.basename(p)).group(1))
        for p in glob.glob(str(source_dir / "extras" / "episode_*"))
    )
    episodes = args.episodes if args.episodes else available
    if args.episode_range:
        lo, hi = args.episode_range
        episodes = [e for e in episodes if lo <= e < hi]
    if args.max_episodes:
        episodes = episodes[: args.max_episodes]
    print(f"task={args.task} episodes={len(episodes)} "
          f"[{episodes[0] if episodes else '-'}..{episodes[-1] if episodes else '-'}]")

    env = RoboCasa365Env(
        env_name=infer_env_name(source_dir),
        split=args.split,
        seed=args.seed,
        image_resolution=args.image_resolution,
        camera_names=[args.camera],
        use_depth=False,
        use_point_cloud=False,
        enable_render=False,
        terminate_on_success=False,
    )

    ep_col, frame_col = [], []
    cols: dict[str, list[np.ndarray]] = {name: [] for name in cfg["sets"]}
    lengths: dict[str, int] = {}
    t0 = time.time()
    try:
        for i, ep in enumerate(episodes):
            got = dump_episode(env, source_dir, ep, cfg["sets"], cfg["fixture_attr"],
                               args.max_frames, reset_only=args.reset_only)
            if got is None:
                print(f"  ep {ep}: no source dir, skipped")
                continue
            n = len(next(iter(got.values())))
            ep_col.append(np.full(n, ep, dtype=np.int64))
            frame_col.append(np.arange(n, dtype=np.int64))
            for name, arr in got.items():
                cols[name].append(arr)
            lengths[str(ep)] = n
            if i % 20 == 0 or i == len(episodes) - 1:
                rate = (time.time() - t0) / max(1, i + 1)
                print(f"  [{i + 1}/{len(episodes)}] ep {ep}: {n} frames "
                      f"({rate:.1f} s/ep, eta {rate * (len(episodes) - i - 1) / 60:.0f} min)",
                      flush=True)
    finally:
        env.close()

    if not ep_col:
        raise SystemExit("no episodes dumped")

    payload = {
        "episode": np.concatenate(ep_col),
        "frame": np.concatenate(frame_col),
        "geom_sets": json.dumps(sorted(cfg["sets"])),
        "task": args.task,
        "lengths": json.dumps(lengths),
    }
    for name, chunks in cols.items():
        payload[name] = np.concatenate(chunks, axis=0)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **payload)
    n = len(payload["episode"])
    print(f"\nwrote {n} frames over {len(lengths)} episodes -> {args.out}")

    summarize(payload, sorted(cfg["sets"]))


if __name__ == "__main__":
    main()

"""Re-lift the MolmoPoint cache from per-pixel depth instead of the voxelised cloud.

The cache's original lift projects the stored point cloud into the camera and takes the
median of the points landing in a small window around the predicted pixel. That couples the
anchor to the 1 cm voxel grid, and the coupling is not benign: a 5x5 px window subtends less
than one voxel for anything closer than ~44 cm in the agentviews (fx 221.7) and ~33 cm in the
wrist camera (fx 166.8). Inside that range whether the window catches a point is decided by
grid phase rather than by whether the surface is there, so the lift fails *hardest* on the
closest, best-aimed detections. On TurnOnMicrowave 88% of failed lifts are wrist-only, and
their pixels are the most accurate in the dataset -- median 3.7 px from the reprojected
button, against 17-21 px for the agentviews.

Nothing needs a real point to exist. The simulator gives one metric depth per pixel, so the
anchor is well defined wherever depth is valid, at whatever precision the depth has. This
script replays each episode and reads ``observation.points.{view}`` -- the organised
(H, W, 3) per-pixel cloud, already in the robot-base frame, before the workspace crop and
before ``voxel_downsample`` -- at the pixels the cache already stores.

Only the per-view lift changes. Fusion is the untouched ``fuse``/``apply_wrist`` from
``pointact.roi_sampling.molmo_anchors``, so a difference against the original cache is a
difference in lifting and nothing else. In particular the agentviews are still averaged only
when they agree: on TurnOnMicrowave 44.8% of anchors sit closer to the stop button than to
the start button, and those buttons are 5.4 cm apart, so averaging a right view with a wrong
one lands the anchor in mid-air between two plausible targets.

Why a replay: depth is never persisted. ``replay.py`` asks the env for it, ``make_point_cloud``
crops and voxelises it immediately, and only the result reaches LMDB. Regenerating it means
stepping the recorded actions again -- hence a V100 array job rather than the minutes a
re-lift from the cache would take.

The output is a cache in the same on-disk format as the input, so the evaluator, the
visualisation and the dataloader all read it with no changes:

    sbatch --export=ALL,TASK=TurnOnMicrowave \
           experiments/13_robocasa365/molmo_depth_anchors_jeanzay.slurm
    python -m data_prep.roi_sampling.dump_molmo_depth_anchors --task TurnOnMicrowave \
        --dataset-dir <task> --merge <task>/roi_meta/depth_anchors_shard*.npz
"""

from __future__ import annotations

import argparse
import glob
import time
from pathlib import Path

import numpy as np

from data_prep.robocasa365_to_lerobot.episode_index_map import MAP_NAME, load_map
from data_prep.roi_sampling.dump_target_positions import infer_env_name

# Kept in step with the cache builder; a mismatch here is a silent frame-offset bug.
STRIDE = 8


def load_episode_map(dataset_dir: Path) -> dict[int, int]:
    """{converted episode -> source episode}. Required, never assumed to be the identity.

    OpenDrawer's 514 source episodes became 496 converted ones under a renumbering, so a
    replay driven by converted indices would step the wrong episode's actions and produce
    anchors that look plausible and are wrong.
    """
    emap = load_map(dataset_dir)
    if emap is None:
        raise SystemExit(
            f"missing {dataset_dir / 'meta' / MAP_NAME}. Build it first:\n"
            f"  python -m data_prep.robocasa365_to_lerobot.episode_index_map "
            f"--source-dir <src>/lerobot --dataset-dir {dataset_dir}"
        )
    return emap


def depth_anchor_from_pixel(points_hw3: np.ndarray, uv, window: int,
                            cam_origin: np.ndarray | None = None,
                            surface_tol: float = 0.03) -> tuple[np.ndarray, int]:
    """Lift one pixel from the organised per-pixel cloud. Returns (xyz | None, n_valid).

    The window exists to survive invalid depth, not to gather support. Invalid pixels are
    dropped rather than allowed to drag the median, which is the property the original cloud
    lift was really buying -- and it is kept here.

    ``cam_origin`` selects an alternative estimator: keep only the pixels within
    ``surface_tol`` of the nearest one, then median those. It was written to test the theory
    that the window straddles a thin target (a drawer handle) and its median lands on the room
    behind it. **The theory was wrong and the estimator is worse** -- on an OpenDrawer smoke it
    turned 1.7 cm calls into 48.7 cm, and it did not rescue the calls where the plain median
    was already 99 cm off (both estimators agree there, so those are pointing failures, not
    lifting ones). Kept only so the next person does not re-derive it; default is None.
    """
    h, w = points_hw3.shape[:2]
    x, y = int(round(float(uv[0]))), int(round(float(uv[1])))
    if not (0 <= x < w and 0 <= y < h):
        return None, 0
    x0, x1 = max(0, x - window), min(w, x + window + 1)
    y0, y1 = max(0, y - window), min(h, y + window + 1)
    patch = points_hw3[y0:y1, x0:x1].reshape(-1, 3)
    valid = np.isfinite(patch).all(axis=1) & (np.abs(patch).sum(axis=1) > 0)
    if not valid.any():
        return None, 0
    pts = patch[valid]
    if cam_origin is None:
        return np.median(pts, axis=0), int(len(pts))
    rng = np.linalg.norm(pts - np.asarray(cam_origin).reshape(1, 3), axis=1)
    near = rng <= rng.min() + surface_tol
    return np.median(pts[near], axis=0), int(near.sum())


def camera_origins(env, obs) -> dict[str, np.ndarray]:
    """Each view's camera centre in the robot-base frame, at the current sim state.

    The wrist camera rides on the arm, so this has to be read per frame rather than from the
    stored calibration. Reuses the same extrinsic convention as ``dump_camera_calib``.
    """
    from pointact.roi_sampling.geometry import base2cam_from_extrinsic
    from data_prep.roi_sampling.dump_camera_calib import LEFT_CAM, RIGHT_CAM, WRIST_CAM

    base_pos = np.asarray(obs["state.base_position"], dtype=np.float64).reshape(3)
    base_quat = np.asarray(obs["state.base_rotation"], dtype=np.float64).reshape(4)
    out = {}
    for tag, cam in (("left", LEFT_CAM), ("right", RIGHT_CAM), ("wrist", WRIST_CAM)):
        _intrinsic, extrinsic = env._get_camera_matrices(cam)
        b2c = base2cam_from_extrinsic(extrinsic, base_pos, base_quat)
        out[tag] = (np.linalg.inv(b2c))[:3, 3]
    return out


def in_workspace(xyz: np.ndarray, workspace: dict) -> bool:
    """The crop the stored clouds were built under.

    Depth makes an anchor available for *any* pixel, including one on a far wall, and on
    OpenDrawer a failed lift usually means exactly that -- the agentview pointed ~180 px from
    the target on a 256 px image. Returning those unguarded would turn a fallback to uniform
    into a confident anchor on the wrong object, so out-of-workspace lifts are refused: there
    are no cloud points out there for a Gaussian to concentrate on anyway.
    """
    return bool(
        workspace["X_BBOX"][0] < xyz[0] < workspace["X_BBOX"][1]
        and workspace["Y_BBOX"][0] < xyz[1] < workspace["Y_BBOX"][1]
        and workspace["Z_BBOX"][0] < xyz[2] < workspace["Z_BBOX"][1]
    )


def export_cache_pixels(dataset_dir: Path, dirname: str, stride: int, out: Path) -> None:
    """Dump the cache's key-frame pixels to an npz, for the replay job to read.

    A separate step because the environments are split: the simulator env has no ``lmdb``
    and the root env has no MuJoCo, so nothing can read the cache and drive a replay in one
    process. Exporting the pixels -- the expensive half, 0.7 s of an 8B model each -- also
    makes the replay job's input an explicit file rather than a live database.
    """
    import lmdb
    from pointact.roi_sampling import molmo_cache
    from pointact.roi_sampling.molmo_anchors import VIEWS

    eps, frames, qids = [], [], []
    uvs = {v: [] for v in VIEWS}
    env = lmdb.open(str(dataset_dir / dirname), readonly=True, lock=False, subdir=True)
    with env.begin(buffers=True) as txn:
        for k, v in txn.cursor():
            ep_s, f_s = bytes(k).decode().split("-")
            f = int(f_s)
            if f % stride:
                continue
            for det in molmo_cache.decode_pixels(
                    np.frombuffer(bytes(v), dtype=molmo_cache.RECORD_DTYPE)):
                eps.append(int(ep_s))
                frames.append(f)
                qids.append(int(det["query_id"]))
                for view in VIEWS:
                    uv = det.get(f"{view}_uv")
                    uvs[view].append(np.asarray(uv, dtype=np.float64)
                                     if uv is not None else np.full(2, np.nan))
    env.close()
    out.parent.mkdir(parents=True, exist_ok=True)
    cols = {"ep": np.asarray(eps), "frame": np.asarray(frames),
            "query_id": np.asarray(qids)}
    for view in VIEWS:
        cols[f"{view}_uv"] = (np.stack(uvs[view], axis=0) if uvs[view]
                              else np.zeros((0, 2)))
    np.savez(out, **cols)
    print(f"wrote {out}: {len(eps)} pointing calls over "
          f"{len(set(eps))} episodes at stride {stride}")


def load_pixels_npz(path: Path) -> dict[int, dict]:
    """{converted episode: {frame: [detection dicts]}}, the shape the replay wants."""
    from pointact.roi_sampling.molmo_anchors import VIEWS

    d = np.load(path)
    out: dict[int, dict] = {}
    for i in range(len(d["ep"])):
        det = {"query_id": int(d["query_id"][i])}
        for view in VIEWS:
            uv = d[f"{view}_uv"][i]
            det[f"{view}_uv"] = None if not np.isfinite(uv).all() else uv
        out.setdefault(int(d["ep"][i]), {}).setdefault(int(d["frame"][i]), []).append(det)
    return out


def lift_episode(env, source_dir: Path, source_ep: int, frames: dict, window: int,
                 workspace: dict, agree_dist: float, wrist_accept: float,
                 require_agentview: bool, max_frames: int | None,
                 surface_tol: float = 0.03) -> dict:
    """Replay one episode and re-lift every stored key frame from per-pixel depth."""
    from pointact.roi_sampling.molmo_anchors import VIEWS, apply_wrist, fuse
    from data_prep.robocasa365_to_lerobot.replay import (
        convert_env_action_to_dataset_action, load_source_actions,
        reorder_source_action_to_env,
    )

    episode_dir = source_dir / "extras" / f"episode_{source_ep:06d}"
    if not episode_dir.exists():
        return {}
    obs, _ = env.reset(initial_state_dir=episode_dir, step_after_reset=False)

    actions = load_source_actions(source_dir, source_ep)["actions_lerobot"]
    action_envs = np.stack([reorder_source_action_to_env(a) for a in actions], axis=0)
    if max_frames is not None:
        action_envs = action_envs[:max_frames]

    want = set(frames)
    out: dict[int, list[dict]] = {}

    def lift_now(f: int) -> None:
        clouds = {v: obs.get(f"observation.points.{v}") for v in VIEWS}
        origins = camera_origins(env, obs)
        anchors = []
        for det in frames[f]:
            # Both estimators come out of the same replay: the GPU cost is the render, so
            # answering "median or nearest surface?" needs one pass, not two runs.
            fused: dict[str, tuple] = {}
            for est, origin_of in (("median", lambda _v: None), ("near", origins.get)):
                per_view: dict[str, tuple[np.ndarray, int]] = {}
                for v in VIEWS:
                    uv, cloud = det.get(f"{v}_uv"), clouds.get(v)
                    if uv is None or cloud is None:
                        continue
                    xyz, n = depth_anchor_from_pixel(np.asarray(cloud), uv, window,
                                                     origin_of(v), surface_tol)
                    if xyz is not None and in_workspace(xyz, workspace):
                        per_view[v] = (xyz, n)
                xyz, agreed = fuse(per_view, agree_dist)
                xyz, _used = apply_wrist(xyz, per_view, wrist_accept, require_agentview)
                fused[est] = (xyz if xyz is not None else np.full(3, np.nan),
                              sum(n for _, n in per_view.values()), agreed)
            anchors.append({
                "xyz": fused["median"][0],
                "xyz_near": fused["near"][0],
                "query_id": int(det["query_id"]),
                "n_support": fused["median"][1],
                "agree": fused["median"][2],
                **{f"{v}_uv": det.get(f"{v}_uv") for v in VIEWS},
            })
        out[f] = anchors

    for f, action_env in enumerate(action_envs):
        if f in want:
            lift_now(f)
        obs, _reward, done, info = env.step(convert_env_action_to_dataset_action(action_env))
        if bool(done or info.get("success", False)):
            f_end = f + 1
            if f_end in want:
                lift_now(f_end)
            break

    return out


def pack_shard(per_episode: dict[int, dict[int, list[dict]]]) -> dict:
    """Flatten to arrays for an npz shard. One row per (episode, frame, query)."""
    from pointact.roi_sampling.molmo_anchors import VIEWS

    cols: dict[str, list] = {k: [] for k in
                             ("ep", "frame", "query_id", "n_support", "agree")}
    xyz, xyz_near, uvs = [], [], {v: [] for v in VIEWS}
    for ep, frames in sorted(per_episode.items()):
        for f, anchors in sorted(frames.items()):
            for a in anchors:
                cols["ep"].append(ep)
                cols["frame"].append(f)
                cols["query_id"].append(a["query_id"])
                cols["n_support"].append(a["n_support"])
                cols["agree"].append(bool(a["agree"]))
                xyz.append(np.asarray(a["xyz"], dtype=np.float64))
                xyz_near.append(np.asarray(a["xyz_near"], dtype=np.float64))
                for v in VIEWS:
                    uv = a.get(f"{v}_uv")
                    uvs[v].append(np.asarray(uv, dtype=np.float64)
                                  if uv is not None else np.full(2, np.nan))
    out = {k: np.asarray(v) for k, v in cols.items()}
    out["xyz"] = (np.stack(xyz, axis=0) if xyz else np.zeros((0, 3)))
    out["xyz_near"] = (np.stack(xyz_near, axis=0) if xyz_near else np.zeros((0, 3)))
    for v in VIEWS:
        out[f"{v}_uv"] = (np.stack(uvs[v], axis=0) if uvs[v] else np.zeros((0, 2)))
    return out


def merge_shards(shards: list[Path], out: Path) -> None:
    parts = [np.load(s) for s in shards]
    keys = set(parts[0].files)
    for p, s in zip(parts, shards):
        if set(p.files) != keys:
            raise SystemExit(f"shard {s} has different columns; do not merge across runs")
    merged = {k: np.concatenate([p[k] for p in parts], axis=0) for k in keys}
    order = np.lexsort((merged["query_id"], merged["frame"], merged["ep"]))
    merged = {k: v[order] for k, v in merged.items()}
    seen = {}
    for e, f, q in zip(merged["ep"], merged["frame"], merged["query_id"]):
        seen[(int(e), int(f), int(q))] = seen.get((int(e), int(f), int(q)), 0) + 1
    dupes = sum(1 for c in seen.values() if c > 1)
    if dupes:
        raise SystemExit(f"{dupes} duplicate (episode, frame, query) rows -- shards overlap")
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **merged)
    n_ep = len(np.unique(merged["ep"]))
    lifted = np.isfinite(merged["xyz"]).all(axis=1)
    print(f"wrote {out}: {len(merged['ep'])} rows over {n_ep} episodes, "
          f"{lifted.mean():.1%} lifted")


def write_cache(npz_path: Path, dataset_dir: Path, out_dirname: str, stride: int,
                map_size_gb: float, field: str = "xyz") -> None:
    """Turn the merged rows into a cache in the builder's own on-disk format.

    Held across the replan window exactly as the builder does, so the dataloader needs no
    stride logic and a comparison against the original cache is like for like.
    """
    import lmdb
    from pointact.roi_sampling import molmo_cache
    from pointact.roi_sampling.molmo_anchors import VIEWS

    d = np.load(npz_path)
    src_env = lmdb.open(str(dataset_dir / "points_3views"), readonly=True, lock=False,
                        subdir=True)
    with src_env.begin() as t:
        lengths: dict[int, int] = {}
        for k, _ in t.cursor():
            ep_s, f_s = k.decode().split("-")
            ep, f = int(ep_s), int(f_s)
            lengths[ep] = max(lengths.get(ep, -1), f)
    src_env.close()

    by_key: dict[tuple[int, int], list[dict]] = {}
    for i in range(len(d["ep"])):
        rec = {"xyz": d[field][i], "query_id": int(d["query_id"][i]),
               "n_support": int(d["n_support"][i]), "agree": bool(d["agree"][i])}
        for v in VIEWS:
            uv = d[f"{v}_uv"][i]
            rec[f"{v}_uv"] = None if not np.isfinite(uv).all() else uv
        by_key.setdefault((int(d["ep"][i]), int(d["frame"][i])), []).append(rec)

    out_env = lmdb.open(str(dataset_dir / out_dirname), subdir=True,
                        map_size=int(map_size_gb * 1024 ** 3))
    n_written = 0
    with out_env.begin(write=True) as wtxn:
        for ep, last in sorted(lengths.items()):
            key_frames = sorted(f for (e, f) in by_key if e == ep)
            for i, f in enumerate(key_frames):
                rec = molmo_cache.encode_record(by_key[(ep, f)]).tobytes()
                end = key_frames[i + 1] if i + 1 < len(key_frames) else last + 1
                for ff in range(f, end):
                    wtxn.put(f"{ep}-{ff}".encode("ascii"), rec)
                    n_written += 1
    out_env.sync()
    out_env.close()
    print(f"wrote {dataset_dir / out_dirname}: {n_written} frame records")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default=None, help="Only used for the log line.")
    ap.add_argument("--dataset-dir", required=True, type=Path,
                    help="Converted LeRobot dataset (holds the cache and the episode map).")
    ap.add_argument("--source-dir", type=Path, default=None,
                    help="Source dataset with extras/episode_XXXXXX; required unless --merge.")
    ap.add_argument("--molmo-dirname", default="points_3views_molmo")
    ap.add_argument("--export-pixels", type=Path, default=None,
                    help="Root env: dump the cache's key-frame pixels here and exit.")
    ap.add_argument("--pixels-npz", type=Path, default=None,
                    help="Simulator env: the pixels to re-lift, from --export-pixels.")
    ap.add_argument("--out", type=Path, default=None, help="Shard/merged npz path.")
    ap.add_argument("--merge", nargs="*", type=Path, default=None,
                    help="Merge these shards into --out and exit (no simulator needed).")
    ap.add_argument("--write-cache", type=Path, default=None,
                    help="Turn a merged npz into an LMDB cache and exit.")
    ap.add_argument("--out-dirname", default="points_3views_molmo_depth")
    ap.add_argument("--cache-field", default="xyz", choices=("xyz", "xyz_near"),
                    help="Which estimator to write into the cache. Default 'xyz' is the "
                         "window median, the one that was measured; 'xyz_near' is the "
                         "nearest-surface variant, which measured WORSE -- see the note on "
                         "depth_anchor_from_pixel before reaching for it.")
    ap.add_argument("--surface-tol", type=float, default=0.03,
                    help="Keep window pixels within this many metres of the nearest one.")
    ap.add_argument("--point-window", type=int, default=2,
                    help="Half-width in pixels; matches the cache build.")
    ap.add_argument("--agree-dist", type=float, default=0.10)
    ap.add_argument("--wrist-accept-dist", type=float, default=0.15)
    ap.add_argument("--allow-lone-wrist", action="store_true",
                    help="Keep a wrist anchor with no agentview to corroborate it.")
    ap.add_argument("--stride", type=int, default=STRIDE)
    ap.add_argument("--image-resolution", type=int, default=256,
                    help="Must match the resolution the pixels were predicted on.")
    # Match dump_target_positions: the episodes being replayed are the target split, and the
    # seed only matters for a reset that is immediately overwritten by the recorded state.
    ap.add_argument("--split", default="target")
    ap.add_argument("--seed", type=int, default=7)
    # Shard on the *converted* episodes present in the cache, not on a source-index range:
    # the two differ wherever conversion dropped a failed replay, and a source-index split
    # would silently give some shards nothing to do and others double.
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--max-episodes", type=int, default=None)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--map-size-gb", type=float, default=4.0)
    args = ap.parse_args()

    if args.export_pixels:
        export_cache_pixels(args.dataset_dir, args.molmo_dirname, args.stride,
                            args.export_pixels)
        return

    if args.merge:
        shards = [Path(p) for pat in args.merge for p in sorted(glob.glob(str(pat)))]
        if not shards:
            raise SystemExit("--merge matched no files")
        merge_shards(shards, args.out or (args.dataset_dir / "roi_meta" / "depth_anchors.npz"))
        return

    if args.write_cache:
        write_cache(args.write_cache, args.dataset_dir, args.out_dirname, args.stride,
                    args.map_size_gb, args.cache_field)
        return

    if args.source_dir is None:
        raise SystemExit("--source-dir is required unless --merge or --write-cache")
    if args.pixels_npz is None:
        raise SystemExit("--pixels-npz is required; produce it with --export-pixels in the "
                         "root env (the simulator env has no lmdb)")

    from data_prep.robocasa365_to_lerobot.replay import DEFAULT_WORKSPACE

    emap = load_episode_map(args.dataset_dir)
    pixels = load_pixels_npz(args.pixels_npz)
    episodes = sorted(pixels)
    if args.num_shards > 1:
        episodes = episodes[args.shard::args.num_shards]  # strided: even work per shard
    if args.max_episodes:
        episodes = episodes[: args.max_episodes]
    missing = [e for e in episodes if e not in emap]
    if missing:
        raise SystemExit(f"{len(missing)} cached episodes are absent from {MAP_NAME}: "
                         f"{missing[:5]}... -- the map and the cache disagree")

    print(f"task={args.task} episodes={len(episodes)} "
          f"[{episodes[0] if episodes else '-'}..{episodes[-1] if episodes else '-'}] "
          f"window={args.point_window} lone_wrist={args.allow_lone_wrist}")

    from pointact.robot_envs.robocasa365_utils.environments import RoboCasa365Env

    env = RoboCasa365Env(
        env_name=infer_env_name(args.source_dir),
        split=args.split,
        seed=args.seed,
        image_resolution=args.image_resolution,
        use_depth=True,
        use_point_cloud=True,
        enable_render=True,
        terminate_on_success=False,
    )

    per_episode: dict[int, dict] = {}
    t0 = time.time()
    try:
        for i, ep in enumerate(episodes):
            got = lift_episode(env, args.source_dir, emap[ep], pixels[ep],
                               args.point_window, DEFAULT_WORKSPACE, args.agree_dist,
                               args.wrist_accept_dist, not args.allow_lone_wrist,
                               args.max_frames, args.surface_tol)
            if got:
                per_episode[ep] = got
            done = i + 1
            rate = (time.time() - t0) / done
            print(f"[{done}/{len(episodes)}] ep {ep} (source {emap[ep]}) "
                  f"{len(got)} key frames  {rate:.1f}s/ep "
                  f"eta {rate * (len(episodes) - done) / 60:.0f}m", flush=True)
    finally:
        try:
            env.close()
        except Exception:
            pass

    out = args.out or (args.dataset_dir / "roi_meta" / "depth_anchors.npz")
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **pack_shard(per_episode))
    print(f"wrote {out} ({len(per_episode)} episodes)")


if __name__ == "__main__":
    main()

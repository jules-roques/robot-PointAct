"""How accurately does MolmoPoint find the thing the instruction names?

Scores the frozen pointer's cached detections against simulator ground truth, in 3D metres
and in image pixels, for one task. This measures the *detector*, independently of any policy
trained on it -- the Stage 3 policy numbers conflate the anchor's quality with everything
downstream of it, and could not say which was at fault.

    python -m data_prep.roi_sampling.eval_molmo_accuracy \
        --dataset-dir <task> --gt labels          # OpenDrawer: MuJoCo handle labels
        --dataset-dir <task> --gt geom --target start_button   # TurnOnMicrowave

Two ground-truth sources, because the two tasks can support different ones:

``labels``
    Centroid of the points MuJoCo labels as the handle (``points_3views_labels``, label 4).
    This is exactly the anchor the GT-oracle arm is centred on, so an error measured against
    it is the gap between the molmo arm and the oracle arm's *input* -- the comparison that
    makes the two policy numbers commensurable. OpenDrawer only.
``geom``
    Per-frame ``geom_xpos`` of the target geom, dumped by ``dump_target_positions.py``. Exact,
    available for any task, and the only option where segmentation labels do not exist or
    would be a handful of pixels (a microwave start button).

What is reported, and why each is here:

* **Outcome split** over every (key frame, query) the pointer was asked. "Pointed nowhere" and
  "pointed, could not lift" are different failures -- the first is the model, the second is
  the geometry -- and the cache's frame-coverage number merges them.
* **3D error distribution**, plus the hit rate inside the sampling sigma (8 cm). Inside sigma
  the Gaussian still concentrates the budget on the target; outside it, the arm is sampling
  the wrong region confidently, which is the failure the policy results imply.
* **Distractor margin**, where the task has one. A microwave has a stop button beside its
  start button; "off by 6 cm" and "on the wrong button" are the same number and different
  problems. Reported as the share of anchors nearer a distractor than the target.
* **2D pixel error per camera**, from the stored pixels against the reprojected GT. This
  separates a pointing error from a lifting error: a few pixels off with metres of 3D error
  means the lift found the wrong surface, not that the model was wrong.
* **Replan jitter**: how far the anchor moves between consecutive pointing calls, against how
  far the target itself moved. A static target with a jumping anchor makes the sampled region
  flicker across the episode, which is one of the two standing hypotheses for why the molmo
  arm underperforms the oracle on identical machinery.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import lmdb
import msgpack
import msgpack_numpy
import numpy as np
from tqdm.auto import tqdm

from data_prep.robocasa365_to_lerobot.episode_index_map import MAP_NAME
from data_prep.robocasa365_to_lerobot.episode_index_map import load_map as load_episode_map
from data_prep.roi_sampling.build_molmo_cache import (
    VIEWS, load_calib, read_episode_states, read_meta, wrist_cam_at,
)
from pointact.roi_sampling import molmo_cache
from pointact.roi_sampling.geometry import project_base_points

msgpack_numpy.patch()

#: The sampling Gaussian's sigma, in metres. An anchor inside this still puts the bulk of the
#: point budget on the target; outside it the arm concentrates on the wrong place.
SIGMA = 0.08

#: Error thresholds reported as hit rates, in metres.
THRESHOLDS = (0.02, 0.04, SIGMA, 0.16, 0.32)

#: MuJoCo segmentation level for the manipulated handle (see environments.POINT_LABEL_*).
HANDLE_LABEL = 4


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


#: Pixel-error thresholds, on the 256 px frames the pointer saw.
PX_THRESHOLDS = (2, 4, 8, 16, 32)


def describe(errs: np.ndarray, unit: str = "cm", scale: float = 100.0,
             thresholds=THRESHOLDS) -> dict:
    """Summary of an error distribution.

    ``scale`` converts the stored unit to the reported one -- metres to centimetres for the
    3D legs, identity for the pixel leg. Getting this wrong is silent and looks plausible,
    so the unit is part of every key name rather than left to the caller's docstring.
    """
    if not len(errs):
        return {"n": 0}
    out = {
        "n": int(len(errs)),
        f"median_{unit}": float(np.median(errs) * scale),
        f"mean_{unit}": float(errs.mean() * scale),
        f"p25_{unit}": float(np.percentile(errs, 25) * scale),
        f"p75_{unit}": float(np.percentile(errs, 75) * scale),
        f"p90_{unit}": float(np.percentile(errs, 90) * scale),
    }
    for t in thresholds:
        out[f"within_{t * scale:g}{unit}"] = float((errs < t).mean())
    return out


def load_gt_geom(path: Path, target: str, distractors: list[str],
                 converted_to_source: dict[int, int] | None = None):
    """``lookup(name, ep, frame) -> xyz | None`` over a target-position dump.

    ``ep`` is a **converted** episode index, because that is what the anchor cache is keyed
    by. The dump is indexed by *source* episode, and on OpenDrawer the two differ -- 18 failed
    replays were dropped and the rest renumbered. ``converted_to_source`` bridges them; pass
    None only when a verified map says the two are the identity.

    An episode dumped with ``--reset-only`` has a single row, because its target does not
    move; that row answers for every frame of the episode. Anything else is indexed per
    frame, and a frame past the end of the dump has no answer rather than a wrong one.
    """
    z = np.load(path, allow_pickle=True)
    available = json.loads(str(z["geom_sets"]))
    for name in [target, *distractors]:
        if name not in available:
            raise SystemExit(f"{path} has no geom set '{name}'; it has {available}")

    eps, frames = z["episode"], z["frame"]
    tables = {name: {} for name in [target, *distractors]}
    static: dict[int, bool] = {}
    for name in tables:
        arr = z[name]
        for ep, fr, xyz in zip(eps.tolist(), frames.tolist(), arr):
            tables[name][(ep, fr)] = xyz
    for ep in np.unique(eps).tolist():
        static[ep] = int((eps == ep).sum()) == 1

    def lookup(name: str, ep: int, frame: int):
        src = ep if converted_to_source is None else converted_to_source.get(ep)
        if src is None:
            return None
        got = tables[name].get((src, 0 if static.get(src) else frame))
        return None if got is None else np.asarray(got, dtype=np.float64)

    return lookup, [target, *distractors]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", required=True, type=Path)
    ap.add_argument("--molmo-dirname", default="points_3views_molmo")
    ap.add_argument("--points-dirname", default="points_3views")
    ap.add_argument("--labels-dirname", default="points_3views_labels")
    ap.add_argument("--gt", choices=("labels", "geom"), default="labels")
    ap.add_argument("--gt-npz", type=Path, default=None,
                    help="Defaults to <dataset>/roi_meta/target_positions.npz.")
    ap.add_argument("--target", default=None,
                    help="Geom set holding the target (--gt geom).")
    ap.add_argument("--distractors", nargs="*", default=[],
                    help="Geom sets to measure the target against (--gt geom).")
    ap.add_argument("--query-id", type=int, default=0)
    ap.add_argument("--stride", type=int, default=8, help="Must match the cache's build stride.")
    ap.add_argument("--max-episodes", type=int, default=None)
    ap.add_argument("--calib", type=Path, default=None)
    ap.add_argument("--no-pixels", action="store_true", help="Skip the 2D reprojection leg.")
    ap.add_argument("--out", type=Path, default=None, help="Write the full report as JSON.")
    ap.add_argument("--dump-rows", type=Path, default=None,
                    help="Write per-anchor rows (episode, frame, error, phase, agreement) as "
                         "npz. The summary hides the shape of the distribution, and on these "
                         "tasks the shape is the finding.")
    args = ap.parse_args()

    d = args.dataset_dir.expanduser().resolve()
    lengths, tasks = read_meta(d / "meta")
    episodes = sorted(lengths)
    if args.max_episodes:
        episodes = episodes[: args.max_episodes]
    info = json.loads((d / "meta" / "info.json").read_text())
    chunks_size = int(info.get("chunks_size", 1000))

    gt_lookup = None
    distractors = list(args.distractors)
    if args.gt == "geom":
        if not args.target:
            raise SystemExit("--gt geom needs --target")
        # A target-position dump is indexed by source episode. Refusing to guess: without a
        # verified map, a task whose conversion dropped episodes would join every row to the
        # wrong episode and still produce plausible-looking numbers.
        emap = load_episode_map(d)
        if emap is None:
            raise SystemExit(
                f"--gt geom needs {d / 'meta' / MAP_NAME} to join source-indexed ground truth "
                f"to converted episodes; build it with "
                f"`python -m data_prep.robocasa365_to_lerobot.episode_index_map`")
        gt_lookup, _names = load_gt_geom(
            args.gt_npz or d / "roi_meta" / "target_positions.npz",
            args.target, distractors, converted_to_source=emap)

    cams, image_hw, wrist_K, wrist_eef2cam = load_calib(
        args.calib or d / "roi_meta" / "camera_calib.npz")
    static_by = {c["name"]: c for c in cams}
    states_cache: dict[int, np.ndarray] = {}

    def cams_at(ep: int, f: int) -> dict:
        if wrist_K is None:
            return static_by
        if ep not in states_cache:
            states_cache[ep] = read_episode_states(d, ep, chunks_size)
        w = wrist_cam_at(states_cache[ep], f, wrist_K, wrist_eef2cam)
        return dict(static_by, **({"wrist": w} if w else {}))

    me = lmdb.open(str(d / args.molmo_dirname), readonly=True, lock=False, readahead=False)
    pe = lmdb.open(str(d / args.points_dirname), readonly=True, lock=False, readahead=False)
    lab_dir = d / args.labels_dirname
    le = lmdb.open(str(lab_dir), readonly=True, lock=True) if lab_dir.exists() else None
    if args.gt == "labels" and le is None:
        raise SystemExit(f"--gt labels needs {lab_dir}; build it with replay.py --labels-only "
                         f"or use --gt geom")
    # The cloud is only needed to turn per-point labels into a centroid.
    need_cloud = args.gt == "labels"

    counts = {"asked": 0, "nowhere": 0, "no_lift": 0, "anchored": 0}
    errs: list[float] = []
    rows: list[dict] = []          # per anchored query, for the breakdowns
    px_err = {v: [] for v in VIEWS}
    px_gt_visible = {v: 0 for v in VIEWS}
    px_pointed = {v: 0 for v in VIEWS}
    jitter_anchor: list[float] = []
    jitter_gt: list[float] = []
    no_gt = 0

    with me.begin(buffers=True) as mt, pe.begin(buffers=True) as pt:
        lt = le.begin(buffers=True) if le else None
        for ep in tqdm(episodes, desc="scoring", unit="ep"):
            prev_anchor = prev_gt = None
            for f in range(0, lengths[ep], args.stride):
                k = f"{ep}-{f}".encode("ascii")
                mb = mt.get(k)
                if mb is None:
                    continue
                counts["asked"] += 1

                # Ground truth first: a frame with no reference cannot score anything, and
                # counting it as a miss would flatter or punish the model arbitrarily.
                gt = None
                if args.gt == "geom":
                    gt = gt_lookup(args.target, ep, f)
                else:
                    lb = lt.get(k)
                    pb = pt.get(k) if need_cloud else None
                    if lb is not None and pb is not None:
                        lab = msgpack.unpackb(bytes(lb)).astype(np.uint8)
                        cloud = msgpack.unpackb(bytes(pb)).astype(np.float64)[:, :3]
                        hit = cloud[lab == HANDLE_LABEL]
                        gt = hit.mean(axis=0) if len(hit) else None
                if gt is None:
                    no_gt += 1
                    continue

                dets = [x for x in molmo_cache.decode_pixels(
                    np.frombuffer(bytes(mb), dtype=molmo_cache.RECORD_DTYPE))
                    if x["query_id"] == args.query_id]
                det = dets[0] if dets else None

                if det is None:
                    counts["nowhere"] += 1
                elif not np.isfinite(det["xyz"]).all():
                    counts["no_lift"] += 1
                else:
                    counts["anchored"] += 1
                    err = float(np.linalg.norm(det["xyz"] - gt))
                    errs.append(err)
                    row = {"ep": ep, "frame": f, "err": err,
                           "agree": bool(det["agree"]), "n_support": int(det["n_support"]),
                           "phase": f / max(1, lengths[ep] - 1),
                           # Both positions, so an analysis can ask *where* a miss went --
                           # a summary statistic cannot distinguish "the other drawer" from
                           # "the target's own earlier position".
                           "anchor": det["xyz"].astype(np.float64), "gt": gt.astype(np.float64)}
                    for name in distractors:
                        g = gt_lookup(name, ep, f)
                        if g is not None:
                            row[f"d_{name}"] = float(np.linalg.norm(det["xyz"] - g))
                    rows.append(row)

                    if prev_anchor is not None:
                        jitter_anchor.append(float(np.linalg.norm(det["xyz"] - prev_anchor)))
                        jitter_gt.append(float(np.linalg.norm(gt - prev_gt)))
                    prev_anchor, prev_gt = det["xyz"].copy(), gt.copy()

                # 2D leg: where the model pointed vs where the GT projects, per camera. Runs
                # even when the lift failed -- that is exactly the case it explains.
                if not args.no_pixels and det is not None:
                    cam_by = cams_at(ep, f)
                    for view in VIEWS:
                        cam = cam_by.get(view)
                        uv = det.get(f"{view}_uv")
                        if cam is None:
                            continue
                        proj, _z, in_front = project_base_points(
                            gt.reshape(1, 3), cam["intrinsic"], cam["base2cam"])
                        h, w = image_hw
                        visible = bool(in_front[0] and 0 <= proj[0, 0] < w and 0 <= proj[0, 1] < h)
                        px_gt_visible[view] += int(visible)
                        if uv is None:
                            continue
                        px_pointed[view] += 1
                        if visible:
                            px_err[view].append(float(np.linalg.norm(np.asarray(uv) - proj[0])))

    me.close(); pe.close()
    if le:
        le.close()

    e = np.asarray(errs)
    report = {
        "task": d.name,
        "gt_source": args.gt,
        "gt_target": args.target if args.gt == "geom" else f"label {HANDLE_LABEL} centroid",
        "episodes": len(episodes),
        "stride": args.stride,
        "query_id": args.query_id,
        "keyframes_with_gt": counts["asked"] - no_gt,
        "keyframes_without_gt": no_gt,
        "outcomes": {
            "anchored": counts["anchored"],
            "pointed_but_no_lift": counts["no_lift"],
            "pointed_nowhere": counts["nowhere"],
            "anchored_rate": counts["anchored"] / max(1, counts["asked"] - no_gt),
        },
        "error_3d": describe(e),
    }

    if len(rows):
        agree = np.array([r["agree"] for r in rows])
        report["error_3d_by_agreement"] = {
            "agentviews_agree": describe(e[agree]),
            "agentviews_disagree": describe(e[~agree]),
        }
        phase = np.array([r["phase"] for r in rows])
        report["error_3d_by_phase"] = {
            f"{lo:.1f}-{lo + 0.25:.1f}": describe(e[(phase >= lo) & (phase < lo + 0.25)])
            for lo in (0.0, 0.25, 0.5, 0.75)
        }

    if distractors:
        margins = {}
        for name in distractors:
            key = f"d_{name}"
            have = np.array([r[key] for r in rows if key in r])
            same = np.array([r["err"] for r in rows if key in r])
            if len(have):
                margins[name] = {
                    "n": int(len(have)),
                    "closer_to_distractor": float((have < same).mean()),
                    "median_distance_cm": float(np.median(have) * 100),
                }
        report["distractors"] = margins

    if not args.no_pixels:
        report["error_2d_px"] = {
            v: {**describe(np.asarray(px_err[v]) if px_err[v] else np.array([]),
                           unit="px", scale=1.0, thresholds=PX_THRESHOLDS),
                "gt_visible_frames": px_gt_visible[v], "pointed_frames": px_pointed[v]}
            for v in VIEWS
        }

    if jitter_anchor:
        report["replan_jitter"] = {
            "anchor_step_median_cm": float(np.median(jitter_anchor) * 100),
            "anchor_step_p90_cm": float(np.percentile(jitter_anchor, 90) * 100),
            "gt_step_median_cm": float(np.median(jitter_gt) * 100),
            "gt_step_p90_cm": float(np.percentile(jitter_gt, 90) * 100),
            "n": len(jitter_anchor),
        }

    # OpenDrawer's instructions name a side, and picking the named drawer is the capability
    # that motivated a pointing model over a box detector. Split on it where it exists.
    sides = {}
    for side in ("left", "right"):
        sel = [r["err"] for r in rows if side in tasks.get(r["ep"], "").lower()]
        if sel:
            sides[side] = describe(np.asarray(sel))
    if len(sides) > 1:
        report["error_3d_by_instruction_side"] = sides

    print(json.dumps(report, indent=2))

    o = report["outcomes"]
    n_scored = max(1, counts["asked"] - no_gt)
    print(f"\n=== {d.name} / {report['gt_target']} ===")
    print(f"scored {n_scored} pointing calls over {len(episodes)} episodes "
          f"({no_gt} key frames had no ground truth and were dropped)")
    print(f"  anchored               {o['anchored']:>7}  {pct(o['anchored'] / n_scored)}")
    print(f"  pointed, lift failed   {o['pointed_but_no_lift']:>7}  "
          f"{pct(o['pointed_but_no_lift'] / n_scored)}")
    print(f"  pointed nowhere        {o['pointed_nowhere']:>7}  "
          f"{pct(o['pointed_nowhere'] / n_scored)}")
    if len(e):
        s = report["error_3d"]
        print(f"  3D error: median {s['median_cm']:.1f} cm, "
              f"p75 {s['p75_cm']:.1f}, p90 {s['p90_cm']:.1f}")
        print("  within: " + "  ".join(
            f"{int(t * 100)}cm {pct(s[f'within_{int(t * 100)}cm'])}" for t in THRESHOLDS))
        # The end-to-end number: of everything the pointer was asked, how often did it put the
        # Gaussian on the target? Misses and failed lifts both fall back to uniform.
        print(f"  usable anchors (<{int(SIGMA * 100)} cm of target, over all calls): "
              f"{pct(s[f'within_{int(SIGMA * 100)}cm'] * o['anchored'] / n_scored)}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2))
        print(f"\nwrote {args.out}")

    if args.dump_rows and rows:
        args.dump_rows.parent.mkdir(parents=True, exist_ok=True)
        cols = {k: np.array([r[k] for r in rows])
                for k in ("ep", "frame", "err", "agree", "n_support", "phase",
                          "anchor", "gt")}
        for name in distractors:
            key = f"d_{name}"
            cols[key] = np.array([r.get(key, np.nan) for r in rows])
        np.savez(args.dump_rows, **cols)
        print(f"wrote {args.dump_rows} ({len(rows)} rows)")


if __name__ == "__main__":
    main()

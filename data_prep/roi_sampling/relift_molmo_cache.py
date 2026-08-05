"""Re-derive the 3D anchors of an existing MolmoPoint cache at new lift parameters.

The cache stores both halves of the pipeline: the per-view **pixels**, which cost 0.7 s of
an 8B model each, and the **3D anchor**, which is 1.7 ms of geometry. Only the first is
expensive, so changing the window, the in-window minimum or the fusion distance does not
need the GPU at all -- this re-lifts from the stored pixels in minutes, no model.

    # sweep, writing nothing:
    python -m data_prep.roi_sampling.relift_molmo_cache --dataset-dir <task> --sweep 1 2 3 4

    # commit one setting to a new cache:
    python -m data_prep.roi_sampling.relift_molmo_cache --dataset-dir <task> \
        --point-window 1 --out-dirname points_3views_molmo_w1

On OpenDrawer, where MuJoCo labels the handle, the sweep also reports the distance to that
ground truth, so a window can be chosen on measured accuracy rather than on coverage alone.

**Caches written before pixels-on-failure was fixed** only kept pixels for frames that
lifted successfully, so a re-lift cannot recover the frames that failed. The tool reports
how many records carry no pixels at all; if that count is high, rebuild rather than re-lift.
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

from pointact.roi_sampling import molmo_cache
from data_prep.roi_sampling.build_molmo_cache import (
    VIEWS, anchor_from_pixel, fuse, load_calib, read_meta,
)

msgpack_numpy.patch()


def relift(cloud_xyz, pixels, cam_by, image_hw, window, min_in_window, agree_dist):
    """One frame's stored detections -> new anchor dicts."""
    out = []
    for det in pixels:
        per_view, uvs = {}, {}
        for view in VIEWS:
            uv = det.get(f"{view}_uv")
            if uv is None:
                continue
            uvs[view] = uv
            got = anchor_from_pixel(cloud_xyz, cam_by[view], uv, image_hw,
                                    window, min_in_window)
            if got is not None:
                per_view[view] = got
        xyz, agreed = fuse(per_view, agree_dist)
        out.append({
            "xyz": xyz if xyz is not None else np.full(3, np.nan),
            "query_id": det["query_id"],
            "n_support": sum(n for _, n in per_view.values()),
            "left_uv": uvs.get("left"), "right_uv": uvs.get("right"),
            "agree": agreed,
            "_both": len(per_view) == 2,
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", required=True, type=Path)
    ap.add_argument("--in-dirname", default="points_3views_molmo")
    ap.add_argument("--out-dirname", default=None,
                    help="Omit to evaluate only and write nothing.")
    ap.add_argument("--points-dirname", default="points_3views")
    ap.add_argument("--labels-dirname", default="points_3views_labels")
    ap.add_argument("--calib", type=Path, default=None)
    ap.add_argument("--sweep", nargs="*", type=int, default=None,
                    help="Window half-widths to evaluate instead of writing.")
    ap.add_argument("--point-window", type=int, default=1)
    ap.add_argument("--min-in-window", type=int, default=1)
    ap.add_argument("--agree-dist", type=float, default=0.10)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--max-episodes", type=int, default=None,
                    help="Evaluate on a subset; writing always covers everything.")
    ap.add_argument("--map-size-gb", type=float, default=4.0)
    args = ap.parse_args()

    d = args.dataset_dir.expanduser().resolve()
    cams, image_hw = load_calib(args.calib or d / "roi_meta" / "camera_calib.npz")
    cam_by = {c["name"]: c for c in cams}
    lengths, _ = read_meta(d / "meta")
    episodes = sorted(lengths)

    pe = lmdb.open(str(d / args.points_dirname), readonly=True, lock=False, readahead=False)
    me = lmdb.open(str(d / args.in_dirname), readonly=True, lock=False, readahead=False)
    lab_dir = d / args.labels_dirname
    le = lmdb.open(str(lab_dir), readonly=True, lock=True) if lab_dir.exists() else None

    # Load the strided frames once; every window setting then re-lifts the same data.
    eval_eps = episodes if args.max_episodes is None else episodes[: args.max_episodes]
    samples, no_pixels, total = [], 0, 0
    with pe.begin(buffers=True) as pt, me.begin(buffers=True) as mt:
        lt = le.begin(buffers=True) if le else None
        for ep in tqdm(eval_eps, desc="loading", unit="ep"):
            for f in range(0, lengths[ep], args.stride):
                k = f"{ep}-{f}".encode("ascii")
                pb, mb = pt.get(k), mt.get(k)
                if pb is None or mb is None:
                    continue
                total += 1
                px = molmo_cache.decode_pixels(np.frombuffer(bytes(mb),
                                                             dtype=molmo_cache.RECORD_DTYPE))
                if not px:
                    no_pixels += 1
                    continue
                cloud = msgpack.unpackb(bytes(pb)).astype(np.float64)[:, :3]
                gt = None
                if lt is not None:
                    lb = lt.get(k)
                    if lb is not None:
                        lab = msgpack.unpackb(bytes(lb)).astype(np.uint8)
                        h = cloud[lab == 4]
                        gt = h.mean(0) if len(h) else None
                samples.append((ep, f, cloud, px, gt))

    print(f"\n{d.name}: {len(samples)} strided frames with stored pixels; "
          f"{no_pixels}/{total} carry none")
    if no_pixels:
        print("  ^ those were written before pixels-on-failure was fixed and cannot be "
              "re-lifted; rebuild if that fraction matters.")

    windows = args.sweep if args.sweep else [args.point_window]
    if args.sweep or args.out_dirname is None:
        print(f"\n{'win':>4} {'px':>5} {'anchors':>8} {'both views':>11} {'agree':>7} "
              f"{'med support':>12} {'median err':>11} {'<sigma':>7}")
        for win in windows:
            got = both = agreed = 0
            sup, errs = [], []
            for _, _, cloud, px, gt in samples:
                res = relift(cloud, px, cam_by, image_hw, win, args.min_in_window,
                             args.agree_dist)
                for a in res:
                    if not np.isfinite(a["xyz"]).all():
                        continue
                    got += 1
                    sup.append(a["n_support"])
                    both += int(a["_both"])
                    agreed += int(a["agree"])
                    if gt is not None and a["query_id"] == 0:
                        errs.append(float(np.linalg.norm(a["xyz"] - gt)))
            n_q = sum(len(px) for _, _, _, px, _ in samples)
            e = np.array(errs) if errs else None
            err_s = f"{np.median(e)*100:>10.1f}" if e is not None else f"{'-':>11}"
            sig_s = f"{(e < 0.08).mean():>6.0%}" if e is not None else f"{'-':>7}"
            print(f"{win:>4} {(2*win+1)**2:>5} {got}/{n_q:<5} {both/max(1,got):>10.0%} "
                  f"{agreed/max(1,both):>6.0%} {int(np.median(sup)) if sup else 0:>12} "
                  f"{err_s} {sig_s}")

    if args.out_dirname:
        out_dir = d / args.out_dirname
        out_dir.mkdir(parents=True, exist_ok=True)
        oe = lmdb.open(str(out_dir), map_size=int(args.map_size_gb * (1024 ** 3)))
        written = 0
        with pe.begin(buffers=True) as pt, me.begin(buffers=True) as mt, \
                oe.begin(write=True) as wt:
            for ep in tqdm(episodes, desc="writing", unit="ep"):
                key_frames = list(range(0, lengths[ep], args.stride))
                for ki, f in enumerate(key_frames):
                    k = f"{ep}-{f}".encode("ascii")
                    pb, mb = pt.get(k), mt.get(k)
                    rec = molmo_cache.empty_record()
                    if pb is not None and mb is not None:
                        px = molmo_cache.decode_pixels(
                            np.frombuffer(bytes(mb), dtype=molmo_cache.RECORD_DTYPE))
                        if px:
                            cloud = msgpack.unpackb(bytes(pb)).astype(np.float64)[:, :3]
                            res = relift(cloud, px, cam_by, image_hw, args.point_window,
                                         args.min_in_window, args.agree_dist)
                            rec = molmo_cache.encode_record(res)
                    blob = rec.tobytes()
                    end = key_frames[ki + 1] if ki + 1 < len(key_frames) else lengths[ep]
                    for ff in range(f, end):
                        wt.put(f"{ep}-{ff}".encode("ascii"), blob)
                        written += 1
        oe.sync()
        oe.close()
        print(f"\nwrote {written} frames -> {out_dir}")
        with open(d / "roi_meta" / f"relift_{args.out_dirname}.json", "w") as fh:
            json.dump({k: str(v) for k, v in vars(args).items()}, fh, indent=2)

    pe.close(); me.close()
    if le:
        le.close()


if __name__ == "__main__":
    main()

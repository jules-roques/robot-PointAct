"""MolmoPoint anchors during a rollout: the eval-time counterpart of the anchor cache.

The training arm reads a precomputed LMDB keyed by ``{episode}-{frame}``. Those keys do not
exist at eval time -- every rollout is a freshly randomised scene -- so the anchor has to be
produced live, at the same replan cadence the cache was built at. This module is what fills
``observation.sampling_anchor`` for a molmo-trained checkpoint.

Runs inside ``envs/robocasa365`` alongside the simulator, and therefore imports nothing
heavier than numpy and zmq: MolmoPoint itself lives in a third process (``envs/molmo``,
transformers 4.57.1) behind ``scripts/run_molmo_server.py``, because its transformers pin is
incompatible with both the trainer and the simulator.

Lifting differs from the cache builder in exactly one respect, and deliberately. The builder
had only a fused base-frame cloud on disk, so it recovered the 2D->3D correspondence by
projecting that cloud back into the camera with a stored calibration and taking the median
of the points landing in a small window. The simulator hands us the correspondence directly:
``observation.points.<cam>`` is an (H, W, 3) base-frame point per *pixel*, row-major and
aligned 1:1 with ``observation.images.<cam>_image`` (RoboCasa365Env._get_obs). So the same
window median is taken by indexing rather than reprojecting -- no calibration file, no
projection round-trip, and no chance of another camera's points contaminating the window.

The window, the workspace crop and the support rule are kept identical to the builder's, so
the anchor is the same quantity computed a more direct way. That claim is measured rather
than asserted: on OpenDrawer the simulator can also supply the ground-truth handle, and
``AnchorStats`` reports the live anchor error against it for comparison with the 4.3 cm
median the cache was audited at.
"""

from __future__ import annotations

import numpy as np

from pointact.roi_sampling.molmo_anchors import (
    STATIC_VIEWS,
    VIEWS,
    apply_wrist,
    fuse,
    queries_for,
)

#: Half-width in pixels of the box a returned point is padded into. Matches
#: build_molmo_cache.py's --point-window default, which was measured: on these 256x256
#: renders the median anchor error is 3.2 cm at window 2 against 4.4 cm at window 6, because
#: a handle is only a few pixels across and a wide window pulls in the cabinet face behind it.
POINT_WINDOW = 2

#: Fewer valid cloud points than this in the window -> no anchor from that view. 1, i.e.
#: effectively off, matching the builder: for a single pointed pixel a sparse neighbourhood
#: is not evidence against the detection, and requiring 5 sent 35% of TurnOnMicrowave frames
#: to a uniform fallback.
MIN_IN_WINDOW = 1

#: Metres. Agentview anchors closer than this are averaged; further apart, the
#: better-supported one wins. Matches the builder's --agree-dist.
AGREE_DIST = 0.10

#: Metres. The wrist anchor is adopted only within this of the agentview one. Matches the
#: builder's --wrist-accept-dist.
WRIST_ACCEPT_DIST = 0.15


def lift_pixel(
    point_map: np.ndarray,
    uv,
    workspace: dict | None,
    window: int = POINT_WINDOW,
    min_in_window: int = MIN_IN_WINDOW,
):
    """One pixel -> (base-frame anchor, support), or None.

    ``point_map`` is (H, W, 3) in the robot-base frame. The pixel is padded into a
    ``2*window+1`` box and the anchor is the median of the valid points inside it.

    "Valid" means inside the workspace box, which does two jobs at once. It reproduces the
    builder's behaviour -- that lifted through a cloud already cropped to the workspace, so a
    pixel aimed outside it produced no anchor rather than a confident wrong one -- and it
    discards invalid-depth pixels, which ``depth_to_point_cloud`` keeps (it drops nothing, so
    a bad depth becomes a point at or near the camera origin rather than a hole).
    """
    h, w = point_map.shape[:2]
    x, y = int(round(float(uv[0]))), int(round(float(uv[1])))
    if not (0 <= x < w and 0 <= y < h):
        return None
    x0, x1 = max(0, x - window), min(w, x + window + 1)
    y0, y1 = max(0, y - window), min(h, y + window + 1)
    patch = np.asarray(point_map[y0:y1, x0:x1], dtype=np.float64).reshape(-1, 3)

    keep = np.isfinite(patch).all(axis=1)
    if workspace is not None:
        for i, key in enumerate(("X_BBOX", "Y_BBOX", "Z_BBOX")):
            lo, hi = workspace[key]
            keep &= (patch[:, i] >= lo) & (patch[:, i] <= hi)
    patch = patch[keep]
    if len(patch) < min_in_window:
        return None
    return np.median(patch, axis=0), int(len(patch))


class AnchorStats:
    """Running counts, printed once per eval so a silent degradation is visible.

    The arm's whole failure mode is quiet: if the pointer answers nothing, every frame falls
    back to a uniform draw and the run still reports a plausible success rate. These counters
    are what distinguish "the method did not help" from "the method did not run".
    """

    def __init__(self):
        self.replans = 0
        self.queries = 0
        self.query_hits = 0
        self.frames_with_anchor = 0
        self.view_hits = {v: 0 for v in VIEWS}
        self.both_views = 0
        self.agree = 0
        self.wrist_lifted = 0
        self.wrist_adopted = 0
        self.gt_errors: list[float] = []

    def summary(self) -> dict:
        n = max(1, self.queries)
        return {
            "replans": self.replans,
            "queries": self.queries,
            "query_hit_rate": round(self.query_hits / n, 4),
            "frame_cover": round(self.frames_with_anchor / max(1, self.replans), 4),
            "view_hits": dict(self.view_hits),
            "agree_rate": round(self.agree / max(1, self.both_views), 4),
            "wrist_adopt_rate": round(self.wrist_adopted / max(1, self.wrist_lifted), 4),
            "wrist_lifted": self.wrist_lifted,
            "gt_error_median_m": (round(float(np.median(self.gt_errors)), 4)
                                  if self.gt_errors else None),
            "gt_error_n": len(self.gt_errors),
        }


class LiveMolmoAnchor:
    """Points at the task's target(s) each replan and lifts the result to a 3D anchor.

    Args:
        task: RoboCasa365 task name, used to select the pointing queries.
        anchor_ids: which of the task's queries become Gaussian centres. Must match the
            checkpoint's ``molmo_anchor_ids`` -- [0] is the manipulated object alone, [0, 1]
            adds the destination. Only the selected queries are asked, so the object-only arm
            costs one forward per replan rather than two.
        client: a connected ``PolicyClient`` for the molmo pointing server.
    """

    def __init__(self, task: str, anchor_ids, client, verbose: bool = False):
        self.task = task
        self.anchor_ids = tuple(int(i) for i in anchor_ids)
        self.client = client
        self.verbose = verbose
        self.stats = AnchorStats()
        # Validate here rather than at the first replan, which is after the 18 GB pointer has
        # loaded and the first scene has been reset. The instruction only picks the wording,
        # never the count, so the empty one is enough to check the ids against.
        n = len(queries_for(task, ""))
        if not self.anchor_ids or max(self.anchor_ids) >= n:
            raise ValueError(
                f"molmo_anchor_ids={self.anchor_ids} out of range for {task!r}, which defines "
                f"{n} pointing quer{'y' if n == 1 else 'ies'}"
            )

    def __call__(self, obs: dict, workspace: dict | None = None) -> np.ndarray | None:
        """The (K, 3) sampling anchors for this frame, or None to fall back to uniform."""
        queries = queries_for(self.task, obs.get("task", ""))
        wanted = [i for i in self.anchor_ids if i < len(queries)]
        if not wanted:
            return None

        images = [np.asarray(obs[f"observation.images.{v}_image"], dtype=np.uint8)
                  for v in VIEWS]
        point_maps = {v: np.asarray(obs[f"observation.points.{v}"], dtype=np.float32)
                      for v in VIEWS}

        self.stats.replans += 1
        anchors = []
        for qi in wanted:
            dets = self.client.call_endpoint(
                "point", {"images": images, "prompt": queries[qi]})
            per_view = {}
            for d in dets:
                view = VIEWS[d["image_num"]] if d["image_num"] < len(VIEWS) else None
                if view is None or view in per_view:
                    continue  # first point per view; extras are other instances
                self.stats.view_hits[view] += 1
                got = lift_pixel(point_maps[view], (d["x"], d["y"]), workspace)
                if got is not None:
                    per_view[view] = got

            self.stats.queries += 1
            if sum(v in per_view for v in STATIC_VIEWS) == 2:
                self.stats.both_views += 1
            xyz, agreed = fuse(per_view, AGREE_DIST)
            if "wrist" in per_view:
                self.stats.wrist_lifted += 1
            xyz, used_wrist = apply_wrist(xyz, per_view, WRIST_ACCEPT_DIST,
                                          require_agentview=True)
            if used_wrist:
                self.stats.wrist_adopted += 1
            if xyz is not None:
                self.stats.query_hits += 1
                self.stats.agree += int(agreed)
                anchors.append(xyz)

        if not anchors:
            return None
        self.stats.frames_with_anchor += 1
        return np.asarray(anchors, dtype=np.float32)

    def record_gt_error(self, anchors, gt) -> None:
        """Audit hook: distance from the *first* anchor to the ground-truth target.

        Only OpenDrawer exposes the labels this needs. Compared against the 4.3 cm median the
        cache was audited at, it is the check that the live pipeline reproduces the trained
        one rather than merely running without error.
        """
        if anchors is None or gt is None or len(anchors) == 0:
            return
        self.stats.gt_errors.append(float(np.linalg.norm(np.asarray(anchors[0]) - np.asarray(gt))))

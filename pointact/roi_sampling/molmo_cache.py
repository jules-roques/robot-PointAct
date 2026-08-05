"""On-disk format for the MolmoPoint anchor cache.

One record per stored frame, keyed ``"{episode}-{frame}"`` exactly like the point LMDB, so
the dataloader looks anchors up with the key it already has. The writer is
``data_prep/roi_sampling/build_molmo_cache.py``; the reader is
``PointActLeRobotDataset3D.load_molmo_anchors``. Both import the layout from here so the
format can only change in one place.

Layout, float32::

    [n_anchors] + MAX_ANCHORS x [x, y, z, query_id, n_support, *per-view uv, agree]

``n_anchors`` may be 0, meaning the pointer found nothing usable for this frame; the
dataloader then falls back to a uniform draw, which is exactly the baseline. Unused slots
are zero-filled, so every record is the same size and can be read with a single
``np.frombuffer``.

Why store more than the anchor:

* ``query_id`` indexes the task's pointing queries (0 = the manipulated object, 1 = the
  destination). Keeping every query in one cache makes "object only" versus "object +
  destination" a dataloader knob rather than a second, hour-long build.
* ``n_support`` is how many cloud points landed in the pixel window — a per-frame quality
  signal, and what breaks ties when the two camera views disagree.
* the raw per-view pixels and the ``agree`` flag are kept for auditing and for the
  visualisation, which draws the model's actual 2D point on the video frame.

Pointing runs at the replan cadence, not every frame, so consecutive frames share an
anchor: the builder writes the held value to every frame rather than making the reader
reconstruct the stride.
"""

from __future__ import annotations

import numpy as np

#: Slots per record. 2 covers every task here (OpenDrawer and TurnOnMicrowave use 1,
#: PickPlaceCounterToStove uses 2: the object and the pan it goes into).
MAX_ANCHORS = 2

#: Cameras a point can be found in, in the order their uv pairs are stored. The wrist camera
#: was added after the first caches were written, so records exist in two lengths and the
#: layout is derived from the record rather than assumed -- see :func:`layout_for`.
VIEW_ORDER = ("left", "right", "wrist")

#: xyz(3) + query_id + n_support + 2 per view + agree.
ANCHOR_STRIDE = 5 + 2 * len(VIEW_ORDER) + 1

RECORD_FLOATS = 1 + MAX_ANCHORS * ANCHOR_STRIDE
RECORD_DTYPE = np.float32


def layout_for(n_floats: int) -> tuple[tuple[str, ...], int]:
    """(views, stride) for a record of this length.

    Older caches were written with two views and are still worth reading: their left/right
    pixels cost 0.7 s of an 8B model each. Anything else is a corrupt record.
    """
    stride, rem = divmod(n_floats - 1, MAX_ANCHORS)
    n_views, vrem = divmod(stride - 6, 2)
    if rem or vrem or not 0 < n_views <= len(VIEW_ORDER):
        raise ValueError(f"unrecognised molmo record length {n_floats}")
    return VIEW_ORDER[:n_views], stride


def empty_record() -> np.ndarray:
    """A record meaning "no usable detection for this frame"."""
    return np.zeros(RECORD_FLOATS, dtype=RECORD_DTYPE)


def encode_record(anchors: list[dict]) -> np.ndarray:
    """Pack up to ``MAX_ANCHORS`` anchor dicts into one flat float32 record.

    Args:
        anchors: dicts with keys ``xyz`` (3,), ``query_id``, ``n_support``, and optionally
            ``<view>_uv`` for each view in :data:`VIEW_ORDER` (each (2,) or None) and
            ``agree`` (bool).
    """
    rec = empty_record()
    keep = anchors[:MAX_ANCHORS]
    rec[0] = len(keep)
    for i, a in enumerate(keep):
        off = 1 + i * ANCHOR_STRIDE
        rec[off : off + 3] = np.asarray(a["xyz"], dtype=np.float32).reshape(3)
        rec[off + 3] = float(a["query_id"])
        rec[off + 4] = float(a.get("n_support", 0))
        for j, view in enumerate(VIEW_ORDER):
            uv = a.get(f"{view}_uv")
            # NaN, not 0, for "this view did not point": 0 is a legal pixel coordinate and
            # the visualisation would happily draw a marker in the corner for every miss.
            rec[off + 5 + 2 * j : off + 7 + 2 * j] = (
                np.asarray(uv, dtype=np.float32).reshape(2) if uv is not None else np.nan
            )
        rec[off + ANCHOR_STRIDE - 1] = float(bool(a.get("agree", False)))
    return rec


def decode_anchors(rec: np.ndarray, query_ids: tuple[int, ...] | None = None) -> np.ndarray | None:
    """Return the (K, 3) anchor positions in ``rec``, or None if there are none usable.

    Args:
        rec: a record as written by :func:`encode_record`.
        query_ids: keep only these query indices; None keeps all of them.
    """
    if rec is None:
        return None
    try:
        _views, stride = layout_for(len(rec))
    except ValueError:
        return None
    n = int(rec[0])
    if n <= 0:
        return None
    out = []
    for i in range(min(n, MAX_ANCHORS)):
        off = 1 + i * stride
        if query_ids is not None and int(rec[off + 3]) not in query_ids:
            continue
        xyz = rec[off : off + 3]
        if np.isfinite(xyz).all():
            out.append(xyz)
    if not out:
        return None
    return np.asarray(out, dtype=np.float64)


def decode_pixels(rec: np.ndarray) -> list[dict]:
    """Per-anchor pixel detections, for the visualisation and for auditing.

    Views absent from an older record simply come back as None, so callers written against
    three views read two-view caches without special-casing.
    """
    if rec is None:
        return []
    try:
        views, stride = layout_for(len(rec))
    except ValueError:
        return []
    out = []
    for i in range(min(int(rec[0]), MAX_ANCHORS)):
        off = 1 + i * stride
        det = {
            "query_id": int(rec[off + 3]),
            "n_support": int(rec[off + 4]),
            "agree": bool(rec[off + stride - 1]),
            "xyz": rec[off : off + 3].astype(np.float64),
        }
        for j, view in enumerate(VIEW_ORDER):
            v = rec[off + 5 + 2 * j : off + 7 + 2 * j] if view in views else None
            det[f"{view}_uv"] = (v.astype(np.float64)
                                 if v is not None and np.isfinite(v).all() else None)
        out.append(det)
    return out

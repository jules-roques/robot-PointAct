"""On-disk format for the MolmoPoint anchor cache.

One record per stored frame, keyed ``"{episode}-{frame}"`` exactly like the point LMDB, so
the dataloader looks anchors up with the key it already has. The writer is
``data_prep/roi_sampling/build_molmo_cache.py``; the reader is
``PointActLeRobotDataset3D.load_molmo_anchors``. Both import the layout from here so the
format can only change in one place.

Layout, float32::

    [n_anchors] + MAX_ANCHORS x [x, y, z, query_id, n_support, lu, lv, ru, rv, agree]

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

#: Floats per anchor slot: xyz(3), query_id, n_support, left_uv(2), right_uv(2), agree.
ANCHOR_STRIDE = 10

RECORD_FLOATS = 1 + MAX_ANCHORS * ANCHOR_STRIDE
RECORD_DTYPE = np.float32


def empty_record() -> np.ndarray:
    """A record meaning "no usable detection for this frame"."""
    return np.zeros(RECORD_FLOATS, dtype=RECORD_DTYPE)


def encode_record(anchors: list[dict]) -> np.ndarray:
    """Pack up to ``MAX_ANCHORS`` anchor dicts into one flat float32 record.

    Args:
        anchors: dicts with keys ``xyz`` (3,), ``query_id``, ``n_support``, and optionally
            ``left_uv``, ``right_uv`` (each (2,) or None) and ``agree`` (bool).
    """
    rec = empty_record()
    keep = anchors[:MAX_ANCHORS]
    rec[0] = len(keep)
    for i, a in enumerate(keep):
        off = 1 + i * ANCHOR_STRIDE
        rec[off : off + 3] = np.asarray(a["xyz"], dtype=np.float32).reshape(3)
        rec[off + 3] = float(a["query_id"])
        rec[off + 4] = float(a.get("n_support", 0))
        for j, key in enumerate(("left_uv", "right_uv")):
            uv = a.get(key)
            # NaN, not 0, for "this view did not point": 0 is a legal pixel coordinate and
            # the visualisation would happily draw a marker in the corner for every miss.
            rec[off + 5 + 2 * j : off + 7 + 2 * j] = (
                np.asarray(uv, dtype=np.float32).reshape(2) if uv is not None else np.nan
            )
        rec[off + 9] = float(bool(a.get("agree", False)))
    return rec


def decode_anchors(rec: np.ndarray, query_ids: tuple[int, ...] | None = None) -> np.ndarray | None:
    """Return the (K, 3) anchor positions in ``rec``, or None if there are none usable.

    Args:
        rec: a record as written by :func:`encode_record`.
        query_ids: keep only these query indices; None keeps all of them.
    """
    if rec is None or len(rec) < RECORD_FLOATS:
        return None
    n = int(rec[0])
    if n <= 0:
        return None
    out = []
    for i in range(min(n, MAX_ANCHORS)):
        off = 1 + i * ANCHOR_STRIDE
        if query_ids is not None and int(rec[off + 3]) not in query_ids:
            continue
        xyz = rec[off : off + 3]
        if np.isfinite(xyz).all():
            out.append(xyz)
    if not out:
        return None
    return np.asarray(out, dtype=np.float64)


def decode_pixels(rec: np.ndarray) -> list[dict]:
    """Per-anchor pixel detections, for the visualisation and for auditing."""
    if rec is None or len(rec) < RECORD_FLOATS:
        return []
    out = []
    for i in range(min(int(rec[0]), MAX_ANCHORS)):
        off = 1 + i * ANCHOR_STRIDE
        def uv(j):
            v = rec[off + 5 + 2 * j : off + 7 + 2 * j]
            return None if not np.isfinite(v).all() else v.astype(np.float64)
        out.append({
            "query_id": int(rec[off + 3]),
            "n_support": int(rec[off + 4]),
            "left_uv": uv(0),
            "right_uv": uv(1),
            "agree": bool(rec[off + 9]),
            "xyz": rec[off : off + 3].astype(np.float64),
        })
    return out

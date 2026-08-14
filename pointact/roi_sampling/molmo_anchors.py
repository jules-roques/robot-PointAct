"""What the cache builder and the live evaluator must agree on, byte for byte.

The MolmoPoint arm exists in two places: ``data_prep/roi_sampling/build_molmo_cache.py``
writes anchors for training, and ``pointact/roi_sampling/live_anchor.py`` produces them
during a rollout. If those two disagree about *anything* -- the wording of a prompt, which
views are searched, how two views are combined -- the policy is evaluated on a different
signal than it was trained on, and the resulting success rate is wrong in a way that looks
entirely plausible.

That failure has already happened once on this arm, from the other direction: eval had no
molmo branch at all and silently fell back to uniform sampling (fixed in ef99441). So the
shared parts live here, imported by both sides, rather than being written twice and kept in
step by hand. The same argument already applies to the record layout in ``molmo_cache.py``.

Deliberately dependency-free beyond numpy: the live side runs inside ``envs/robocasa365``,
which has no lmdb, no av and no torch.
"""

from __future__ import annotations

import re

import numpy as np

#: Views Molmo points in. All of them ride in ONE request, so a frame costs one forward
#: whatever the view count. "wrist" is the close-up: at grasp time the target fills far more
#: of its frame than of the agentviews, which is what a small object needs.
VIEWS = ("left", "right", "wrist")

#: The agentviews are rigidly mounted to the robot base and see the scene throughout the
#: episode; the wrist view only sometimes contains the target. They are fused against each
#: other, and the wrist is applied afterwards as a corroborated refinement.
STATIC_VIEWS = ("left", "right")


def opendrawer_queries(instruction: str) -> list[str]:
    """"Open the left drawer." -> point at that drawer's handle.

    The side matters and is the whole reason a pointing model is used here: an
    open-vocabulary box detector finds every drawer and cannot tell which one the
    instruction means.
    """
    m = re.search(r"\b(left|right)\b", instruction, re.I)
    side = f"{m.group(1).lower()} " if m else ""
    return [f"Point to the handle of the {side}drawer."]


def ppcs_queries(instruction: str) -> list[str]:
    """"Pick the apple from the plate and place it in the pan." -> the apple, then the pan.

    Query 0 is the manipulated object and query 1 the destination; ``molmo_anchor_ids``
    selects which become Gaussian centres, so both arms come from this one definition.
    """
    m = re.search(r"pick the (.+?) from the", instruction, re.I)
    obj = m.group(1).strip() if m else "object"
    return [f"Point to the {obj}.", "Point to the pan."]


def tom_queries(instruction: str) -> list[str]:
    return ["Point to the start button on the microwave."]


#: task -> instruction -> pointing queries. Derived from each episode's own instruction
#: rather than hard-coded per task, because PickPlaceCounterToStove varies the object.
#:
#: These strings are part of the trained arm. Rewriting one silently makes every existing
#: checkpoint's cache stale, so a change here means rebuilding all three caches (~11 GPU-h)
#: and re-training -- see the note in runs/tom-molmo-n4096-s0.yaml about why the
#: TurnOnMicrowave query was left alone even though it is the one that struggles.
TASK_QUERIES = {
    "OpenDrawer": opendrawer_queries,
    "PickPlaceCounterToStove": ppcs_queries,
    "TurnOnMicrowave": tom_queries,
}


def queries_for(task: str, instruction: str) -> list[str]:
    """The pointing prompts for one episode of ``task``, given its own instruction."""
    try:
        fn = TASK_QUERIES[task]
    except KeyError:
        raise KeyError(
            f"no pointing queries defined for task {task!r}; known: {sorted(TASK_QUERIES)}"
        ) from None
    return fn(instruction)


def fuse_mean(per_view: dict[str, tuple[np.ndarray, int]]):
    """Mean of every view that produced a 3D point. Returns (xyz | None, n_views).

    This is the whole fusion rule. It deliberately replaces the earlier arrangement --
    average the two agentviews only when they agree within 10 cm, else keep the
    better-supported one, then adopt the wrist only if it corroborates them -- for two
    reasons.

    The first is that those rules threw away real detections. The wrist is the closest camera
    to the target and is often the ONLY view in which MolmoPoint finds a small control at all;
    requiring an agentview to validate it discarded exactly those calls, and on
    TurnOnMicrowave and PickPlaceCounterToStove that is what capped coverage rather than any
    missing geometry.

    The second is that the gates confounded the measurement. This study asks how well a frozen
    pointer localises a target; when three views disagree, that IS the pointer being wrong,
    and a hand-tuned rule that hides the disagreement behind a "pick the better-supported one"
    fallback reports something other than what it set out to measure. Disagreement now shows
    up as a worse anchor, which is the honest reading. Frames with no view at all fall through
    to the sampler's fallback (eef), not to a repaired anchor.
    """
    got = [a for a, _n in per_view.values()]
    if not got:
        return None, 0
    return np.mean(np.stack(got, axis=0), axis=0), len(got)


#: Values ``molmo_view_select`` may take. "per_view" is the shipped arm: every view that
#: lifts becomes its own Gaussian centre. "closest_gt" is the stage-5 upper bound below.
VIEW_SELECT = ("per_view", "closest_gt")


def select_closest(per_view: dict[str, tuple[np.ndarray, int]], gt):
    """The one view whose lift lands nearest the ground truth. Returns (xyz|None, view|None).

    An UPPER BOUND on the pointer, not a deployable rule: choosing among the views by their
    distance to the answer is privileged information, exactly like the oracle arm's anchor.
    What it isolates is how much of MolmoPoint's error is *view-selection* error rather than
    pointing error -- the per-view arm spends budget on every view's hypothesis, including
    the wrong ones, and this says what the policy would get if that spend were perfect.

    It is also the anchor the accuracy table already reports: ``eval_molmo_accuracy`` scores
    the centre nearest the ground truth, so a policy trained on this arm consumes the anchor
    whose within-sigma numbers are already tabulated, instead of a mixture the table never
    described.

    ``gt is None`` (no dump row for this frame) returns None rather than quietly reverting to
    a different rule -- the caller then takes the arm's configured fallback, and counts it.
    """
    if not per_view or gt is None:
        return None, None
    gt = np.asarray(gt, dtype=np.float64).reshape(3)
    view = min(per_view, key=lambda v: float(np.linalg.norm(per_view[v][0] - gt)))
    return per_view[view][0], view


def fuse(per_view: dict[str, tuple[np.ndarray, int]], agree_dist: float):
    """SUPERSEDED by :func:`fuse_mean`; kept to reproduce the earlier caches.

    Combine the agentview anchors for one query. Returns (xyz | None, agreed).

    Agreeing views are averaged; disagreeing ones fall back to the better-supported view,
    since a disagreement means at least one of them lifted through an occluder and the
    midpoint of a right answer and a wrong one is simply a third wrong answer.
    """
    got = [(v, a, n) for v, (a, n) in per_view.items() if v in STATIC_VIEWS]
    if not got:
        return None, False
    if len(got) == 1:
        return got[0][1], False
    (_, a0, n0), (_, a1, n1) = got
    if float(np.linalg.norm(a0 - a1)) <= agree_dist:
        return (a0 + a1) / 2.0, True
    return (a0 if n0 >= n1 else a1), False


def apply_wrist(agentview, per_view, accept_dist: float, require_agentview: bool):
    """Refine the agentview anchor with the wrist one. Returns (xyz, used_wrist).

    The wrist camera is close to the target and therefore the most precise view when it can
    see it -- but for much of an episode it CANNOT. At the start the arm is parked and the
    target is out of frame entirely, and the model will still answer with something. So the
    wrist is treated as a refinement that must be corroborated: it is adopted only when it
    lands within ``accept_dist`` of what the agentviews already agreed on, and a wrist
    anchor with no agentview to check it against is discarded by default.
    """
    w = per_view.get("wrist")
    if w is None:
        return agentview, False
    if agentview is None:
        return (None if require_agentview else w[0]), (not require_agentview)
    if float(np.linalg.norm(w[0] - agentview)) <= accept_dist:
        return w[0], True
    return agentview, False

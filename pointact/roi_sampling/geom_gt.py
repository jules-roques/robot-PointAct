"""Where the target actually is, according to the simulator's own geometry.

Two stage-5 arms are privileged and need this: the oracle sampler centres its density on the
target, and the MolmoPoint best-view arm picks whichever camera's lifted detection lands
closest to it. Both consume the answer twice -- offline when a cache is built, live during a
rollout -- and the two must be the SAME quantity, or the policy is scored on a signal it was
not trained on. That failure has already happened on the molmo arm (eval had no molmo branch
at all, ef99441), so the shared parts live here and are imported by both sides, exactly as
``molmo_anchors.py`` does for the pointing prompts.

Why geometry rather than the rendered segmentation labels the original oracle arm used:
a label centroid is the mean of *visible surface* points, so it moves with the camera, is
biased toward whichever face is lit, and is undefined when the target is occluded -- which
is precisely the frame where "which view is right" matters most. ``geom_xpos`` is
volumetric, view-independent, always defined, and extends to a new task by adding one table
entry instead of teaching the label LUT about a new fixture.

Dependency-free beyond numpy and the standard library: the live side runs inside
``envs/robocasa365``, which has no lmdb, no av and no torch. The MuJoCo-facing helpers take
an already-built env and touch nothing but ``env.env.sim``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

#: Filename of the converted->source episode map, written by
#: ``data_prep/robocasa365_to_lerobot/episode_index_map.py``. Duplicated from there rather
#: than imported, because ``pointact`` must not depend on ``data_prep``.
EPISODE_MAP_NAME = "source_episode_map.json"

#: task -> {output name: how to find the geoms whose mean xpos is that position}.
#:
#: ``geom`` names an exact geom (after the fixture's naming prefix); ``body_contains`` takes
#: every geom hanging off a body of that fixture whose name contains the substring, which is
#: how ``environments._build_geom_label_lut`` assigns the handle/door labels -- so the
#: OpenDrawer entry below reproduces the original oracle arm's target from geometry instead
#: of from rendered labels. ``object`` names a key of ``env.objects`` instead, for tasks
#: whose target is a movable object rather than part of a fixture.
#:
#: Moved here from data_prep/roi_sampling/dump_target_positions.py, which still owns the
#: replay that fills the dump but now imports the table.
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
            "handle": {"body_contains": ["handle"]},
            # A drawer's front panel is a "door" body in RoboCASA's naming, whatever the
            # fixture is called, and the handle hangs off a body whose name contains "door"
            # too -- hence the exclusion. This is the same handle-before-door precedence
            # environments._build_geom_label_lut uses to assign labels 4 and 3.
            "drawer_panel": {"body_contains": ["drawer", "door"], "body_excludes": ["handle"]},
        },
    },
    # Both pointing queries here name a movable object, so there is no fixture to hang the
    # geoms off: `ppcs_queries` asks for the food item (query 0) and "the pan" (query 1), and
    # the task builds them as env.objects["obj"] and env.objects["container"]. Each is the
    # other's distractor -- late in an episode the food ends up *in* the pan, so a
    # "closer to the pan than to the food" reading is only meaningful early on.
    "PickPlaceCounterToStove": {
        "sets": {
            "obj": {"object": "obj"},
            "pan": {"object": "container"},
        },
    },
    # CoffeeSetupMug builds the mug as env.objects["obj"] (kitchen_coffee.PickPlaceCoffee
    # ._get_obj_cfgs, obj_groups="mug"), so it needs no fixture and takes the same shape as
    # PickPlaceCounterToStove. Unverified against a live env: this task's data has not been
    # downloaded yet, and dump_episode refuses a set that matched no geoms, so a wrong entry
    # fails loudly on the first shard rather than producing a plausible wrong position.
    "CoffeeSetupMug": {
        "sets": {
            "obj": {"object": "obj"},
        },
    },
    # NOTE: CloseBlenderLid is deliberately absent. Its target is the lid, which is an
    # auxiliary FIXTURE (env.blender.blender_lid, body "<lid name>_main"), not an entry of
    # env.objects, and the `body_contains` matcher filters on `fixture.name` being a
    # substring of the body name -- which holds only if the blender fixture happens to be
    # named "blender". Add the entry after checking the real names against a built env; a
    # guess here would silently anchor on the blender body instead of its lid.
}

#: task -> geom set the ORACLE arm centres its density on. "The object to grab", except on
#: TurnOnMicrowave, where nothing is grasped and the target is the control being pressed.
ORACLE_TARGET = {
    "OpenDrawer": "handle",
    "TurnOnMicrowave": "start_button",
    "PickPlaceCounterToStove": "obj",
    "CoffeeSetupMug": "obj",
}

#: task -> {MolmoPoint query id: geom set that query is asking for}. Indexes
#: ``molmo_anchors.TASK_QUERIES`` output, so it is the bridge between "Point to the apple."
#: and the apple's real position -- what the best-view arm selects on.
QUERY_TARGETS = {
    "OpenDrawer": {0: "handle"},
    "TurnOnMicrowave": {0: "start_button"},
    "PickPlaceCounterToStove": {0: "obj", 1: "pan"},
    "CoffeeSetupMug": {0: "obj"},
}


def oracle_target_for(task: str) -> str:
    """The geom set the oracle arm centres on, or a loud failure."""
    try:
        return ORACLE_TARGET[task]
    except KeyError:
        raise KeyError(
            f"no oracle target defined for task {task!r}; known: {sorted(ORACLE_TARGET)}. "
            f"Add it to pointact.roi_sampling.geom_gt.ORACLE_TARGET, and the geoms it "
            f"resolves to in TARGET_GEOMS."
        ) from None


def load_episode_map(dataset_dir: Path) -> dict[int, int]:
    """{converted episode -> source episode}. Required, never assumed to be the identity.

    OpenDrawer's 514 source episodes became 496 converted ones under a renumbering, so a
    lookup driven by converted indices against a source-indexed dump would answer with a
    different episode's target -- a plausible number that is simply wrong.
    """
    p = Path(dataset_dir) / "meta" / EPISODE_MAP_NAME
    if not p.exists():
        raise FileNotFoundError(
            f"missing {p}. Build it with:\n"
            f"  python -m data_prep.robocasa365_to_lerobot.episode_index_map "
            f"--source-dir <src>/lerobot --dataset-dir {dataset_dir}"
        )
    return {int(k): int(v)
            for k, v in json.loads(p.read_text())["converted_to_source"].items()}


def load_targets(path: Path, names: list[str],
                 converted_to_source: dict[int, int] | None = None):
    """``lookup(name, ep, frame) -> xyz | None`` over a ``target_positions.npz``.

    ``ep`` is a **converted** episode index, because that is what every cache in this repo is
    keyed by, while the dump is indexed by *source* episode; ``converted_to_source`` bridges
    them. Pass None only when a verified map says the two are the identity.

    An episode dumped with ``--reset-only`` has a single row, because its target does not
    move; that row answers for every frame of the episode. Anything else is indexed per
    frame, and a frame past the end of the dump has no answer rather than a wrong one.

    This is ``eval_molmo_accuracy.load_gt_geom`` moved here so the dataloader, the offline
    selection pass and the accuracy script cannot drift apart.
    """
    z = np.load(path, allow_pickle=True)
    available = json.loads(str(z["geom_sets"]))
    for name in names:
        if name not in available:
            raise KeyError(f"{path} has no geom set '{name}'; it has {available}")

    eps, frames = z["episode"], z["frame"]
    tables: dict[str, dict[tuple[int, int], np.ndarray]] = {name: {} for name in names}
    for name in names:
        for ep, fr, xyz in zip(eps.tolist(), frames.tolist(), z[name]):
            tables[name][(ep, fr)] = xyz
    static = {int(ep): int((eps == ep).sum()) == 1 for ep in np.unique(eps).tolist()}

    def lookup(name: str, ep: int, frame: int):
        src = ep if converted_to_source is None else converted_to_source.get(int(ep))
        if src is None:
            return None
        got = tables[name].get((int(src), 0 if static.get(int(src)) else int(frame)))
        return None if got is None else np.asarray(got, dtype=np.float64)

    return lookup


def world_to_base(points_world: np.ndarray, base_pos, base_quat_xyzw) -> np.ndarray:
    """(N, 3) world points expressed in the robot-base frame the point clouds use."""
    from scipy.spatial.transform import Rotation as R

    rot = R.from_quat(np.asarray(base_quat_xyzw, dtype=np.float64).reshape(4)).as_matrix()
    p = np.asarray(points_world, dtype=np.float64).reshape(-1, 3)
    return (p - np.asarray(base_pos, dtype=np.float64).reshape(1, 3)) @ rot


def resolve_geom_ids(env, fixture, spec: dict) -> list[int]:
    """Geom ids matching one entry of a task's ``sets``.

    ``geom`` is looked up through the fixture's ``naming_prefix`` (that is how RoboCASA's own
    success check finds the button); ``body_contains`` walks every geom and keeps those whose
    body belongs to the fixture and contains any of the substrings, minus any matching
    ``body_excludes``. An empty ``body_contains`` entry means the whole fixture.

    ``object`` takes every geom of one ``env.objects`` entry, by exact body name. The mean
    over *all* of an object's geoms -- collision and visual alike, which sit on top of each
    other -- is its centroid, which is the thing a pointing model is being asked for. Not
    ``body_xpos`` of the root body: that is the MJCF frame origin, which for an asymmetric
    mesh like a pan is not where the object looks like it is.

    Matching body names *exactly* rather than by ``naming_prefix``, because RoboCASA's
    ``try_to_place_in`` registers the receptacle an object starts in as a second object named
    ``<name>_container``: on PickPlaceCounterToStove the food is ``obj`` and the plate under
    it is ``obj_container``, whose bodies also start with ``obj_``. A prefix match silently
    returns the food *and* the plate, and their midpoint is a ground truth that is wrong by a
    few centimetres -- the scale the whole study is measuring at.
    """
    model = env.env.sim.model
    if "object" in spec:
        key = spec["object"]
        objects = getattr(env.env, "objects", {}) or {}
        if key not in objects:
            raise RuntimeError(f"env.objects has no '{key}'; it has {sorted(objects)}")
        obj = objects[key]
        # `.bodies` is the authoritative list but raises on some object types whose `_bodies`
        # carries an unnamed entry (the pan does). The root body alone is right for a
        # single-body object, and the caller refuses a set that matched nothing, so a
        # narrower match fails loudly instead of silently answering with the wrong position.
        try:
            want = {obj.root_body, *(obj.bodies or [])}
        except (TypeError, AttributeError):
            want = {obj.root_body}
        return [gid for gid in range(int(model.ngeom))
                if (model.body_id2name(int(model.geom_bodyid[gid])) or "") in want]

    if "geom" in spec:
        name = f"{fixture.naming_prefix}{spec['geom']}"
        try:
            return [int(model.geom_name2id(name))]
        except Exception:
            return []

    want = spec["body_contains"]
    if isinstance(want, str):
        want = [want]
    drop = spec.get("body_excludes", [])
    fixture_name = fixture.name
    out = []
    for gid in range(int(model.ngeom)):
        body = model.body_id2name(int(model.geom_bodyid[gid])) or ""
        if fixture_name not in body:
            continue
        if any(x in body for x in drop):
            continue
        if any(w in body for w in want):
            out.append(gid)
    return out


class LiveGeomTargets:
    """Target positions read straight out of a running env, in the robot-base frame.

    The live counterpart of a ``target_positions.npz``: same table, same geom resolution,
    same ``world_to_base``, so an offline dump and a rollout answer identically. Geom ids are
    resolved once per ``reset()`` because RoboCasa re-randomises the scene on every trial and
    the ids do not survive it.

    Costs one ``sim.data.geom_xpos`` slice per call -- no rendering, no segmentation pass,
    nothing that would make an eval slower or change what the policy sees.
    """

    def __init__(self, task: str, wanted: list[str] | None = None):
        try:
            entry = TARGET_GEOMS[task]
        except KeyError:
            raise KeyError(
                f"no TARGET_GEOMS entry for task {task!r}; known: {sorted(TARGET_GEOMS)}"
            ) from None
        self.task = task
        self.sets = entry["sets"]
        self.fixture_attr = entry.get("fixture_attr")
        if wanted is not None:
            missing = [n for n in wanted if n not in self.sets]
            if missing:
                raise KeyError(f"task {task!r} has no geom set(s) {missing}; "
                               f"it has {sorted(self.sets)}")
            self.sets = {n: self.sets[n] for n in wanted}
        self._gids: dict[str, list[int]] = {}

    def reset(self, env) -> None:
        """Re-resolve geom ids against the freshly randomised scene. Call after env.reset()."""
        fixture = None
        if self.fixture_attr is not None:
            fixture = getattr(env.env, self.fixture_attr, None)
            if fixture is None:
                raise RuntimeError(
                    f"env has no attribute '{self.fixture_attr}'; TARGET_GEOMS is out of step "
                    f"with the task")
        self._gids = {name: resolve_geom_ids(env, fixture, spec)
                      for name, spec in self.sets.items()}
        missing = [n for n, ids in self._gids.items() if not ids]
        if missing:
            raise RuntimeError(f"no geoms matched {missing} on task {self.task}")

    def __call__(self, env, obs: dict) -> dict[str, np.ndarray]:
        """{set name: xyz} for the current simulator state, in the robot-base frame."""
        if not self._gids:
            raise RuntimeError("LiveGeomTargets.reset(env) must be called after every "
                               "env.reset(): geom ids do not survive a scene re-randomisation")
        base_pos = np.asarray(obs["state.base_position"], dtype=np.float64).reshape(3)
        base_quat = np.asarray(obs["state.base_rotation"], dtype=np.float64).reshape(4)
        out = {}
        for name, ids in self._gids.items():
            xw = env.env.sim.data.geom_xpos[ids].reshape(-1, 3)
            out[name] = world_to_base(xw, base_pos, base_quat).mean(axis=0).astype(np.float32)
        return out

"""List what a task's scene actually contains, so a TARGET_GEOMS entry is read, not guessed.

Adding a task to :data:`pointact.roi_sampling.geom_gt.TARGET_GEOMS` means naming the geoms
whose mean position IS the target. Getting that wrong does not fail -- it answers with a
position a few centimetres off, which is the scale this whole study measures at. That has
already happened once: on PickPlaceCounterToStove a prefix match returned the food *and* the
plate underneath it (``obj`` and ``obj_container``), and the "ground truth" was their
midpoint (fixed in 2483a0e).

So: build the task's env, print the movable objects, the fixture attributes, and the bodies
each candidate resolves to, then write the entry from that.

    python -m data_prep.roi_sampling.list_target_geoms --task CoffeeSetupMug
    python -m data_prep.roi_sampling.list_target_geoms --task CloseBlenderLid --grep lid

Runs in the simulator env and needs a GL context (MUJOCO_GL=egl on a compute node).
"""

from __future__ import annotations

import argparse

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", required=True)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--grep", default=None,
                    help="Only show bodies whose name contains this substring.")
    ap.add_argument("--max-bodies", type=int, default=60)
    args = ap.parse_args()

    from pointact.robot_envs.robocasa365_utils.environments import RoboCasa365Env
    from pointact.roi_sampling.geom_gt import TARGET_GEOMS, resolve_geom_ids, world_to_base

    env = RoboCasa365Env(env_name=args.task, seed=args.seed, image_resolution=64,
                         use_depth=False, use_point_cloud=False, enable_render=True)
    obs, _ = env.reset()
    model = env.env.sim.model
    base_pos = np.asarray(obs["state.base_position"], dtype=np.float64).reshape(3)
    base_quat = np.asarray(obs["state.base_rotation"], dtype=np.float64).reshape(4)

    print(f"task={args.task} seed={args.seed}  instruction={obs.get('task')!r}")

    objects = getattr(env.env, "objects", {}) or {}
    print(f"\nenv.objects ({len(objects)}) -- candidates for a {{'object': <key>}} spec:")
    for key, obj in sorted(objects.items()):
        try:
            bodies = sorted({obj.root_body, *(obj.bodies or [])})
        except (TypeError, AttributeError):
            bodies = [obj.root_body]
        print(f"  {key:20s} root_body={obj.root_body!r} bodies={bodies}")

    # Fixture attributes a `fixture_attr` spec could name. Probed the way
    # environments._target_fixture_names does, plus whatever else the task set on itself.
    print("\nfixture-ish attributes on the env:")
    for name in sorted(vars(env.env)):
        val = getattr(env.env, name, None)
        if hasattr(val, "naming_prefix") and hasattr(val, "name"):
            extra = [a for a in sorted(vars(val)) if hasattr(getattr(val, a, None), "name")
                     and hasattr(getattr(val, a, None), "naming_prefix")]
            print(f"  env.{name:20s} name={val.name!r} prefix={val.naming_prefix!r}"
                  + (f" sub-fixtures={extra}" if extra else ""))

    print(f"\nbodies with geoms{f' matching {args.grep!r}' if args.grep else ''}:")
    per_body: dict[str, int] = {}
    for gid in range(int(model.ngeom)):
        body = model.body_id2name(int(model.geom_bodyid[gid])) or ""
        if args.grep and args.grep.lower() not in body.lower():
            continue
        per_body[body] = per_body.get(body, 0) + 1
    for i, (body, n) in enumerate(sorted(per_body.items())):
        if i >= args.max_bodies:
            print(f"  ... and {len(per_body) - args.max_bodies} more (raise --max-bodies)")
            break
        print(f"  {body:50s} {n} geom(s)")

    # If the task already has an entry, resolve it and show where it lands -- the check that
    # the spec means what it was intended to mean.
    entry = TARGET_GEOMS.get(args.task)
    if entry is None:
        print(f"\nno TARGET_GEOMS entry for {args.task} yet.")
        return
    fixture = None
    if entry.get("fixture_attr"):
        fixture = getattr(env.env, entry["fixture_attr"], None)
    print(f"\nresolving the existing TARGET_GEOMS entry for {args.task}:")
    for set_name, spec in entry["sets"].items():
        ids = resolve_geom_ids(env, fixture, spec)
        if not ids:
            print(f"  {set_name:16s} MATCHED NOTHING -- spec={spec}")
            continue
        xw = env.env.sim.data.geom_xpos[ids].reshape(-1, 3)
        xyz = world_to_base(xw, base_pos, base_quat).mean(axis=0)
        bodies = sorted({model.body_id2name(int(model.geom_bodyid[g])) or "" for g in ids})
        print(f"  {set_name:16s} {len(ids):3d} geom(s) -> base-frame "
              f"[{xyz[0]:.3f} {xyz[1]:.3f} {xyz[2]:.3f}]  bodies={bodies}")


if __name__ == "__main__":
    main()

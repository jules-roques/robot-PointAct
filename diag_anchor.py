"""Why does the eval client never produce a ground-truth sampling anchor?

Recreates exactly what run_robocasa365_client.py does (env with use_segmentation=True, reset,
step) and reports which observation keys exist and what ground_truth_anchor() returns.
"""
import sys

import numpy as np

sys.path.insert(0, "experiments/13_robocasa365")

from pointact.robot_envs.robocasa365_utils.environments import (
    POINT_LABEL_NAMES,
    RoboCasa365Env,
)

env = RoboCasa365Env(
    env_name="OpenDrawer",
    split="target",
    seed=7,
    image_resolution=256,
    use_depth=True,
    use_point_cloud=True,
    use_segmentation=True,
)
obs, _ = env.reset()

print("has .drawer attr on inner env:", hasattr(env.env, "drawer"))
label_keys = sorted(k for k in obs if "point_labels" in k)
print("point_label keys:", label_keys)
print("point keys:", sorted(k for k in obs if k.startswith("observation.points")))

if label_keys:
    for k in label_keys:
        a = np.asarray(obs[k])
        print(f"  {k}: shape={a.shape} dtype={a.dtype} "
              f"counts={{{', '.join(f'{POINT_LABEL_NAMES[i]}:{int((a==i).sum())}' for i in range(5))}}}")

from run_robocasa365_client import ground_truth_anchor  # noqa: E402

anchor = ground_truth_anchor(obs)
print("\nground_truth_anchor ->", anchor)

# and a few steps in, since the handle may be occluded at t=0
noop = np.zeros(13, dtype=np.float32)
noop[6] = 1.0
for i in range(15):
    obs, _, done, _ = env.step(noop)
    if done:
        break
print("after 15 steps ->", ground_truth_anchor(obs))
env.close()

"""Regression test for patch 0001, bug (1): categories with a registry key but no models.

Draws 200 objects. The original crash hit within 7 draws, so 200 is a real test rather than a
formality. Exits non-zero on the first `ValueError: Probabilities contain NaN`.

    uv run --project envs/robocasa365 python envs/robocasa365/patches/test_empty_category_guard.py
"""
import sys

import numpy as np
from robocasa.models.objects.kitchen_object_utils import (
    OBJ_CATEGORIES,
    sample_kitchen_object,
)

REGISTRIES = ["aigen", "lightwheel", "objaverse"]
DRAWS = 200

zero_asset = [
    cat for cat, d in OBJ_CATEGORIES.items()
    if sum(len(d[r].mjcf_paths) if r in d else 0 for r in REGISTRIES) == 0
]
print(f"{len(zero_asset)} of {len(OBJ_CATEGORIES)} categories have zero assets in every "
      f"registry; the patch must simply never choose them")

rng = np.random.default_rng(7)
for i in range(DRAWS):
    try:
        sample_kitchen_object(
            groups="all", graspable=True, rng=rng, obj_registries=("objaverse", "aigen")
        )
    except Exception as exc:  # noqa: BLE001 -- any failure here is a test failure
        print(f"FAILED at draw {i}: {type(exc).__name__}: {exc}")
        sys.exit(1)

print(f"{DRAWS}/{DRAWS} object draws succeeded with no NaN")

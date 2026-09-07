"""Regression test for patch 0001, bug (2): categories emptied by the split filter.

Sweeps all 12 combinations of `split` x registry tuple, 150 draws each. The `split="target"`
rows are the ones that crashed before the patch: `split_th = max(len - 5, ceil(len / 2))` sends
a single-model category to `reg_choices[1:]`, i.e. nothing. Exits non-zero if any combination
still fails.

    uv run --project envs/robocasa365 python envs/robocasa365/patches/test_split_guard.py
"""
import itertools
import sys

import numpy as np
from robocasa.models.objects.kitchen_object_utils import sample_kitchen_object

DRAWS = 150
COMBOS = list(itertools.product(
    [None, "pretrain", "target"],
    [("objaverse", "aigen"), ("objaverse",), ("aigen",), ("objaverse", "aigen", "lightwheel")],
))

fails = 0
for split, registries in COMBOS:
    rng = np.random.default_rng(7)
    ok = 0
    try:
        for _ in range(DRAWS):
            sample_kitchen_object(
                groups="all", graspable=True, rng=rng,
                obj_registries=registries, split=split,
            )
            ok += 1
        print(f"  split={str(split):<9} regs={str(registries):<40} {ok}/{DRAWS} OK")
    except Exception as exc:  # noqa: BLE001 -- any failure here is a test failure
        fails += 1
        print(f"  split={str(split):<9} regs={str(registries):<40} FAILED at {ok}: "
              f"{type(exc).__name__}: {str(exc)[:70]}")

if fails:
    print(f"\n{fails} of {len(COMBOS)} combinations still fail")
    sys.exit(1)
print(f"\nall {len(COMBOS)} combinations passed")

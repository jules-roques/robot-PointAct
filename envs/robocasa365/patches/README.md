# Required RoboCasa patches

The submodule pointer in `envs/robocasa365/robocasa` is pinned to the RoboCasa365 release
(`0c81ff9`, `v0.2-2-g0c81ff9`), and the parent README says the working copy must not drift from
it. There is one deliberate exception, recorded here: without it, evaluation runs die at a
random trial.

## What breaks without the patch

Both bugs end the same way — `ValueError: Probabilities contain NaN` raised out of
`sample_kitchen_object_helper`, from

```python
p = np.array([len(choices[reg]) for reg in obj_registries]) / sum(...)
```

when the denominator is `0`. It fires at whatever trial first happens to draw an empty
category, so a 500-rollout eval can get 400 trials in and then lose the whole job. It is not
reproducible from the run arguments, which is what makes it expensive rather than merely
annoying.

The two independent ways the denominator reaches zero:

1. **A category with a registry key but no models.** The upstream guard tests
   `reg in OBJ_CATEGORIES[cat]` — whether the registry *key* exists — not whether it holds any
   models. On this install `aigen_objs` extracted as a bare directory skeleton (152 categories,
   0 `model.xml`, 20 MB against objaverse's 1.4 GB), so 60 categories pass the guard with an
   empty `mjcf_paths`. The patch tests for actual models, which is what the upstream comment
   above the guard already says it intends.

2. **A category emptied by the split filter.** `split_th = max(len - 5, ceil(len / 2))` sends a
   single-model category to `reg_choices[1:]` — nothing — under `split="target"`. This one
   cannot be fixed by filtering the category list up front, because the category is emptied
   *after* it passes every up-front guard. The patch draws a category, checks the built
   candidate list is non-empty, and re-draws if not; that covers every way a category can come
   out empty instead of enumerating them. It also prints each skipped category once, since a
   silently-skipped category is how a broken asset install stays invisible.

Neither is specific to one cluster: (1) depends on how the asset packs extracted, (2) is
reachable on any install that uses `split="target"`, which every eval here does.

## Apply

```bash
envs/robocasa365/patches/apply.sh
```

Idempotent — it detects an already-patched checkout and does nothing. Run it after any
`git submodule update`, after a fresh clone, and in any new worktree: a worktree gets its own
submodule working copy, so the patch does not come along with it.

## Verify

```bash
uv run --project envs/robocasa365 python envs/robocasa365/patches/test_empty_category_guard.py
uv run --project envs/robocasa365 python envs/robocasa365/patches/test_split_guard.py
```

The first draws 200 objects (the original crash hit within 7, so 200 is a real test); the
second sweeps all 12 combinations of `split` x registry tuple. `apply.sh --check` runs both.

## Upstreaming

Both are genuine upstream bugs and neither fix is specific to this fork, so they are candidates
for a PR against RoboCasa. Until that lands, this directory is the record — if it is lost, the
crash comes back with no trace of why.

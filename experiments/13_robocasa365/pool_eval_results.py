"""Pool per-seed eval results for every arm under an experiment root, as JSON.

Complements `summarize_stage_a.py` rather than replacing it. That script renders the stage-A
point-count x sampling table for the gate mail, and to do so it parses run *names* with a
strict regex over a fixed task list (`od`, `ppcs`, `tom`). This one keys on the arm directory
itself, so it works for any naming scheme -- the stage-5 `s5-blender-oracle-n8192-s0` arms and
the stage-6 arms included -- and emits JSON for a caller to format.

It also carries two integrity guards, both of which have caught real problems:

  * **duplicate seed** -- a smoke run's result file matching the glob, which would inflate `n`
    and double-count its successes.
  * **mixed `env_name` inside one arm** -- read from what the eval actually recorded, not
    inferred from the directory name. This is the stronger form of the cross-task pooling bug
    `summarize_stage_a.py`'s `--task` filter guards against: it catches an arm whose results
    came from the wrong env even when the directory is named correctly.

Both raise rather than warn. A silently wrong `n` is the one failure mode that survives review,
because the number still looks plausible.

    python experiments/13_robocasa365/pool_eval_results.py \
        $SCRATCH/PointAct_exprs/robocasa365/stage5 --step 30000
"""

import argparse
import glob
import json
import math
import os


def wilson(successes: int, trials: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval, in percent.

    Not the Wald interval: at n=100 with rates near 0 or 1 -- CloseBlenderLid sits at 1.4% --
    Wald under-covers badly and can put a bound outside [0, 100].
    """
    if trials == 0:
        return (0.0, 0.0)
    p = successes / trials
    denom = 1 + z * z / trials
    centre = (p + z * z / (2 * trials)) / denom
    half = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / denom
    return (100 * (centre - half), 100 * (centre + half))


def pool(root: str, step: str) -> dict:
    out = {}
    for ckpt_dir in sorted(glob.glob(os.path.join(root, "*", "results", "checkpoint-" + step))):
        arm = ckpt_dir.split(os.sep)[-3]
        seeds, envs = {}, set()
        for path in sorted(glob.glob(os.path.join(ckpt_dir, "per_trial_seed*_n*.json"))):
            record = json.loads(open(path).read())
            seed = record["seed"]
            if seed in seeds:
                raise SystemExit(f"duplicate seed {seed} in {ckpt_dir} ({os.path.basename(path)})")
            seeds[seed] = (record["successes"], record["num_trials"])
            envs.add(record["env_name"])
        if not seeds:
            continue
        if len(envs) != 1:
            raise SystemExit(f"mixed envs {sorted(envs)} in {ckpt_dir}")
        successes = sum(v[0] for v in seeds.values())
        trials = sum(v[1] for v in seeds.values())
        low, high = wilson(successes, trials)
        out[arm] = dict(
            env=envs.pop(),
            seeds=sorted(seeds),
            k=successes,
            n=trials,
            rate=round(100 * successes / trials, 1),
            ci=[round(low, 1), round(high, 1)],
            per_seed={s: round(100 * v[0] / v[1], 1) for s, v in sorted(seeds.items())},
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root", help="Experiment root holding one directory per arm.")
    parser.add_argument("--step", default="30000", help="Checkpoint step to pool (default 30000).")
    args = parser.parse_args()
    print(json.dumps(pool(args.root, args.step), indent=1))


if __name__ == "__main__":
    main()

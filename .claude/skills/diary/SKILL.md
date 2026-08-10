---
name: diary
description: Orient at the start of a session — reconcile the Google Doc work diary's next steps against the live Slurm queue, recent job history, and the results Sheet, and report what is running, done, blocked, or drifting. Use when the user types /diary or asks "where am I", "what's running", "what's next".
---

# /diary

Turn the work diary from a document into a status board. The mechanical data gathering is
a script; the reconciliation — deciding which diary item a job corresponds to — is your
job, because job names and prose next-steps never match literally.

## 1. Resolve the doc

The diary Doc id lives in `.claude/skills/diary/config.json`. If the user passed a URL as
an argument, extract the id (`/document/d/<id>/edit`) and write it into that file before
continuing, so it is only ever asked for once. If there is no id and none was passed, run
the rest anyway and say the diary leg is unconfigured — the cluster and Sheet halves are
still worth having.

## 2. Fetch the doc — connector first

Try `mcp__claude_ai_Google_Drive__read_file_content` with the id. **The diary is
anti-chronological: the newest entry is at the top, so the first page is the part that
matters.** Read enough to cover the current next-steps list; don't pull the whole history.

If the connector returns `insufficient authentication scopes` (its known failure state —
it needs re-authorizing in claude.ai → Connectors), fall back to the script's Docs-API
leg, which uses the gspread service account. If both fail, report which and continue.

## 3. Gather the cluster and Sheet state

```
uv run --quiet --no-project --with 'gspread>=6,<7' --with google-api-python-client \
    --with google-auth python .claude/skills/diary/collect.py
```

`--no-project` is required — without it uv resolves the repo's dependencies and dies on
`open3d`, which has no cp312 wheel. Useful flags: `--days N` (sacct window, default 3),
`--no-google` (cluster only, and the only mode that works from a compute node),
`--doc/--sheet/--tab`, `--max-rows`, `--max-chars`.

**Google legs need the IDRIS proxy, so run this from a Jean Zay login node**, never inside
a batch job. On a compute node, pass `--no-google`.

## 4. Reconcile, then report

Match each next-step in the diary against the evidence. Job names encode the experiment
(`train-od-anchor-v5mm`, `eval-od-uniform-n8192-v5mm-s0` → arm `anchor`/`uniform`, task
`od` = OpenDrawer, `v5mm` = the 5 mm voxel arm), so most items resolve by name. Report a
compact table, most-actionable first:

| Status | Meaning |
|---|---|
| `BLOCKED` | needs a decision or a fix — a `FAILED`/`TIMEOUT` job, or a dependency that will never clear |
| `RUNNING` | in flight; give elapsed and time left |
| `QUEUED` | pending, and say what it is waiting on |
| `DONE` | finished **and** the numbers are in the Sheet |
| `UNRECORDED` | jobs completed but results are not in the Sheet yet — the most common real gap |
| `PLANNED` | in the diary, nothing on the cluster |
| `DRIFT` | running on the cluster but absent from the diary, or contradicted by a result already in the Sheet |

Then a short prose section: what changed since the last diary entry, and the one or two
things worth doing next. Keep it to what is actionable — this is an orientation tool, not
a report.

## Judgement calls

- **A relaunch is not a failure.** The same job name appearing `FAILED` in history and
  `RUNNING` in the queue means it was fixed and resubmitted — that is `RUNNING`, not
  `BLOCKED`. Only flag it if nothing is currently running under that name.
- **Array jobs are one fact.** The script already collapses a 5-task eval array into one
  line; keep it that way in the report.
- **Sheet numbers may lag the cluster by design** — writes are manual. `UNRECORDED` is a
  normal state, not an error, but it is the thing most worth surfacing.
- **Don't launch, cancel, or resubmit anything.** Read-only. Propose the command and let
  the user run it.
- **Don't write to the Sheet from here.** If results need recording, say so and point at
  `scripts/push_results_to_sheets.py`; that script's in-place-edit discipline (never
  insert/delete rows, never recreate tabs — comments and images are cell-anchored) exists
  for a reason.

## Project conventions to apply when reading results

- Eval is **100 trials per checkpoint at 10/20/30/40/50K steps**. A run with a different
  trial count is from the older 50/150 sweep — still valid, differently precise. Say so
  when tabulating it next to new numbers.
- Stage labels are explicit sentences, not letters (`Stage 1: Num points & Train steps`).
  Use the diary's own wording for a stage rather than inventing a short name.

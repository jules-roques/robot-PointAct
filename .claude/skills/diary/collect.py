"""Gather the mechanical half of a /diary status check.

Four independent legs -- cluster queue, recent job history, the results Sheet, and the
work-diary Doc. Each leg is wrapped so that one failing never takes the others down: a
dead leg prints a WARN line and the rest of the digest still comes out. That matters
because the legs have genuinely different failure modes (slurm is always up on a login
node; the Google legs need the IDRIS proxy and are unreachable from inside a batch job).

Output is compact text on stdout, meant to be read by Claude during the /diary skill but
perfectly readable by a human running it directly:

    uv run --with 'gspread>=6,<7' --with google-api-python-client \
        python .claude/skills/diary/collect.py

The Doc leg is best-effort here. The preferred path is the Google Drive MCP connector,
which the skill tries first -- this file's Docs-API fallback only fires when the connector
is unavailable, and needs the Docs API enabled on the GCP project.
"""

import os
import re
import csv
import glob
import json
import argparse
import datetime
import subprocess

DEFAULT_KEY_DIR = os.path.expanduser('~/.config/gspread')
DEFAULT_SHEET = '1-o0Eboh7hDgSIIgSin1l2ymLxuWBrGhk-m41gmtKbdA'
DEFAULT_TAB = 'PointACT'
CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.json')

# Terminal states worth surfacing. COMPLETED is included so a finished run shows up as
# evidence a diary item is done; the rest are all things that need a human decision.
INTERESTING = ('COMPLETED', 'FAILED', 'TIMEOUT', 'OUT_OF_MEMORY', 'NODE_FAIL')


def warn(leg, exc):
    print(f'  WARN  {leg} unavailable: {type(exc).__name__}: {exc}')


def load_config():
    try:
        with open(CONFIG) as f:
            return json.load(f)
    except FileNotFoundError:
        return {}
    except Exception:
        return {}


def resolve_key(path=None):
    if path is not None:
        return path
    keys = sorted(glob.glob(os.path.join(DEFAULT_KEY_DIR, '*.json')))
    if not keys:
        raise RuntimeError(f'no service account key in {DEFAULT_KEY_DIR}')
    return keys[0]


def doc_id(ref):
    """Accept a bare id or a full /document/d/<id>/edit url."""
    if ref and '/' in ref:
        m = re.search(r'/d/([A-Za-z0-9_-]+)', ref)
        if m:
            return m.group(1)
    return ref


# --------------------------------------------------------------------------- slurm

def leg_queue():
    print('## Queue (squeue)')
    try:
        out = subprocess.run(
            ['squeue', '-u', os.environ.get('USER', ''), '-o', '%i|%P|%j|%T|%M|%L|%R'],
            capture_output=True, text=True, timeout=60, check=True).stdout
    except Exception as e:
        return warn('squeue', e)

    rows = [l.split('|') for l in out.strip().splitlines()[1:]]
    if not rows:
        print('  (nothing queued or running)')
        return
    print(f'  {"JOBID":<14} {"PART":<8} {"NAME":<36} {"STATE":<9} {"ELAPSED":>9} {"LEFT":>9}  REASON')
    for r in rows:
        if len(r) < 7:
            continue
        jid, part, name, state, elapsed, left, reason = r[:7]
        reason = '' if reason.startswith('jz') or reason == '(null)' else reason
        print(f'  {jid:<14} {part:<8} {name:<36} {state:<9} {elapsed:>9} {left:>9}  {reason}')


def leg_history(days):
    print(f'\n## Finished in the last {days}d (sacct)')
    try:
        out = subprocess.run(
            ['sacct', '-u', os.environ.get('USER', ''), '-S', f'now-{days}days', '-X',
             '--format=JobID,JobName%40,State,End,Elapsed', '-P'],
            capture_output=True, text=True, timeout=120, check=True).stdout
    except Exception as e:
        return warn('sacct', e)

    seen = []
    for row in csv.DictReader(out.splitlines(), delimiter='|'):
        state = row.get('State', '').split()[0] if row.get('State') else ''
        if state not in INTERESTING:
            continue
        seen.append((row['JobID'], row['JobName'], state, row.get('End', ''), row.get('Elapsed', '')))

    if not seen:
        print(f'  (no terminal jobs in the last {days} days)')
        return

    # Collapse array tasks and repeated launches of the same name+state into one line --
    # a 5-task eval array is one fact, not five, and the raw list buries everything else.
    groups = {}
    for jid, name, state, end, elapsed in seen:
        key = (name, state)
        g = groups.setdefault(key, {'n': 0, 'last': '', 'elapsed': ''})
        g['n'] += 1
        if end > g['last']:
            g['last'], g['elapsed'] = end, elapsed

    print(f'  {"NAME":<40} {"STATE":<12} {"N":>3}  {"LAST END":<20} ELAPSED')
    for (name, state), g in sorted(groups.items(), key=lambda kv: kv[1]['last'], reverse=True):
        print(f'  {name:<40} {state:<12} {g["n"]:>3}  {g["last"]:<20} {g["elapsed"]}')


# ---------------------------------------------------------------------- google legs

def leg_sheet(sheet, tab, max_rows):
    print(f'\n## Results Sheet ({tab})')
    try:
        import gspread
        gc = gspread.service_account(filename=resolve_key())
        ws = gc.open_by_key(sheet).worksheet(tab)
        values = ws.get_all_values()
    except Exception as e:
        return warn('sheet', e)

    rows = [r for r in values if any(c.strip() for c in r)]
    if not rows:
        print('  (tab is empty)')
        return
    print(f'  {len(rows)} non-empty rows; showing up to {max_rows}')
    for r in rows[:max_rows]:
        # The tab does not start at column A and has ragged padding, so trim empty cells
        # from both ends -- column position carries no meaning we need here, only content.
        cells = [c.strip().replace('\n', ' ') for c in r]
        while cells and not cells[-1]:
            cells.pop()
        while cells and not cells[0]:
            cells.pop(0)
        if cells:
            print('  | ' + ' | '.join(cells))
    if len(rows) > max_rows:
        print(f'  ... {len(rows) - max_rows} more rows')


def leg_doc(ref, max_chars):
    print('\n## Work diary (Doc)')
    did = doc_id(ref)
    if not did:
        print('  SKIP  no doc id configured (set doc_id in .claude/skills/diary/config.json,')
        print('        or pass --doc <url>). Preferred path is the Drive MCP connector.')
        return
    try:
        from google.oauth2.service_account import Credentials
        from googleapiclient.discovery import build
        creds = Credentials.from_service_account_file(
            resolve_key(), scopes=['https://www.googleapis.com/auth/documents.readonly'])
        doc = build('docs', 'v1', credentials=creds).documents().get(documentId=did).execute()
    except Exception as e:
        warn('doc', e)
        print('        -> fall back to the Drive MCP connector, or share the doc with the')
        print('           service account and enable the Docs API on the GCP project.')
        return

    out = []
    for el in doc.get('body', {}).get('content', []):
        para = el.get('paragraph')
        if not para:
            continue
        text = ''.join(r.get('textRun', {}).get('content', '')
                       for r in para.get('elements', []))
        if text.strip():
            out.append(text.rstrip())

    body = '\n'.join(out)
    print(f'  title: {doc.get("title", "?")}')
    print(f'  {len(body)} chars; showing the first {max_chars} (newest entries, '
          f'anti-chronological)')
    print('  ---8<---')
    for line in body[:max_chars].splitlines():
        print(f'  {line}')
    if len(body) > max_chars:
        print(f'  ...truncated, {len(body) - max_chars} chars remain')


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--days', type=int, default=3, help='sacct lookback window')
    p.add_argument('--sheet', default=None, help='spreadsheet id or url')
    p.add_argument('--tab', default=None, help='worksheet/tab name')
    p.add_argument('--doc', default=None, help='work-diary Doc id or url')
    p.add_argument('--max-rows', type=int, default=40, help='sheet rows to print')
    p.add_argument('--max-chars', type=int, default=6000, help='doc chars to print')
    p.add_argument('--no-google', action='store_true',
                   help='cluster legs only; skip Sheet and Doc')
    a = p.parse_args()

    cfg = load_config()
    now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
    host = os.uname().nodename
    print(f'# diary status  --  {now}  on {host}')
    if a.no_google:
        pass
    elif not os.environ.get('https_proxy'):
        print('  NOTE  https_proxy is unset. On Jean Zay the Google legs only work from a')
        print('        login node; from a compute node they will time out.')

    leg_queue()
    leg_history(a.days)
    if not a.no_google:
        leg_sheet(a.sheet or cfg.get('sheet', DEFAULT_SHEET),
                  a.tab or cfg.get('tab', DEFAULT_TAB), a.max_rows)
        leg_doc(a.doc or cfg.get('doc_id'), a.max_chars)


if __name__ == '__main__':
    main()

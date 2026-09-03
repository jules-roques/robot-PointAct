"""Write experiment results into a Google Sheet tab, in place.

Edits are surgical: only the cells covered by the values being written are touched.
Tabs are never deleted and recreated, and rows/columns are never inserted or deleted --
the results sheet carries hand-written notes, advisor comments and floating images, all
of which are anchored to cell positions and would be moved or destroyed by structural
edits. Overlay images are invisible to the Sheets API, so damage to them cannot even be
detected after the fact.

Nothing is stored locally: this talks to the Sheets API over HTTPS and the Google copy
is the only copy. Note that edits made here are NOT undoable with ctrl-Z in the browser
(that only reverses your own session). To undo one, use File > Version history, where
these writes show up under the service account's name.

Auth uses a service account key, by default the single .json in ~/.config/gspread/.
The target sheet must be shared with that account's email as an Editor.

Google APIs are reachable from the Jean Zay login node through the IDRIS proxy, which is
already in the environment. Run it there, never from inside a batch job:

    uv run --with 'gspread>=6,<7' --with gspread-formatting \
        python scripts/push_results_to_sheets.py \
        --sheet <url-or-id> --tab PointACT --values results.csv --anchor A5 --header
"""

import os
import csv
import json
import glob
import argparse

import gspread


DEFAULT_KEY_DIR = os.path.expanduser('~/.config/gspread')


def resolve_key(path=None):
    if path is not None:
        return path
    keys = sorted(glob.glob(os.path.join(DEFAULT_KEY_DIR, '*.json')))
    if not keys:
        raise SystemExit(f'no service account key found in {DEFAULT_KEY_DIR}')
    if len(keys) > 1:
        raise SystemExit(f'several keys in {DEFAULT_KEY_DIR}, pass one with --key: {keys}')
    return keys[0]


def sheet_id(sheet):
    # accepts a bare id or a full /spreadsheets/d/<id>/edit url
    if '/' not in sheet:
        return sheet
    parts = sheet.split('/')
    if 'd' in parts:
        return parts[parts.index('d') + 1]
    raise SystemExit(f'could not parse a spreadsheet id out of {sheet!r}')


def load_values(path):
    if path.endswith('.json'):
        with open(path) as f:
            rows = json.load(f)
    else:
        with open(path, newline='') as f:
            rows = [row for row in csv.reader(f)]
    if not rows or not isinstance(rows[0], list):
        raise SystemExit('values file must hold a list of rows')
    return rows


def target_range(anchor, rows):
    first = gspread.utils.a1_to_rowcol(anchor)
    height = len(rows)
    width = max(len(r) for r in rows)
    last = gspread.utils.rowcol_to_a1(first[0] + height - 1, first[1] + width - 1)
    return f'{anchor}:{last}'


def format_header(ws, anchor, width):
    from gspread_formatting import CellFormat, TextFormat, Color, format_cell_ranges

    row0, col0 = gspread.utils.a1_to_rowcol(anchor)
    rng = f'{anchor}:{gspread.utils.rowcol_to_a1(row0, col0 + width - 1)}'
    format_cell_ranges(ws, [(rng, CellFormat(
        backgroundColor=Color(0.267, 0.447, 0.769),
        textFormat=TextFormat(bold=True, foregroundColor=Color(1, 1, 1)),
        horizontalAlignment='CENTER',
    ))])
    print(f'formatted header {rng}')


def main(args):
    gc = gspread.service_account(filename=resolve_key(args.key))
    sh = gc.open_by_key(sheet_id(args.sheet))
    try:
        ws = sh.worksheet(args.tab)
    except gspread.WorksheetNotFound:
        if not args.create_tab:
            raise SystemExit(
                f'tab {args.tab!r} not found in {sh.title!r} '
                f'(existing: {[w.title for w in sh.worksheets()]}); '
                f'pass --create_tab to add it')
        ws = sh.add_worksheet(title=args.tab, rows=200, cols=26)
        print(f'created tab {args.tab!r}')

    rows = load_values(args.values)
    print(f'{sh.title!r} / {ws.title!r}: writing {len(rows)} row(s) at {args.anchor}')

    # USER_ENTERED so numbers, dates and percentages are parsed rather than
    # landing as text that cannot be charted or averaged
    ws.update(values=rows, range_name=args.anchor, value_input_option='USER_ENTERED')
    print(f'wrote {target_range(args.anchor, rows)}')

    if args.header:
        format_header(ws, args.anchor, max(len(r) for r in rows))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sheet', required=True, help='spreadsheet url or id')
    parser.add_argument('--tab', required=True)
    parser.add_argument('--values', required=True, help='.csv or .json list of rows')
    parser.add_argument('--anchor', default='A1', help='top-left cell to write from')
    parser.add_argument('--key', default=None, help='service account json')
    parser.add_argument('--header', action='store_true', default=False,
                        help='style the first written row as a header')
    parser.add_argument('--create_tab', action='store_true', default=False)
    args = parser.parse_args()
    main(args)

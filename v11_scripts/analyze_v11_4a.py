#!/usr/bin/env python3
"""v11-4a analysis: single-factor coarse-scan experiment results.

Each variant invocation (iv4a_<variant>) pairs the variant rows with the
production rows of both sides inside one invocation, so per-round ratios
variant/prod are drift-free. Parses:
  - iv4a_summary.txt: rc + factor lines
  - iv4a_<v>.replay.csv: round,side,order,variant,... scan_us_per_query,...
Prints per-variant tables and writes v11_4a_analysis.out.
Usage: python3 analyze_v11_4a.py [RUN_DIR]
"""
import csv
import re
import sys
from pathlib import Path
from statistics import median

RUN = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent
VARIANTS = ['cluster0', 'dedshape', 'flat', 'flat16']

# v11 P1/P2 re-measured unpruned references (same lists, 5 rounds).
V11_UNPRUNED = {'int_scan': 5604.4, 'int_prep': 29.6, 'ded_scan': 4391.0,
                'ded_prep': 7.1, 'ratio': 1.302}


def load_csv(name):
    path = RUN / f'{name}.replay.csv'
    if not path.exists():
        return None
    rows = []
    with open(path) as f:
        for row in csv.DictReader(f):
            rows.append({
                'round': int(row['round']), 'side': row['side'],
                'order': row['order'], 'variant': row['variant'],
                'scan': float(row['scan_us_per_query']),
                'prep': float(row['prep_us_per_query']),
                'ns': float(row['ns_per_vector']),
                'vectors': int(row['vectors']),
                'lists': int(row['lists']),
                'blocks': int(row['blocks']),
            })
    return rows


def sel(rows, side, order, variant):
    return sorted((r for r in rows if r['side'] == side and
                   r['order'] == order and r['variant'] == variant),
                  key=lambda r: r['round'])


def fmt_variant(v, lines):
    rows = load_csv(f'iv4a_{v}')
    if not rows:
        lines.append(f'--- {v}: MISSING replay csv ---')
        return
    prod = sel(rows, 'int', 'orig', 'full')
    var = sel(rows, 'int', 'orig', v)
    ded = sel(rows, 'ded', 'orig', 'full')
    work = {r['round']: r for r in prod}
    for r in var:
        p = work.get(r['round'])
        if p and (r['vectors'], r['lists'], r['blocks']) != (
                p['vectors'], p['lists'], p['blocks']):
            lines.append(f'  !! work differs at round {r["round"]}: '
                         f'variant vec={r["vectors"]} vs prod {p["vectors"]}')
    lines.append(f'--- {v} (rounds={len(var)}, vectors/round='
                 f'{var[0]["vectors"] if var else "?"}) ---')
    if prod and var:
        ratios = [r['scan'] / p['scan'] for r, p in zip(var, prod)
                  if p['scan'] > 0]
        pairs = ' '.join(f'{r["scan"]:.0f}/{p["scan"]:.0f}'
                         for r, p in zip(var, prod))
        lines.append(
            f'  scan us/q variant={median(r["scan"] for r in var):.1f} '
            f'prod={median(p["scan"] for p in prod):.1f} '
            f'ded={median(d["scan"] for d in ded):.1f}')
        lines.append(
            f'  paired ratio variant/prod med={median(ratios):.3f} '
            f'rounds={" ".join(f"{x:.3f}" for x in ratios)}')
        lines.append(f'  paired raw (variant/prod per round): {pairs}')
        if ded:
            dr = [r['scan'] / d['scan'] for r, d in zip(var, ded)
                  if d['scan'] > 0]
            lines.append(f'  paired ratio variant/ded med={median(dr):.3f} '
                         f'(prod/ded med='
                         f'{median(p["scan"] / d["scan"] for p, d in zip(prod, ded)):.3f})')
        nv = [r['ns'] for r in var]
        lines.append(f'  ns/vector variant={median(nv):.2f} '
                     f'prod={median(p["ns"] for p in prod):.2f} '
                     f'ded={median(d["ns"] for d in ded):.2f}')


def main():
    lines = ['=== v11 4a analysis (single-factor coarse-scan) ===']
    summ = RUN / 'iv4a_summary.txt'
    if summ.exists():
        lines.append('--- suite summary ---')
        for line in summ.read_text().splitlines():
            if ('rc=' in line or line.startswith('===') or
                    line.startswith('  factor ')):
                lines.append(f'  {line}')
    lines.append(f'reference v11 P1/P2 unpruned (separate invocations): '
                 f'int {V11_UNPRUNED["int_scan"]}+{V11_UNPRUNED["int_prep"]} '
                 f'| ded {V11_UNPRUNED["ded_scan"]}+'
                 f'{V11_UNPRUNED["ded_prep"]} | ratio '
                 f'{V11_UNPRUNED["ratio"]}')
    for v in VARIANTS:
        fmt_variant(v, lines)
    lines.append('--- interpretation ---')
    lines.append('  cluster0 vs prod: scalar epilogue (prefix load + '
                 'g_add/g_error gather + AoS stores) cost fraction.')
    lines.append('  dedshape vs cluster0: key-keep mask walk cost fraction.')
    lines.append('  flat vs prod: mapped-segment locality effect.')
    lines.append('  flat16 vs flat: alignment sensitivity (4KB page + 16B).')
    out = '\n'.join(lines)
    print(out)
    (RUN / 'v11_4a_analysis.out').write_text(out + '\n')
    print(f'\nwrote {RUN / "v11_4a_analysis.out"}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

#!/usr/bin/env python3
"""v11 P1+P2 analysis: replay equivalence verification + re-measured timings.

Parses (all under v11_runs/):
  - p1p2_summary.txt / p1p2_console.log: rc + qid-invariance lines
  - v11p2_prod.log, v11p2_unpruned.log, v11p1_*.log: sample verification,
    lists_total/vectors_total, scan_bytes/batch_bytes, scope
  - v11p2_prod.replay.csv, v11p2_unpruned.replay.csv: 5-round timings
    (columns: round,side,order,variant,queries,lists,vectors,blocks,
     prep_us_per_query,scan_us_per_query,ns_per_vector,ns_per_block,checksum)

Prints a consolidated report and writes v11_p1p2_analysis.out.
Usage: python3 analyze_v11_replays.py [RUN_DIR]
"""
import csv
import re
import sys
from pathlib import Path
from statistics import median

RUN = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent

V10_REF = {
    'prod': {'int_scan': 6064.7, 'int_prep': 33.3, 'ded_scan': 4920.1,
             'ded_prep': 8.8, 'ratio': 1.237, 'int_ns': 18.72, 'ded_ns': 17.53},
    'unpruned': {'int_scan': 5199.4, 'int_prep': 28.9, 'ded_scan': 3931.8,
                 'ded_prep': 6.1, 'ratio': 1.328, 'int_ns': 16.05,
                 'ded_ns': 14.01},
}

REPLAYS = [
    ('v11p2_prod', 'prod', 5),
    ('v11p2_unpruned', 'unpruned', 5),
    ('v11p1_qid_shuffle', 'qid_shuffle', 1),
    ('v11p1_row_shuffle', 'row_shuffle', 1),
    ('v11p1_negative', 'negative', 1),
]


def parse_log(name):
    out = {'rc': None, 'samples': [], 'passed': False, 'failed': False,
           'lists': {}, 'bytes': {}, 'scope': None, 'skipped': None}
    summ = (RUN / 'p1p2_summary.txt')
    if summ.exists():
        for line in summ.read_text().splitlines():
            if f'replay {name} rc=' in line:
                out['rc'] = int(re.search(r'rc=(\d+)', line).group(1))
    log = RUN / f'{name}.log'
    if not log.exists():
        out['missing'] = True
        return out
    for line in log.read_text(errors='replace').splitlines():
        if line.startswith('samples int:') or line.startswith('samples ded:'):
            side = 'int' if line.startswith('samples int') else 'ded'
            vals = dict(re.findall(r'(\w+)=([-\d.e+]+)', line))
            out['samples'].append((side, vals))
        elif 'sample verification: PASSED' in line:
            out['passed'] = True
        elif 'sample verification FAILED' in line:
            out['failed'] = True
        elif line.startswith('int: lists_total='):
            out['lists']['int'] = line
        elif line.startswith('ded: lists_total='):
            out['lists']['ded'] = line
        elif line.startswith('int: ') and 'scan_bytes=' in line:
            out['bytes']['int'] = line
        elif line.startswith('ded: ') and 'batch_bytes=' in line:
            out['bytes']['ded'] = line
        elif line.startswith('scope:'):
            out['scope'] = line
        elif line.startswith('samples: skipped'):
            out['skipped'] = line
    return out


def parse_replay_csv(name):
    """{side: {order: {variant: [round dicts]}}}"""
    path = RUN / f'{name}.replay.csv'
    if not path.exists():
        return None
    data = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            side, order, variant = row['side'], row['order'], row['variant']
            data.setdefault(side, {}).setdefault(order, {}).setdefault(
                variant, []).append({
                    'round': int(row['round']),
                    'scan': float(row['scan_us_per_query']),
                    'prep': float(row['prep_us_per_query']),
                    'ns': float(row['ns_per_vector']),
                    'vectors': int(row['vectors']),
                    'lists': int(row['lists']),
                    'blocks': int(row['blocks']),
                    'checksum': float(row['checksum']),
                })
    return data


def fmt_replay(name, kind, rounds, lines):
    lp = parse_log(name)
    lines.append(f'--- {name} (rc={lp["rc"]}) ---')
    for side in ('int', 'ded'):
        if 'lists' in lp and side in lp['lists']:
            lines.append(f'  {lp["lists"][side]}')
    if lp.get('bytes'):
        for side in ('int', 'ded'):
            if side in lp['bytes']:
                lines.append(f'  {lp["bytes"][side]}')
    if lp.get('skipped'):
        lines.append(f'  {lp["skipped"]}')
    for side, vals in lp['samples']:
        lines.append(f'  samples {side}: checked={vals.get("checked")} '
                     f'queries={vals.get("queries_with_samples")} '
                     f'key_mismatch={vals.get("key_mismatches")} '
                     f'value_mismatch={vals.get("value_mismatches")} '
                     f'tail_blocks={vals.get("tail_blocks")} '
                     f'distinct_ids={vals.get("distinct_model_ids") or vals.get("distinct_list_ids")} '
                     f'max_err(dist/lb/dot)={vals.get("distance")}/'
                     f'{vals.get("lower_bound")}/{vals.get("coarse_dot")} '
                     f'tol={vals.get("tolerance")}')
    lines.append(f'  verification: {"PASSED" if lp["passed"] else "FAILED" if lp["failed"] else "n/a"}')
    if lp.get('scope'):
        lines.append(f'  {lp["scope"]}')
    data = parse_replay_csv(name)
    if data and rounds > 1:
        for order in ('orig', 'seq'):
            for variant in ('kernel', 'full'):
                rows = {}
                for side in ('int', 'ded'):
                    rs = data.get(side, {}).get(order, {}).get(variant, [])
                    rows[side] = sorted(rs, key=lambda r: r['round'])
                if not all(rows.values()):
                    continue
                med = {s: (median(r['scan'] for r in rows[s]),
                           median(r['prep'] for r in rows[s]),
                           median(r['ns'] for r in rows[s]))
                       for s in ('int', 'ded')}
                ratios = [rows['int'][i]['scan'] / rows['ded'][i]['scan']
                          for i in range(len(rows['int']))]
                lines.append(
                    f'  {order}/{variant}: int scan={med["int"][0]:.1f} '
                    f'prep={med["int"][1]:.1f} ns/vec={med["int"][2]:.2f} | '
                    f'ded scan={med["ded"][0]:.1f} prep={med["ded"][1]:.1f} '
                    f'ns/vec={med["ded"][2]:.2f} | scan ratio med='
                    f'{median(ratios):.3f} rounds='
                    f'{" ".join(f"{r:.3f}" for r in ratios)}')
        # v10 comparison for prod/unpruned
        if kind in V10_REF:
            ref = V10_REF[kind]
            rows = {s: sorted(data[s]['orig']['full'], key=lambda r: r['round'])
                    for s in ('int', 'ded')}
            mi = (median(r['scan'] for r in rows['int']),
                  median(r['prep'] for r in rows['int']),
                  median(r['ns'] for r in rows['int']))
            md = (median(r['scan'] for r in rows['ded']),
                  median(r['prep'] for r in rows['ded']),
                  median(r['ns'] for r in rows['ded']))
            lines.append(
                f'  vs v10 P4 [{kind}]: int {mi[0]:.1f}+{mi[1]:.1f} '
                f'(v10 {ref["int_scan"]}+{ref["int_prep"]}) | ded '
                f'{md[0]:.1f}+{md[1]:.1f} (v10 {ref["ded_scan"]}'
                f'+{ref["ded_prep"]}) | int ns/vec {mi[2]:.2f} '
                f'(v10 {ref["int_ns"]}) | ded ns/vec {md[2]:.2f} '
                f'(v10 {ref["ded_ns"]})')
    elif data and rounds == 1:
        for side in ('int', 'ded'):
            rs = data.get(side, {}).get('orig', {}).get('full', [])
            for r in rs:
                lines.append(f'  {side} orig/full r{r["round"]}: scan='
                             f'{r["scan"]:.1f} prep={r["prep"]:.1f} '
                             f'ns/vec={r["ns"]:.2f} checksum={r["checksum"]:.1f}')


def main():
    lines = []
    lines.append('=== v11 P1+P2 analysis ===')
    summ = RUN / 'p1p2_summary.txt'
    if summ.exists():
        lines.append('--- suite summary ---')
        for line in summ.read_text().splitlines():
            if ('rc=' in line or 'qid-invariance' in line or
                    line.startswith('===')):
                lines.append(f'  {line}')
    for name, kind, rounds in REPLAYS:
        fmt_replay(name, kind, rounds, lines)
    out = '\n'.join(lines)
    print(out)
    (RUN / 'v11_p1p2_analysis.out').write_text(out + '\n')
    print(f'\nwrote {RUN / "v11_p1p2_analysis.out"}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

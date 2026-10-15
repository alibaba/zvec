#!/usr/bin/env python3
"""v12 production experiment analysis: gated refine->Top-60->exact-rerank
(rrf60@np200, yaml RefinerConfig gate) vs production (prod50@np293) on the
full t1/t4 bench path.

Reports:
  - recall validation lines from the run summary
  - raw qps of every round for every config (nothing discarded)
  - per-config median qps
  - same-round paired ratios rrf60/prod50 for t1 and t4 (medians + spread)
  - replay cross-reference (indicative only: replay lacks probe selection
    and end-to-end overheads, and cross-invocation drift is 15-25%)

Usage: python3 analyze_v12_prod.py [RUN_DIR]
"""
import re
import sys
from pathlib import Path
from statistics import median

RUN = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / 'prod'

CFGS = ['v12_prod50_np293_t1', 'v12_rrf60_np200_t1',
        'v12_prod50_np293_t4', 'v12_rrf60_np200_t4']
NAMES = {
    'v12_prod50_np293_t1': 'prod50 np293 t1',
    'v12_rrf60_np200_t1': 'rrf60 np200 t1',
    'v12_prod50_np293_t4': 'prod50 np293 t4',
    'v12_rrf60_np200_t4': 'rrf60 np200 t4',
}
ROUNDS = [1, 2, 3, 4, 5]


def parse_log(path):
    txt = path.read_text()
    qps = re.search(r'qps: ([0-9.]+)', txt)
    lat = re.search(r'Avg latency: ([0-9.]+)ms', txt)
    pq = re.search(r'Process query: (\d+)', txt)
    out = {
        'qps': float(qps.group(1)) if qps else None,
        'lat_ms': float(lat.group(1)) if lat else None,
        'queries': int(pq.group(1)) if pq else None,
    }
    for tag, pat in [('p50', r'50 Percentile:\s+([0-9.]+) ms'),
                     ('p90', r'90 Percentile:\s+([0-9.]+) ms'),
                     ('p99', r'99 Percentile:\s+([0-9.]+) ms')]:
        m = re.search(pat, txt)
        out[tag] = float(m.group(1)) if m else None
    return out


def main():
    print(f'run dir: {RUN}')
    summary = (RUN / 'v12_prod_summary.txt')
    if summary.exists():
        print('=== validation (from run summary) ===')
        for line in summary.read_text().splitlines():
            if line.startswith('recall ') or line.startswith('=== v12 prod VALIDATE'):
                print('  ' + line)

    data = {}
    for c in CFGS:
        data[c] = {}
        for r in ROUNDS:
            p = RUN / f'{c}_r{r}.log'
            if not p.exists():
                print(f'  MISSING {p}')
                continue
            data[c][r] = parse_log(p)

    print('\n=== raw qps (all rounds, rotating order; warmup excluded) ===')
    hdr = 'config              ' + ''.join(f'r{r:>8} ' for r in ROUNDS) + '   median'
    print(hdr)
    med = {}
    for c in CFGS:
        row = NAMES[c].ljust(19)
        vals = [data[c].get(r, {}).get('qps') for r in ROUNDS]
        for v in vals:
            row += f'{v:8.1f} ' if v else '     -- '
        keep = [v for v in vals if v]
        med[c] = median(keep) if keep else None
        row += f'{med[c]:9.1f}' if med[c] else '       --'
        print(row)

    print('\n=== avg latency ms (all rounds) ===')
    for c in CFGS:
        row = NAMES[c].ljust(19)
        for r in ROUNDS:
            v = data[c].get(r, {}).get('lat_ms')
            row += f'{v:8.2f} ' if v else '     -- '
        print(row)

    print('\n=== same-round paired ratios rrf60/prod50 (qps) ===')
    for t in ('t1', 't4'):
        a = f'v12_prod50_np293_{t}'
        b = f'v12_rrf60_np200_{t}'
        ratios = []
        for r in ROUNDS:
            va = data[a].get(r, {}).get('qps')
            vb = data[b].get(r, {}).get('qps')
            ratios.append(vb / va if va and vb else None)
        row = f'{t} '.ljust(6)
        for v in ratios:
            row += f'{v:8.3f} ' if v else '     -- '
        keep = [v for v in ratios if v]
        if keep:
            row += f'  median x{median(keep):.3f}  min x{min(keep):.3f}  max x{max(keep):.3f}'
        print(row)

    print('\n=== recall anchors ===')
    print('  prod50 np293 : 99.002% (production replica, v8 anchor)')
    print('  rrf60 np200  : see validation above (expect ~99.264%)')

    print('\n=== replay cross-reference (indicative only) ===')
    print('  replay rrf60@np200: 4603.1us/q total vs prod50@np293: 6817.8us/q')
    print('  replay ratio x0.675 lacks probe selection + end-to-end overheads;')
    print('  cross-invocation drift 15-25%; the paired bench ratios above are')
    print('  the primary result.')

    out = RUN / 'v12_prod_analysis.out'
    print(f'\n(writing this output to {out})')


if __name__ == '__main__':
    main()

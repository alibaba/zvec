#!/usr/bin/env python3
"""v10 P1 analysis: full-rotation paired benchmark (all diagnostics off).

Every round contains ALL configs (rotating order), so per-round paired
ratios are same-conditions comparisons. Reports:
  - raw qps of every round for every config (nothing discarded)
  - per-config median qps and latency percentiles
  - same-round paired ratios integrated/dedicated
  - 'ratio of medians' vs 'median of per-round ratios' as SEPARATE numbers

Usage: python3 analyze_v10_p1.py [P1_RUN_DIR]
"""
import re
import sys
from pathlib import Path
from statistics import median

HOME = Path.home()
LV = (HOME / 'workspace' / 'zvec' / 'rabitq_refactor' / 'local_validation' /
      'v10')
P1 = Path(sys.argv[1]) if len(sys.argv) > 1 else sorted(LV.glob('p1_*'))[-1]

CFGS = ['v10_base_np293_t1', 'v10_reuse_np293_t1', 'v10_ded_np249_t1',
        'v10_base_np293_t4', 'v10_reuse_np293_t4', 'v10_ded_np249_t4']
NAMES = {
    'v10_base_np293_t1': 'int baseline np293 t1',
    'v10_reuse_np293_t1': 'int reuse    np293 t1',
    'v10_ded_np249_t1': 'ded           np249 t1',
    'v10_base_np293_t4': 'int baseline np293 t4',
    'v10_reuse_np293_t4': 'int reuse    np293 t4',
    'v10_ded_np249_t4': 'ded           np249 t4',
}
ROUNDS = [1, 2, 3, 4, 5]


def parse_log(path):
    txt = path.read_text()
    qps = re.search(r'qps: ([0-9.]+)', txt)
    pq = re.search(r'Process query: (\d+)', txt)
    out = {
        'qps': float(qps.group(1)) if qps else None,
        'queries': int(pq.group(1)) if pq else None,
    }
    for tag, pat in [('p50', r'50 Percentile:\s+([0-9.]+) ms'),
                     ('p90', r'90 Percentile:\s+([0-9.]+) ms'),
                     ('p99', r'99 Percentile:\s+([0-9.]+) ms')]:
        m = re.search(pat, txt)
        out[tag] = float(m.group(1)) if m else None
    return out


def main():
    print(f'P1 run dir: {P1}')
    data = {}   # cfg -> {round: metrics}
    for c in CFGS:
        data[c] = {}
        for r in ROUNDS:
            p = P1 / f'{c}_r{r}.log'
            if not p.exists():
                print(f'  MISSING {p}')
                continue
            data[c][r] = parse_log(p)
    # time-based benchmarking: each 30s round processes qps*30 queries,
    # so counts differ by config by design; check only that all are >0
    bad = [c for c in CFGS
           if any(not m['queries'] for m in data[c].values())]
    counts = sorted({m['queries'] for c in CFGS for m in data[c].values()
                     if m['queries']})
    print(f'queries per 30s round: min={counts[0]} max={counts[-1]} '
          f'({"all logs have counts" if not bad else f"MISSING: {bad}"}) '
          '(differs per config: time-based benchmarking)')

    print('\n=== raw qps (all rounds, rotating order) ===')
    hdr = 'config                   ' + ''.join(f'r{r:>8} ' for r in ROUNDS)
    print(hdr)
    for c in CFGS:
        row = NAMES[c].ljust(24)
        for r in ROUNDS:
            v = data[c].get(r, {}).get('qps')
            row += f'{v:8.1f} ' if v else '     -- '
        print(row)

    print('\n=== per-config medians (5 rounds) ===')
    print(f'{"config":24s} {"qps_med":>9} {"qps_min":>9} {"qps_max":>9} '
          f'{"p50_ms":>7} {"p90_ms":>7} {"p99_ms":>7}')
    med = {}
    for c in CFGS:
        qs = [m['qps'] for m in data[c].values() if m['qps']]
        med[c] = median(qs)
        p50 = median([m['p50'] for m in data[c].values() if m['p50']])
        p90 = median([m['p90'] for m in data[c].values() if m['p90']])
        p99 = median([m['p99'] for m in data[c].values() if m['p99']])
        print(f'{NAMES[c]:24s} {med[c]:9.1f} {min(qs):9.1f} {max(qs):9.1f} '
              f'{p50:7.2f} {p90:7.2f} {p99:7.2f}')

    print('\n=== same-round paired ratios (integrated / dedicated) ===')
    pairs = [('v10_base_np293_t1', 'v10_ded_np249_t1', 't1 baseline/ded'),
             ('v10_reuse_np293_t1', 'v10_ded_np249_t1', 't1 reuse/ded   '),
             ('v10_base_np293_t4', 'v10_ded_np249_t4', 't4 baseline/ded'),
             ('v10_reuse_np293_t4', 'v10_ded_np249_t4', 't4 reuse/ded   ')]
    print(f'{"pair":17s}' + ''.join(f'r{r:>6} ' for r in ROUNDS) +
          'ratio-of-medians  median-of-ratios')
    for a, b, label in pairs:
        ratios = []
        for r in ROUNDS:
            qa = data[a].get(r, {}).get('qps')
            qb = data[b].get(r, {}).get('qps')
            ratios.append(qa / qb if qa and qb else None)
        row = label.ljust(17)
        for v in ratios:
            row += f'{v:6.3f} ' if v else '   -- '
        rom = med[a] / med[b]
        mor = median([v for v in ratios if v])
        row += f'{rom:16.3f}  {mor:16.3f}'
        print(row)
    print('  (ratio-of-medians and median-of-per-round-ratios are DIFFERENT '
          'statistics; both kept, labeled separately)')

    # v9 reference medians if available
    v9 = HOME / 'workspace' / 'zvec' / 'rabitq_refactor' / 'zvec' / 'v9_out' / \
        'bench_20261013-1708'
    v9map = {'v10_base_np293_t1': 'v9_base_np293_t1',
             'v10_reuse_np293_t1': 'v9_reuse_np293_t1',
             'v10_ded_np249_t1': 'v9_ded_np249_t1',
             'v10_base_np293_t4': 'v9_base_np293_t4',
             'v10_reuse_np293_t4': 'v9_reuse_np293_t4',
             'v10_ded_np249_t4': 'v9_ded_np249_t4'}
    if v9.exists():
        print('\n=== v9 reference medians (same configs) ===')
        for c in CFGS:
            qs = []
            for p in sorted(v9.glob(f'{v9map[c]}_r*.log')):
                m = re.search(r'qps: ([0-9.]+)', p.read_text())
                if m:
                    qs.append(float(m.group(1)))
            if qs:
                print(f'  {NAMES[c]:24s} v9 median qps={median(qs):9.1f} '
                      f'(n={len(qs)}) vs v10 {med[c]:9.1f}')
    print('\nDONE')
    return 0


if __name__ == '__main__':
    sys.exit(main())

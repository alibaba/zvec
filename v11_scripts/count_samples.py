#!/usr/bin/env python3
"""Count scan-sample rows per mode/field coverage in the v11 diag exports."""
import csv
import sys
from collections import Counter
from pathlib import Path

RUN = Path(__file__).parent

for name in ['int_orig.samples.csv', 'ded_orig.samples.csv',
             'ded_orig_unpruned.samples.csv', 'int_qid.samples.csv',
             'ded_qid.samples.csv', 'int_row.samples.csv',
             'ded_row.samples.csv']:
    p = RUN / name
    if not p.exists():
        print(f'{name}: MISSING')
        continue
    modes = Counter()
    queries = set()
    with open(p) as f:
        for row in csv.DictReader(f):
            modes[row['mode']] += 1
            queries.add((row['mode'], row['query']))
    detail = ' '.join(f'{m}={c}' for m, c in sorted(modes.items()))
    qs = sorted({q for m, q in queries if m == sorted(modes)[0]})
    print(f'{name}: {detail} | queries/mode={len(qs)} first={qs[:3]} '
          f'last={qs[-3:] if qs else []}')
sys.exit(0)

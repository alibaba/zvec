#!/usr/bin/env python3
"""Count lists.csv rows per mode for the v11-3 '1.6万列表/查询对' correction."""
import csv
import sys
from collections import Counter
from pathlib import Path

P0 = (Path.home() / 'workspace' / 'zvec' / 'rabitq_refactor' /
       'local_validation' / 'v10' / 'p0_20261013-192634')

for name in ['p0_int_np293_export.lists.csv', 'p0_ded_np249_unpruned.lists.csv']:
    p = P0 / name
    if not p.exists():
        print(f'{name}: MISSING')
        continue
    cnt = Counter()
    qids = set()
    with open(p) as f:
        for row in csv.DictReader(f):
            cnt[row['mode']] += 1
            qids.add(int(row['query']))
    print(f'{name}: modes={dict(cnt)} total={sum(cnt.values())} '
          f'queries={len(qids)} lists/q={ {m: c // len(qids) for m, c in cnt.items()} }')
sys.exit(0)

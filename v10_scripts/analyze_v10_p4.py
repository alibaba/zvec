#!/usr/bin/env python3
"""V10 P4 analysis: kernel-bench correctness + real access-sequence replay.

Inputs: a v10 p4_* run dir containing kernel_bench.log, v10p4_prod.log,
v10p4_prod.replay.csv, v10p4_unpruned.log, v10p4_unpruned.replay.csv.

Usage: python3 analyze_v10_p4.py [P4_RUN_DIR]
"""
import csv
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

# v10 P1 end-to-end medians (t1, single thread), qps -> us/query
P1_T1 = {
    'int baseline': 137.9,
    'int reuse': 136.6,
    'ded np249': 174.8,
}

KERNEL_LANES = ('turbo scanner.estimate', 'ded split_batch_estdist',
                'kernel accumulate_hacc')


def parse_kernel_bench(log_path):
    out = {'rows': {}, 'ok': False, 'crosscheck': None}
    if not log_path.exists():
        return out
    for line in log_path.read_text().splitlines():
        if line.startswith('config:'):
            out['config'] = line
        elif line.startswith('dispatch:'):
            out['dispatch'] = line
        elif line.startswith('crosscheck:'):
            out['crosscheck'] = line
        elif line.startswith('kernel share'):
            out['share'] = line.strip()
        else:
            m = re.match(r'^(\S.*?)\s+([\d.]+) ms/pass\s+([\d.]+) ns/block\(32\)'
                         r'\s+([\d.]+) ns/row\s+([\d.]+) Mrow/s', line)
            if m and m.group(1) in KERNEL_LANES:
                out['rows'][m.group(1)] = {
                    'ms_pass': float(m.group(2)),
                    'ns_block': float(m.group(3)),
                    'ns_row': float(m.group(4)),
                    'mrow_s': float(m.group(5)),
                }
    ok = True
    if out['crosscheck']:
        ck = out['crosscheck']
        for tok in ('max|d(est)|=', 'max|d(lower)|=', 'max|d(dot)|='):
            val = float(ck.split(tok)[1].split()[0])
            ok = ok and val <= 1e-3
        if 'compared=0' in ck or ' rows=0' in ck:
            ok = False
    else:
        ok = False
    out['ok'] = ok
    return out


def parse_replay_csv(csv_path):
    rows = list(csv.DictReader(open(csv_path)))
    by_key = defaultdict(list)
    for r in rows:
        key = (r['side'], r['order'], r['variant'])
        by_key[key].append(r)
    return rows, by_key


def median(vals):
    return statistics.median(vals)


def fmt(v, nd=2):
    return f'{v:.{nd}f}'


def replay_table(name, csv_path, log_path):
    print(f'\n=== {name} ({csv_path.name}) ===')
    log = log_path.read_text().splitlines() if log_path.exists() else []
    for line in log:
        if line.startswith(('queries=', 'int: ', 'ded: ')):
            print(' ', line)
    verify = [l for l in log if l.startswith('verify:')]
    for l in verify:
        print(' ', l)
        assert 'FAILED' not in l, f'{name}: replay verification failed'

    rows, by_key = parse_replay_csv(csv_path)
    rounds = sorted({int(r['round']) for r in rows})
    print(f'  rounds: {len(rounds)} ({rounds[0]}..{rounds[-1]}), '
          f'{len(rows)} rows')

    # determinism re-check from the CSV itself
    bad = []
    seen = defaultdict(set)
    for r in rows:
        seen[(r['side'], r['order'], r['variant'])].add(
            (r['vectors'], r['blocks'], r['checksum']))
    for k, s in seen.items():
        if len(s) != 1:
            bad.append(k)
    print(f'  CSV determinism (vectors/blocks/checksum per key): '
          f'{"OK" if not bad else "MISMATCH " + str(bad)}')
    assert not bad, f'{name}: nondeterministic replay rows'

    stats = {}
    for (side, order, variant), rs in sorted(by_key.items()):
        scan = [float(r['scan_us_per_query']) for r in rs]
        prep = [float(r['prep_us_per_query']) for r in rs]
        nsv = [float(r['ns_per_vector']) for r in rs]
        nsb = [float(r['ns_per_block']) for r in rs]
        stats[(side, order, variant)] = {
            'scan_med': median(scan),
            'scan_min': min(scan),
            'scan_max': max(scan),
            'prep_med': median(prep),
            'ns_vec_med': median(nsv),
            'ns_blk_med': median(nsb),
            'vectors': int(rs[0]['vectors']),
            'blocks': int(rs[0]['blocks']),
            'lists': int(rs[0]['lists']),
            'queries': int(rs[0]['queries']),
        }
    print(f'  {"side/order/variant":<22}{"scan us/q":>11}{"prep us/q":>11}'
          f'{"ns/vec":>9}{"ns/blk":>9}{"vectors/q":>12}{"lists/q":>10}')
    for k, s in stats.items():
        vq = s['vectors'] / s['queries']
        lq = s['lists'] / s['queries']
        print(f'  {"/".join(k):<22}{s["scan_med"]:>11.1f}'
              f'{s["prep_med"]:>11.2f}{s["ns_vec_med"]:>9.2f}'
              f'{s["ns_blk_med"]:>9.1f}{vq:>12.0f}{lq:>10.1f}')

    # Round-wise trend for the production-representative variant.
    print(f'  round trend (orig/full), us/query and per-round int/ded ratio:')
    per_round = defaultdict(dict)
    for r in rows:
        if r['order'] == 'orig' and r['variant'] == 'full':
            per_round[int(r['round'])][r['side']] = float(
                r['scan_us_per_query'])
    for rd in sorted(per_round):
        pr = per_round[rd]
        line = f'    round {rd}: ' + '  '.join(
            f'{s}={v:8.1f}' for s, v in sorted(pr.items()))
        if 'int' in pr and 'ded' in pr:
            line += f'  int/ded={pr["int"] / pr["ded"]:.3f}'
        print(line)
    return stats


def ratios(name, stats):
    print(f'\n--- {name}: paired ratios ---')
    print(f'  {"comparison":<34}{"scan ratio":>12}{"scan+prep ratio":>17}'
          f'{"ns/vec ratio":>14}')
    rows = []

    def get(side, order, variant):
        return stats.get((side, order, variant))

    for order in ('orig', 'seq'):
        for variant in ('kernel', 'full'):
            a, b = get('int', order, variant), get('ded', order, variant)
            if a and b:
                sr = a['scan_med'] / b['scan_med']
                tr = (a['scan_med'] + a['prep_med']) / (
                    b['scan_med'] + b['prep_med'])
                vr = a['ns_vec_med'] / b['ns_vec_med']
                rows.append((f'int/ded {order}/{variant}', sr, tr, vr))
    for side in ('int', 'ded'):
        for variant in ('kernel', 'full'):
            a, b = get(side, 'orig', variant), get(side, 'seq', variant)
            if a and b:
                sr = a['scan_med'] / b['scan_med']
                tr = (a['scan_med'] + a['prep_med']) / (
                    b['scan_med'] + b['prep_med'])
                vr = a['ns_vec_med'] / b['ns_vec_med']
                rows.append((f'{side} orig/seq {variant}', sr, tr, vr))
    for side in ('int', 'ded'):
        for order in ('orig', 'seq'):
            a, b = get(side, order, 'kernel'), get(side, order, 'full')
            if a and b:
                sr = a['scan_med'] / b['scan_med']
                tr = (a['scan_med'] + a['prep_med']) / (
                    b['scan_med'] + b['prep_med'])
                vr = a['ns_vec_med'] / b['ns_vec_med']
                rows.append((f'{side} kernel/full {order}', sr, tr, vr))
    for label, sr, tr, vr in rows:
        print(f'  {label:<34}{sr:>12.3f}{tr:>17.3f}{vr:>14.3f}')
    return rows


def main():
    if len(sys.argv) < 2:
        run = sorted(Path('/home/xingrui/workspace/zvec/rabitq_refactor/'
                          'local_validation/v10').glob('p4_*'))[-1]
    else:
        run = Path(sys.argv[1])
    print(f'P4 run dir: {run}')

    print('\n=== kernel bench (synthetic, v9 reference flags) ===')
    kb = parse_kernel_bench(run / 'kernel_bench.log')
    print(' ', kb.get('config', 'MISSING'))
    print(' ', kb.get('dispatch', 'MISSING'))
    print(' ', kb.get('crosscheck', 'MISSING'))
    print(' ', kb.get('share', 'MISSING'))
    for name in KERNEL_LANES:
        r = kb['rows'].get(name)
        if r:
            print(f'  {name:<26} ns/block={r["ns_block"]:>8.2f} '
                  f'ns/row={r["ns_row"]:>7.3f} Mrow/s={r["mrow_s"]:>6.2f}')
    assert kb['ok'], ('kernel bench crosscheck failed or missing '
                      '(zero checked blocks is not a pass)')
    print('  crosscheck verdict: PASS (nonzero blocks, all lanes within 1e-3)')
    t = kb['rows']['turbo scanner.estimate']['ns_block']
    d = kb['rows']['ded split_batch_estdist']['ns_block']
    print(f'  NOTE synthetic turbo/ded ns/block = {fmt(t / d, 3)}; ratios of '
          'independent microbench times are NOT exclusive segments of the '
          'end-to-end gap')

    prod_stats = replay_table('replay pair: production (int baseline vs ded '
                              'default)', run / 'v10p4_prod.replay.csv',
                              run / 'v10p4_prod.log')
    ratios('production pair', prod_stats)

    unpr_stats = replay_table('replay pair: pruning-free (both fully scanned)',
                              run / 'v10p4_unpruned.replay.csv',
                              run / 'v10p4_unpruned.log')
    ratios('pruning-free pair', unpr_stats)

    print('\n=== end-to-end anchoring (v10 P1, t1) ===')
    for label, qps in P1_T1.items():
        us = 1e6 / qps
        print(f'  {label:<15} {qps:>7.1f} qps = {us:>8.0f} us/query total')
    for tag, stats in (('prod', prod_stats), ('unpruned', unpr_stats)):
        for side in ('int', 'ded'):
            s = stats.get((side, 'orig', 'full'))
            if s:
                tot = s['scan_med'] + s['prep_med']
                print(f'  {tag} {side} replay scan+prep = {tot:.0f} us/query '
                      f'(scan {s["scan_med"]:.0f} + prep {s["prep_med"]:.0f})')

    print('\nOK: all replay verifications passed')


if __name__ == '__main__':
    main()

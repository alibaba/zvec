#!/usr/bin/env python3
"""v10 P0 analysis: determinism, diagnostics on/off, v8/v9/v10 chains,
and quantification of the np256-vs-v8 cross-nprobe 'DIFF 50/14'.

All comparisons are per (query, mode) on the full ordered result rows
(rank, key, score) plus ID sets, GT hits and scanned-list sets.
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

HOME = Path.home()
REPO = HOME / 'workspace' / 'zvec' / 'rabitq_refactor' / 'zvec'
P0 = REPO.parent / 'local_validation' / 'v10' / 'p0_20261013-192634'
V9 = REPO / 'v9_out' / 'export_20261013-174804'
V8 = HOME / 'bench' / 'rabitq_refactor' / 'v8'
GT_FILE = HOME / 'bench' / 'dataset' / 'cohere' / '1m' / 'neighbors.txt'
QFILE = HOME / 'bench' / 'dataset' / 'cohere' / '1m' / 'cohere_test_vector_1000.new.txt'

MODES_INT = ['baseline', 'unpruned', 'reuse', 'reuse_unpruned']


def load_results(path):
    """-> {mode: {qid: [(rank, key, score), ...]}} ordered by rank."""
    res = defaultdict(lambda: defaultdict(list))
    with open(path) as f:
        for row in csv.DictReader(f):
            res[row['mode']][int(row['query'])].append(
                (int(row['rank']), int(row['key']), float(row['score'])))
    for m in res:
        for q in res[m]:
            res[m][q].sort()
    return dict(res)


def load_lists(path):
    """-> {mode: {qid: [(list_id, scan_count), ...]}}."""
    lists = defaultdict(lambda: defaultdict(list))
    with open(path) as f:
        for row in csv.DictReader(f):
            lists[row['mode']][int(row['query'])].append(
                (int(row['list_id']), int(row['scan_count'])))
    return dict(lists)


def load_gt():
    gt = {}
    for line in open(GT_FILE):
        qid, _, ids = line.strip().partition(';')
        gt[int(qid)] = [int(x) for x in ids.split()[:50]]
    return gt


def query_file_order():
    order = []
    for line in open(QFILE):
        line = line.strip()
        if line:
            order.append(int(line.partition(';')[0]))
    return order


def gt_hits(gt, keys):
    return len(set(keys) & set(gt))


def cmp_results(a, b, label, gt, verbose_samples=3):
    """Compare {mode: {qid: [(rank,key,score)]}}; print classification."""
    modes = sorted(set(a) | set(b))
    total_diff_rows = 0
    for m in modes:
        ma, mb = a.get(m, {}), b.get(m, {})
        qids = sorted(set(ma) | set(mb))
        rows_same = rows_diff = id_same = id_diff = 0
        order_only = gt_changed = 0
        samples = []
        maxscore = 0.0
        for q in qids:
            ra, rb = ma.get(q, []), mb.get(q, [])
            if ra == rb:
                rows_same += 1
                continue
            rows_diff += 1
            ka = [x[1] for x in ra]
            kb = [x[1] for x in rb]
            sa, sb = set(ka), set(kb)
            if sa == sb:
                id_same += 1
                kind = 'order/rank-only' if sorted(ka) == sorted(kb) else 'id-multiset'
            else:
                id_diff += 1
                kind = 'id-set'
            ha, hb = gt_hits(gt.get(q, []), sa), gt_hits(gt.get(q, []), sb)
            if ha != hb:
                gt_changed += 1
                kind += f' GT{hb - ha:+d}'
            common = sa & sb
            for (r1, k1, s1) in ra:
                if k1 in common:
                    for (r2, k2, s2) in rb:
                        if k2 == k1:
                            maxscore = max(maxscore, abs(s1 - s2))
                            break
            if len(samples) < verbose_samples:
                added = sorted(sb - sa)
                removed = sorted(sa - sb)
                samples.append((q, kind, added[:5], removed[:5], ha, hb))
        if rows_diff:
            print(f"  {label} [{m}]: rows_same={rows_same} rows_diff={rows_diff} "
                  f"(id-set diff={id_diff}, id-same={id_same}) "
                  f"gt-hit-changed queries={gt_changed} "
                  f"max common-id |score diff|={maxscore:g}")
            for q, kind, add, rem, ha, hb in samples:
                print(f"    sample q={q}: {kind}, added={add}, removed={rem}, "
                      f"gt_hits {ha}->{hb}")
        else:
            print(f"  {label} [{m}]: all {rows_same} queries identical "
                  f"(rank+key+score)")
        total_diff_rows += rows_diff
    return total_diff_rows


def check_mapping(gt):
    order = query_file_order()
    assert len(order) == len(set(order)) == 1000, 'query file order invalid'
    assert set(order) == set(gt), 'query qids != GT qids'
    print(f"query file: {len(order)} lines, qids unique, set == GT qid set; "
          f"first-order qid={order[0]}, last={order[-1]}")
    print(f"  (results.csv 'query' column holds the file qid; validated by "
          f"GT hit computations elsewhere)")


def explain_cross_nprobe(gt, side, hi_np_res, lo_np_res, hi_lists, lo_lists,
                         hi_tag, lo_tag):
    print(f"\n=== cross-nprobe explanation: {hi_tag} vs {lo_tag} ({side}) ===")
    for m in sorted(set(hi_np_res) & set(lo_np_res)):
        qids = sorted(set(hi_np_res[m]) & set(lo_np_res[m]))
        diff_q = [q for q in qids
                  if hi_np_res[m].get(q) != lo_np_res[m].get(q)]
        sup_ok = sup_fail = 0
        extra_lists = []
        gt_delta = {}
        for q in qids:
            hi = {l for l, _ in hi_lists.get(m, {}).get(q, [])}
            lo = {l for l, _ in lo_lists.get(m, {}).get(q, [])}
            if hi >= lo:
                sup_ok += 1
                extra_lists.append(len(hi - lo))
            else:
                sup_fail += 1
            gt_delta[q] = (gt_hits(gt.get(q, []),
                                   {k for _, k, _ in hi_np_res[m].get(q, [])})
                           - gt_hits(gt.get(q, []),
                                     {k for _, k, _ in lo_np_res[m].get(q, [])}))
        tot = sum(gt_delta.values())
        hist = defaultdict(int)
        for v in gt_delta.values():
            hist[v] += 1
        print(f"  [{m}] differing queries: {len(diff_q)}/{len(qids)}; "
              f"hi-nprobe list set ⊇ lo-nprobe: {sup_ok}/{len(qids)} "
              f"(fail={sup_fail}); extra lists per query: "
              f"mean={sum(extra_lists) / len(extra_lists):.1f} "
              f"min={min(extra_lists)} max={max(extra_lists)}")
        print(f"    GT hit delta over all queries: total={tot:+d} "
              f"histogram={dict(sorted(hist.items()))}")
        for q in diff_q[:5]:
            hi = hi_np_res[m][q]
            lo = lo_np_res[m][q]
            hi_ids = {k for _, k, _ in hi}
            lo_ids = {k for _, k, _ in lo}
            print(f"    q={q}: added={sorted(hi_ids - lo_ids)[:6]} "
                  f"removed={sorted(lo_ids - hi_ids)[:6]} "
                  f"gt{gt_delta[q]:+d}")
        if diff_q:
            gtz = [q for q in diff_q if gt_delta[q] == 0]
            print(f"    differing queries with GT delta 0: {len(gtz)} / "
                  f"{len(diff_q)} (pure rank/id reshuffle within top-50)")


def main():
    gt = load_gt()
    print('=== query<->GT mapping ===')
    check_mapping(gt)

    print('\n=== A. determinism at dee6e15b (plain a vs plain b) ===')
    for cfg, name in [('int_np293', 'int np293'), ('int_np256', 'int np256'),
                      ('ded_np249', 'ded np249'), ('ded_np256', 'ded np256')]:
        a = load_results(P0 / f'p0_{cfg}_a.results.csv')
        b = load_results(P0 / f'p0_{cfg}_b.results.csv')
        cmp_results(a, b, f'{name} a-vs-b', gt)

    print('\n=== B. diagnostics on/off at dee6e15b ===')
    for cfg, name, variants in [
            ('int_np293', 'int np293', ['export', 'timing']),
            ('int_np256', 'int np256', ['export', 'timing']),
            ('ded_np249', 'ded np249', ['export']),
            ('ded_np256', 'ded np256', ['export'])]:
        plain = load_results(P0 / f'p0_{cfg}_a.results.csv')
        for v in variants:
            r = load_results(P0 / f'p0_{cfg}_{v}.results.csv')
            cmp_results(plain, r, f'{name} plain-vs-{v}', gt)

    print('\n=== C. cross-commit chains (same nprobe) ===')
    chains = [
        ('int np293', P0 / 'p0_int_np293_a.results.csv', V9 / 'int_np293.results.csv',
         V8 / 'v8_base_np293.results.csv', ['v10-plain', 'v9-export', 'v8']),
        ('int np256', P0 / 'p0_int_np256_a.results.csv', V9 / 'int_np256.results.csv',
         None, ['v10-plain', 'v9-export']),
        ('ded np249', P0 / 'p0_ded_np249_a.results.csv', V9 / 'ded_np249.results.csv',
         V8 / 'v8_ded_np249.results.csv', ['v10-plain', 'v9-export', 'v8']),
        ('ded np256', P0 / 'p0_ded_np256_a.results.csv', V9 / 'ded_np256.results.csv',
         None, ['v10-plain', 'v9-export']),
    ]
    loaded = []
    for name, p10, p9, p8, tags in chains:
        r10, r9 = load_results(p10), load_results(p9)
        r8 = load_results(p8) if p8 else None
        loaded.append((name, r10, r9, r8))
        print(f'  -- {name} --')
        cmp_results(r10, r9, 'v10-plain vs v9-export', gt)
        if r8 is not None:
            cmp_results(r10, r8, 'v10-plain vs v8', gt)
    # unpruned dedicated chain
    print('  -- ded np256/np249 unpruned mode --')
    for cfg, v9p in [('ded_np256', V9 / 'ded_np256_unpruned.results.csv'),
                     ('ded_np249', V9 / 'ded_np249_unpruned.results.csv')]:
        r10 = load_results(P0 / f'p0_{cfg}_unpruned.results.csv')
        r9 = load_results(v9p)
        cmp_results(r10, r9, f'{cfg} v10-unpruned vs v9-unpruned', gt)

    print('\n=== D. quantified explanation of v9 DIFF 50/14 ===')
    hi_int = load_results(V9 / 'int_np293.results.csv')
    lo_int = load_results(V9 / 'int_np256.results.csv')
    hi_int_lists = load_lists(V9 / 'int_np293.lists.csv')
    lo_int_lists = load_lists(V9 / 'int_np256.lists.csv')
    explain_cross_nprobe(gt, 'integrated', hi_int, lo_int, hi_int_lists,
                         lo_int_lists, 'np293 (v9)', 'np256 (v9)')
    print('  (the v9 analysis.log compared np256-v9 against np293-v8 because '
          'no v8 np256 results exist; that is a cross-nprobe comparison)')
    hi_ded = load_results(V9 / 'ded_np256.results.csv')
    lo_ded = load_results(V9 / 'ded_np249.results.csv')
    hi_ded_lists = load_lists(V9 / 'ded_np256.lists.csv')
    lo_ded_lists = load_lists(V9 / 'ded_np249.lists.csv')
    explain_cross_nprobe(gt, 'dedicated', hi_ded, lo_ded, hi_ded_lists,
                         lo_ded_lists, 'np256 (v9)', 'np249 (v9)')
    print('  (the v9 analysis.log compared np256-v9 against np249-v8 for the '
          'same reason)')

    print('\nDONE')


if __name__ == '__main__':
    main()

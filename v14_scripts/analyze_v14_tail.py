#!/usr/bin/env python3
"""Recompute tail contributions of the independent-set diagnostic per query.

v14 prof summary claimed the indep-set refined mean (30,189 for int np293)
was 'driven by a ~5% pathological tail'. That claim was never computed; a
5%-at-max upper bound is only ~14% of the total. This script reads the
per-query diagnostic CSVs and reports, per config and metric
(refined_count / post_scan_ns / query_ns):
  - p50/p90/p95/p99/max,
  - the share of the TOTAL held by the heaviest 5% / 10% / 25% queries
    (each metric ranked independently),
  - the overlap (Jaccard) between the heaviest-5% query sets of the
    refined_count, post_scan_ns and query_ns rankings (are the count tail
    and the time tail the same queries?).
Also prints the tuning-set counterparts for contrast. Counts are exact;
times carry stats-collection overhead and are for relative attribution.

Usage: python3 analyze_v14_tail.py [RUN_DIR]   (writes v14_tail_analysis.txt)
"""
import csv
import sys
from pathlib import Path
from statistics import median

RUN = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent
QUANTILES = (0.50, 0.90, 0.95, 0.99)
SHARES = (0.05, 0.10, 0.25)
METRICS = ('refined_count', 'post_scan_ns', 'query_ns')

# (label, csv path, mode) -- baseline == pruning on, reuse off; dedicated
# is the single dedicated mode. Production-equivalent scan settings.
RUNS = [
    ('int np293 tuning', 'v14_prof_int_np293.queries.csv', 'baseline'),
    ('rrf np180/55 tuning', 'v14_prof_int_np180_top55.queries.csv',
     'baseline'),
    ('native np249 tuning', 'v14_prof_ded_np249.queries.csv', 'dedicated'),
    ('natref np170/55 tuning', 'v14_prof_ded_np170_top55.queries.csv',
     'dedicated'),
    ('int np293 INDEP', 'v14_prof_int_np293_indep.queries.csv', 'baseline'),
    ('rrf np180/55 INDEP', 'v14_prof_int_np180_top55_indep.queries.csv',
     'baseline'),
    ('native np249 INDEP', 'v14_prof_ded_np249_indep.queries.csv',
     'dedicated'),
    ('natref np170/55 INDEP', 'v14_prof_ded_np170_top55_indep.queries.csv',
     'dedicated'),
]


def pct(sorted_vals, q):
    """Linear-interpolation percentile on an already-sorted list."""
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    pos = q * (len(sorted_vals) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = pos - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def top_share(pairs, share):
    """(query, value) -> (share of total, {top queries})."""
    pairs = sorted(pairs, key=lambda p: p[1], reverse=True)
    k = max(1, round(share * len(pairs)))
    top = {q for q, _ in pairs[:k]}
    total = sum(v for _, v in pairs)
    return (sum(v for _, v in pairs[:k]) / total if total else 0.0), top


def main():
    lines = ['=== v14 tail-contribution recompute (per-query diag CSVs) ===']
    for label, csv_name, mode in RUNS:
        path = RUN / csv_name
        if not path.exists():
            lines.append(f'--- {label}: MISSING {csv_name} (skipped) ---')
            continue
        rows = []
        with path.open(newline='', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                if row['mode'] != mode:
                    continue
                rows.append({m: float(row[m]) for m in METRICS}
                            | {'query': int(row['query'])})
        if not rows:
            lines.append(f'--- {label}: no mode={mode} rows in {csv_name} ---')
            continue
        n = len(rows)
        lines.append(f'--- {label} (n={n}, mode={mode}) ---')
        tops = {}
        for m in METRICS:
            sv = sorted(r[m] for r in rows)
            qs = ' '.join(f'p{int(q * 100)}={pct(sv, q):.0f}' for q in QUANTILES)
            lines.append(f'  {m}: {qs} max={sv[-1]:.0f} '
                         f'mean={sum(sv) / n:.0f}')
            shares = []
            for s in SHARES:
                share, top = top_share([(r['query'], r[m]) for r in rows], s)
                tops[(m, s)] = top
                shares.append(f'top{int(s * 100)}%={share * 100:.1f}%')
            lines.append(f'    contribution: {" ".join(shares)} of total')
        for s in SHARES:
            a = tops[('refined_count', s)]
            b = tops[('post_scan_ns', s)]
            c = tops[('query_ns', s)]
            j = lambda x, y: len(x & y) / len(x | y) if x | y else 1.0
            lines.append(
                f'    top{int(s * 100)}% overlap: refined~post_scan '
                f'J={j(a, b):.2f}  refined~query J={j(a, c):.2f}  '
                f'post_scan~query J={j(b, c):.2f}')
    out = '\n'.join(lines)
    print(out)
    (RUN / 'v14_tail_analysis.txt').write_text(out + '\n', encoding='utf-8')
    print(f"\nwrote {RUN / 'v14_tail_analysis.txt'}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

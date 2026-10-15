#!/usr/bin/env python3
"""v13 analysis: rotation-fixed replay timing, nprobe fill-in and N sweep.

Inputs (under the run dir):
  - v13_timing_summary.txt: replay rc / summary / verify lines
  - v13_np{N}.replay.csv: same columns as v12
  - v13_validate_summary.txt: per-query candidate verification (N=52..100)

Statistics (v12/v13 rules): per-variant medians over ALL rounds plus
per-round paired ratios vs prod50 within the same invocation. The v13
tool rotates the actual EXECUTION order every round (round r shifts the
task list by (r-1) mod count), so no variant keeps a permanent warm/cold
position; CSV row order equals execution order.

refines is the TOTAL refine call count summed over the 1000 queries;
reported both as total and per-query average (the v12 report line
"refine/q=765k" mislabeled the total as per-query).

Writes v13_analysis.out. Usage: python3 analyze_v13.py [RUN_DIR]
"""
import csv
import re
import sys
from pathlib import Path
from statistics import median

RUN = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent

NPROBES = [150, 160, 170, 180, 190, 200, 249, 272, 293, 320]
TOPNS = [52, 55, 60, 80, 100]
TARGET = 99.0
QCOUNT = 1000
V8_RECALL = {272: 98.968, 288: 98.994, 292: 98.998, 293: 99.002,
             296: 99.008, 300: 99.014, 320: 99.042}


def parse_csv(np):
    path = RUN / f'v13_np{np}.replay.csv'
    if not path.exists():
        return None
    data = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            if row['side'] != 'int' or row['order'] != 'orig':
                continue
            data.setdefault(row['variant'], []).append({
                'round': int(row['round']),
                'scan': float(row['scan_us_per_query']),
                'prep': float(row['prep_us_per_query']),
                'rerank': float(row['rerank_us_per_query']),
                'refines': int(row['refines']),
                'recall': float(row['recall']),
                'coverage': float(row['coverage']),
                'vectors': int(row['vectors']),
            })
    for v in data.values():
        v.sort(key=lambda r: r['round'])
    return data


def tot(r):
    return r['scan'] + r['prep'] + r['rerank']


def main():
    lines = []
    lines.append('=== v13 analysis: rotation-fixed timing + np160-190 + N=52/55/60 ===')

    summ = RUN / 'v13_timing_summary.txt'
    if summ.exists():
        text = summ.read_text()
        rcs = re.findall(r'replay np=(\d+) rc=(\d+)', text)
        lines.append(f'replay rc: {rcs or "none"}')
        verifies = [l for l in text.splitlines() if l.startswith('verify:')]
        ok_all = all('all replayed work matched' in v for v in verifies)
        lines.append(f'verify: {"all PASSED" if verifies and ok_all else "CHECK"} '
                     f'({len(verifies)} replays)')
        ctrl = [l for l in text.splitlines() if 'normal control rc=' in l]
        lines.extend(ctrl or ['samples normal control: MISSING'])

    val = RUN / 'v13_validate_summary.txt'
    if val.exists():
        lines.append('--- per-query candidate verification (np293, vs production topk=N) ---')
        for l in val.read_text().splitlines():
            if 'rerank verify' in l or 'replay topk' in l:
                lines.append('  ' + l)

    points = []
    for np in NPROBES:
        data = parse_csv(np)
        if not data or 'prod50' not in data:
            lines.append(f'np{np}: replay csv missing')
            continue
        prod = {r['round']: r for r in data['prod50']}
        p_total = median(tot(r) for r in prod.values())
        p_recall = median(r['recall'] for r in prod.values()) * 100.0
        p_ref = median(r['refines'] for r in prod.values())
        p_vecs = {r['vectors'] for r in prod.values()}
        points.append((np, 'prod50', p_total, p_recall, p_ref, []))
        ref = V8_RECALL.get(np)
        lines.append(
            f'np{np}: prod50 total={p_total:.1f}us/q qps={1e6/p_total:.0f} '
            f'refine_total={p_ref:.0f} ({p_ref/QCOUNT:.0f}/q) '
            f'recall={p_recall:.3f}%'
            + (f' (v8 bench {ref:.3f}%)' if ref else ''))
        for n in TOPNS:
            for scheme in ('rrf', 'rlb'):
                name = f'{scheme}{n}'
                if name not in data:
                    continue
                rs = {r['round']: r for r in data[name]}
                if {r['vectors'] for r in rs.values()} != p_vecs:
                    lines.append(f'np{np} {name}: WORK MISMATCH '
                                 f'{sorted({r["vectors"] for r in rs.values()})} '
                                 f'vs prod50 {sorted(p_vecs)}')
                c_total = median(tot(r) for r in rs.values())
                c_scan = median(r['scan'] for r in rs.values())
                c_rr = median(r['rerank'] for r in rs.values())
                c_recall = median(r['recall'] for r in rs.values()) * 100.0
                c_ref = median(r['refines'] for r in rs.values())
                ratios = [tot(rs[r]) / tot(prod[r])
                          for r in sorted(rs) if r in prod]
                med_ratio = median(ratios) if ratios else 0.0
                flag = ''
                if any(abs(x - med_ratio) > 0.2 * med_ratio for x in ratios):
                    flag = ' [OUTLIER ROUNDS -- re-run invocation]'
                points.append((np, name, c_total, c_recall, c_ref, ratios))
                lines.append(
                    f'np{np}: {name} total={c_total:.1f}us/q '
                    f'qps={1e6/c_total:.0f} scan={c_scan:.1f} '
                    f'rerank={c_rr:.1f} refine_total={c_ref:.0f} '
                    f'({c_ref/QCOUNT:.0f}/q) recall={c_recall:.3f}% '
                    f'| paired med x{med_ratio:.3f} '
                    f'rounds={" ".join(f"{x:.3f}" for x in ratios)}{flag}')

    lines.append(f'--- recall frontier (target R@50 >= {TARGET}%) ---')
    best = None
    for np, name, total, recall, _ref, _ratios in points:
        if name.startswith('rrf') and recall >= TARGET:
            if best is None or total < best[2]:
                best = (np, name, total, recall)
    for np, name, total, recall, _ref, _ratios in points:
        if name.startswith('rrf') and recall >= TARGET:
            lines.append(f'  {name}@np{np}: recall={recall:.3f}% '
                         f'total={total:.1f}us/q')
    if best:
        base = [p for p in points if p[0] == 293 and p[1] == 'prod50']
        lines.append(
            f'best: {best[1]}@np{best[0]} recall={best[3]:.3f}% '
            f'total={best[2]:.1f}us/q')
        if base:
            lines.append(
                f'reference: prod50@np293 recall={base[0][3]:.3f}% '
                f'total={base[0][2]:.1f}us/q | cross-invocation delta '
                f'x{best[2]/base[0][2]:.3f} (indicative only)')
    else:
        lines.append('no rrf config reached the target; full table above')

    lines.append('scope: totals exclude probe selection and finalize on '
                 'both sides; rrf totals include refine, heap maintenance, '
                 'original-vector reads and exact scoring; replay totals '
                 'are not end-to-end QPS')
    out = '\n'.join(lines)
    print(out)
    (RUN / 'v13_analysis.out').write_text(out + '\n')
    print(f'\nwrote {RUN / "v13_analysis.out"}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

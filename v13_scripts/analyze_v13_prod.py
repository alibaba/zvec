#!/usr/bin/env python3
"""v13 4-way production analysis: QPS / latency / P95 / P99 / RSS medians and
per-round paired ratios from v13_prod_summary.txt (+ holdout summary).

Schemes (t1 and t4):
  int    v12_prod50_np293     integrated baseline
  rrf    v13_rrf{N}_np{np}    integrated + refine rerank
  native v13_native_np249     dedicated kIVFRabitq baseline
  natref v13_natref_np{np}    dedicated + refine rerank

Warmup runs are excluded from medians. All 5 rounds are kept; per-round
paired ratios compare schemes within the same round (same rotation slot
order), which is the drift-resistant quantity on a shared host.

Writes v13_prod_analysis.out. Usage: python3 analyze_v13_prod.py [RUN_DIR]
"""
import re
import sys
from pathlib import Path
from statistics import median

RUN = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent
LINE = re.compile(
    r'bench (?P<tag>\S+) rc=(?P<rc>\d+) qps=(?P<qps>[0-9.]+|MISSING) '
    r'avg_ms=(?P<avg>[0-9.]+|MISSING) p95_ms=(?P<p95>[0-9.]+|MISSING) '
    r'p99_ms=(?P<p99>[0-9.]+|MISSING) rss_kb=(?P<rss>[0-9]+|MISSING) '
    r'loadavg=(?P<load>[0-9.]+)')


def load_runs(path):
    runs = {}
    if not path.exists():
        return runs
    for m in LINE.finditer(path.read_text()):
        if m.group('rc') != '0':
            continue
        tag = m.group('tag')
        if tag.startswith('warmup_'):
            continue
        name = re.sub(r'_r\d+$', '', tag)
        run_r = re.search(r'_r(\d+)$', tag)
        runs.setdefault(name, {})[int(run_r.group(1)) if run_r else 0] = {
            'qps': float(m.group('qps')), 'avg': float(m.group('avg')),
            'p95': float(m.group('p95')), 'p99': float(m.group('p99')),
            'rss': float(m.group('rss')), 'load': float(m.group('load'))}
    return runs


def main():
    lines = ['=== v13 4-way production analysis (rebuilt binaries) ===']
    runs = load_runs(RUN / 'v13_prod_summary.txt')
    if not runs:
        lines.append('no bench lines parsed from v13_prod_summary.txt')
        print('\n'.join(lines))
        return 1

    schemes = {}
    for name, rr in runs.items():
        # classify by prefix
        if name.startswith('v12_prod50_np293'):
            scheme = 'int'
        elif name.startswith('v13_rrf'):
            scheme = 'rrf'
        elif name.startswith('v13_native_np249'):
            scheme = 'native'
        elif name.startswith('v13_natref'):
            scheme = 'natref'
        else:
            continue
        threads = 't4' if '_t4' in name else 't1'
        schemes.setdefault((scheme, threads), {})[name] = rr

    for (scheme, threads), cfgs in sorted(schemes.items()):
        lines.append(f'--- {scheme} ({threads}) ---')
        for name, rr in sorted(cfgs.items()):
            rounds = sorted(rr)
            q = [rr[r]['qps'] for r in rounds]
            p95 = [rr[r]['p95'] for r in rounds]
            p99 = [rr[r]['p99'] for r in rounds]
            rss = [rr[r]['rss'] for r in rounds]
            loads = [rr[r]['load'] for r in rounds]
            lines.append(
                f'{name}: rounds={" ".join(f"{x:.1f}" for x in q)} '
                f'| qps med={median(q):.1f} p95 med={median(p95):.1f}ms '
                f'p99 med={median(p99):.1f}ms rss med={median(rss)/1024/1024:.2f}GB '
                f'loadavg med={median(loads):.1f}')

    lines.append('--- per-round paired QPS ratios (same round, same rotation) ---')
    for threads in ('t1', 't4'):
        for a, b in (('rrf', 'int'), ('natref', 'native'),
                     ('natref', 'int'), ('rrf', 'natref'), ('natref', 'rrf')):
            ca = schemes.get((a, threads))
            cb = schemes.get((b, threads))
            if not ca or not cb:
                continue
            ra = list(ca.values())[0]
            rb = list(cb.values())[0]
            common = sorted(set(ra) & set(rb))
            if not common:
                continue
            ratios = [ra[r]['qps'] / rb[r]['qps'] for r in common]
            lines.append(
                f'{threads} {a}/{b}: '
                f'{" ".join(f"{x:.3f}" for x in ratios)} '
                f'| med x{median(ratios):.3f}')

    hold = RUN / 'v13_holdout_summary.txt'
    if hold.exists():
        lines.append('--- holdout (qid 500-999, independent confirmation) ---')
        for l in hold.read_text().splitlines():
            if (l.startswith('holdout recall') or l.startswith('holdout bench')
                    or l.startswith('WARN:')):
                lines.append('  ' + l)
    else:
        lines.append('holdout summary: not run yet')

    out = '\n'.join(lines)
    print(out)
    (RUN / 'v13_prod_analysis.out').write_text(out + '\n')
    print(f'\nwrote {RUN / "v13_prod_analysis.out"}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

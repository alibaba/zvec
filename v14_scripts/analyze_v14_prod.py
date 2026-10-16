#!/usr/bin/env python3
"""v14 4-way production analysis with a data-integrity gate.

Medians (QPS / P95 / P99 / RSS) and per-round paired ratios come from
v14_prod_summary.txt (+ independent-set summary). BEFORE any conclusion
is emitted the gate verifies the formal timing grid: 4 schemes x 2 thread
tiers x 5 rounds = 40 unique rc=0 records (warmups are excluded from the
40, but all 8 must exist with rc=0), every qps/avg/p95/p99/rss field
present, finite and within sanity bounds (p95 <= p99), no duplicate
(config, round) keys, no failed rounds, no unparseable or FAILED lines,
and paired comparisons require identical round sets (never the
intersection). Any violation exits non-zero, prints the problems to
stderr and suppresses the formal conclusions and v14_prod_analysis.out.

Schemes (t1 and t4):
  int    v12_prod50_np293          integrated baseline
  rrf    v13_rrf{N}_np{np}         integrated + refine rerank
  native v13_native_np{np}         dedicated kIVFRabitq baseline
  natref v14_natref_np{np}_top{N}  dedicated + refine rerank (v14 sweep)

The independent-set section reports recall/QPS on 900 queries from the
cohere 10m test split (an exact-md5 scan removed the 100 queries that are
members of the 1M base; the kept 900 are disjoint from the base and from
the 1000 tuning queries, so they never took part in tuning): this IS the
independent validation (unlike the v13 q500-999 grouped check). Per-round
paired ratios compare schemes within the same round (same rotation slot
order), the drift-resistant quantity on a shared host; note the median
and the paired ratio are different statistics -- the median summarizes
absolute throughput, the ratio cancels round-level drift.

Writes v14_prod_analysis.out. Usage: python3 analyze_v14_prod.py [RUN_DIR]
"""
import re
import sys
from pathlib import Path
from statistics import median

ROUNDS = 5
TIERS = ('t1', 't4')
SCHEME_RE = {
    'int': re.compile(r'^v12_prod50_np293$'),
    'rrf': re.compile(r'^v13_rrf\d+_np\d+$'),
    'native': re.compile(r'^v13_native_np\d+$'),
    'natref': re.compile(r'^v14_natref_np\d+_top\d+$'),
}
PAIRS = (('rrf', 'int'), ('natref', 'native'), ('natref', 'int'),
         ('rrf', 'natref'), ('natref', 'rrf'))
FIELDS = ('qps', 'avg', 'p95', 'p99', 'rss', 'load')
BOUNDS = {'qps': (0.0, 1e6, True), 'avg': (0.0, 1e5, True),
          'p95': (0.0, 1e5, True), 'p99': (0.0, 1e5, True),
          'rss': (1024.0, 1e12, True), 'load': (0.0, 1e5, False)}

RUN = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent
LINE = re.compile(
    r'bench (?P<tag>\S+) rc=(?P<rc>\d+) qps=(?P<qps>[0-9.]+|MISSING) '
    r'avg_ms=(?P<avg>[0-9.]+|MISSING) p95_ms=(?P<p95>[0-9.]+|MISSING) '
    r'p99_ms=(?P<p99>[0-9.]+|MISSING) rss_kb=(?P<rss>[0-9]+|MISSING) '
    r'loadavg=(?P<load>[0-9.]+)')
INDEP_BENCH = re.compile(
    r'^indep bench (?P<name>\S+) rc=(?P<rc>\d+) '
    r'qps=(?P<qps>[0-9.]+|MISSING) rss_kb=(?P<rss>[0-9]+|MISSING) '
    r'loadavg=(?P<load>[0-9.]+)$')
INDEP_RECALL = re.compile(
    r'^indep recall (?P<name>\S+) rc=(?P<rc>\d+) '
    r'Recall@50: (?P<v>[0-9.]+|MISSING)')


def parse_summary(text, errors):
    formal, warm, recalls = {}, {}, []
    for lineno, line in enumerate(text.splitlines(), 1):
        s = line.strip()
        m = LINE.search(s)
        if not m:
            if s.startswith('bench '):
                errors.append(f'line {lineno}: unparseable bench line: {s}')
            elif s.startswith('FAILED:'):
                errors.append(f'line {lineno}: aborted run: {s}')
            elif s.startswith('recall ') and ' Recall@50: ' in s:
                if ' rc=0 ' not in s:
                    errors.append(f'line {lineno}: failed recall line: {s}')
                recalls.append(s)
            continue
        tag = m.group('tag')
        rec = {'tag': tag, 'line': lineno, 'rc': int(m.group('rc')),
               'raw': {f: m.group(f) for f in FIELDS}}
        if tag.startswith('warmup_'):
            if tag in warm:
                errors.append(
                    f'line {lineno}: duplicate warmup record {tag} '
                    f'(first at line {warm[tag]["line"]})')
            warm[tag] = rec
            continue
        run_r = re.search(r'_r(\d+)$', tag)
        name = re.sub(r'_r\d+$', '', tag)
        rnd = int(run_r.group(1)) if run_r else 0
        if (name, rnd) in formal:
            first = formal[(name, rnd)]
            errors.append(
                f'line {lineno}: duplicate (config, round) record {tag}: '
                f'round {rnd} of {name} already recorded at line '
                f'{first["line"]}')
        formal[(name, rnd)] = rec
    return formal, warm, recalls


def classify(name):
    tier = name[-2:] if name.endswith(('_t1', '_t4')) else ''
    base = name[:-3] if tier else name
    for scheme, pat in SCHEME_RE.items():
        if pat.match(base):
            return scheme, tier
    return None, tier


def validate_structure(formal, warm, errors):
    by_cfg = {}
    for (name, rnd), rec in sorted(formal.items()):
        if rec['rc'] != 0:
            errors.append(
                f'FAILED round: {rec["tag"]} rc={rec["rc"]} '
                f'(line {rec["line"]})')
        scheme, tier = classify(name)
        if scheme is None or tier not in TIERS:
            errors.append(
                f'unrecognized config name: {name} (line {rec["line"]})')
            continue
        by_cfg.setdefault((scheme, tier), {}).setdefault(name, {})[rnd] = rec
    for scheme in SCHEME_RE:
        for tier in TIERS:
            cfgs = by_cfg.get((scheme, tier), {})
            if not cfgs:
                errors.append(
                    f'missing config: scheme {scheme} ({tier}) has no records')
                continue
            if len(cfgs) > 1:
                errors.append(
                    f'ambiguous configs for {scheme} ({tier}): '
                    f'{", ".join(sorted(cfgs))} (expected exactly one)')
            for name, rr in cfgs.items():
                missing = [r for r in range(1, ROUNDS + 1) if r not in rr]
                if missing:
                    errors.append(
                        f'missing rounds: {name} lacks '
                        f'{", ".join(f"r{r}" for r in missing)}')
                extra = sorted(r for r in rr if not 1 <= r <= ROUNDS)
                if extra:
                    errors.append(
                        f'unexpected rounds: {name} has '
                        f'{", ".join(f"r{r}" for r in extra)}')
    n = sum(len(rr) for cfgs in by_cfg.values() for rr in cfgs.values())
    if n != 4 * 2 * ROUNDS:
        errors.append(
            f'formal record count {n} != {4 * 2 * ROUNDS} '
            f'(4 schemes x 2 tiers x {ROUNDS} rounds)')
    formal_names = {name for (name, rnd) in formal}
    for tag, rec in sorted(warm.items()):
        if rec['rc'] != 0:
            errors.append(
                f'FAILED warmup: {tag} rc={rec["rc"]} (line {rec["line"]})')
        if tag[len('warmup_'):] not in formal_names:
            errors.append(
                f'unexpected warmup tag {tag}: no matching formal config')
    for name in sorted(formal_names):
        if f'warmup_{name}' not in warm:
            errors.append(f'missing warmup record: warmup_{name}')


def check_fields(formal, errors):
    vals = {}
    for (name, rnd), rec in sorted(formal.items()):
        v = {}
        for f in FIELDS:
            raw = rec['raw'][f]
            if raw == 'MISSING':
                errors.append(f'field MISSING: {rec["tag"]} {f}')
                continue
            try:
                x = float(raw)
            except ValueError:
                errors.append(f'field unparseable: {rec["tag"]} {f}={raw}')
                continue
            lo, hi, strict = BOUNDS[f]
            if not ((lo < x if strict else lo <= x) and x <= hi):
                errors.append(
                    f'field out of range: {rec["tag"]} {f}={raw} '
                    f'(expected {lo}..{hi})')
                continue
            v[f] = x
        if 'p95' in v and 'p99' in v and v['p95'] > v['p99']:
            errors.append(
                f'field inconsistent: {rec["tag"]} '
                f'p95_ms={rec["raw"]["p95"]} > p99_ms={rec["raw"]["p99"]}')
        vals[(name, rnd)] = v
    return vals


def check_indep(text, errors):
    keep = []
    for lineno, line in enumerate(text.splitlines(), 1):
        s = line.strip()
        if not (s.startswith('indep recall') or s.startswith('indep bench')
                or s.startswith('WARN:')):
            continue
        keep.append(s)
        if s.startswith('WARN:'):
            continue
        m = (INDEP_RECALL if s.startswith('indep recall')
             else INDEP_BENCH).match(s)
        if not m:
            errors.append(f'indep line {lineno}: unparseable: {s}')
            continue
        if m.group('rc') != '0':
            errors.append(f'indep line {lineno}: rc={m.group("rc")}: {s}')
            continue
        if s.startswith('indep recall'):
            v = m.group('v')
            if v == 'MISSING':
                errors.append(f'indep line {lineno}: Recall@50 MISSING')
            elif not 0.0 <= float(v) <= 100.0:
                errors.append(f'indep line {lineno}: Recall@50={v} out of range')
        else:
            for f, lo, hi in (('qps', 0.0, 1e6), ('rss', 1024.0, 1e12)):
                raw = m.group(f)
                if raw == 'MISSING':
                    errors.append(f'indep line {lineno}: {f} MISSING')
                elif not lo < float(raw) <= hi:
                    errors.append(f'indep line {lineno}: {f}={raw} out of range')
    return keep


def main():
    errors = []
    lines = ['=== v14 4-way production analysis (rebuilt binaries) ===']
    summary = RUN / 'v14_prod_summary.txt'
    formal, warm, recalls, vals = {}, {}, [], {}
    if not summary.exists():
        errors.append(f'missing summary file: {summary}')
    else:
        formal, warm, recalls = parse_summary(
            summary.read_text(encoding='utf-8'), errors)
        validate_structure(formal, warm, errors)
        vals = check_fields(formal, errors)

    indep_lines = []
    indep = RUN / 'v14_indep_summary.txt'
    if indep.exists():
        indep_lines = check_indep(
            indep.read_text(encoding='utf-8'), errors)

    if errors:
        print('\n'.join(lines))
        print(f'INTEGRITY FAILURE: {len(errors)} problem(s) in {summary} -- '
              f'no formal conclusions; v14_prod_analysis.out NOT written')
        for e in errors:
            print(f'  - {e}', file=sys.stderr)
        return 1

    schemes = {}
    for (name, rnd), v in vals.items():
        scheme, tier = classify(name)
        schemes.setdefault((scheme, tier), {}).setdefault(name, {})[rnd] = v

    lines.append(
        f'integrity: {len(formal)}/{4 * 2 * ROUNDS} formal records OK '
        f'(4 schemes x 2 tiers x {ROUNDS} rounds, unique rc=0; '
        f'{len(warm)}/{len({n for n, _ in formal})} warmups rc=0; '
        f'qps/avg/p95/p99/rss present, finite, in range; '
        f'paired round sets identical)')

    if recalls:
        lines.append(
            '--- recall gates (context; qualification ran in run_v14_prod.sh) ---')
        lines.extend('  ' + r for r in recalls)

    for (scheme, tier), cfgs in sorted(schemes.items()):
        lines.append(f'--- {scheme} ({tier}) ---')
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
    for tier in TIERS:
        for a, b in PAIRS:
            ca, cb = schemes.get((a, tier)), schemes.get((b, tier))
            if not ca or not cb:
                continue
            ra, rb = list(ca.values())[0], list(cb.values())[0]
            if set(ra) != set(rb):
                print('=== v14 4-way production analysis (rebuilt binaries) ===')
                print(f'INTEGRITY FAILURE: paired {tier} {a}/{b} round sets '
                      f'differ ({sorted(ra)} vs {sorted(rb)}); refusing '
                      f'intersection pairing')
                print(f'  - paired {tier} {a}/{b} round sets differ '
                      f'({sorted(ra)} vs {sorted(rb)}); refusing '
                      f'intersection pairing', file=sys.stderr)
                return 1
            rounds = sorted(ra)
            ratios = [ra[r]['qps'] / rb[r]['qps'] for r in rounds]
            lines.append(
                f'{tier} {a}/{b}: '
                f'{" ".join(f"{x:.3f}" for x in ratios)} '
                f'| med x{median(ratios):.3f}')

    if indep_lines:
        lines.append(
            '--- independent query set (cohere 10m test split, 900 queries '
            'after removing 100 exact members of the 1M base; disjoint from '
            'tuning set and base; GT = exact kFlat scan, key-mapped numpy '
            'brute-force-verified) ---')
        lines.extend('  ' + l for l in indep_lines)
    else:
        lines.append('independent query set: not run yet '
                     '(independent validation not completed)')

    out = '\n'.join(lines)
    print(out)
    (RUN / 'v14_prod_analysis.out').write_text(out + '\n', encoding='utf-8')
    print(f'\nwrote {RUN / "v14_prod_analysis.out"}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

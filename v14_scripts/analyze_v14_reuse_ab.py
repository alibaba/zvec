#!/usr/bin/env python3
"""v14/v15 review: analyze the reuse OFF/ON production A/B with an
integrity gate before any conclusion.

Consumes v14_reuse_ab_summary.txt (run_v14_reuse_ab.sh): 'bench <tag> ...'
records (8 configs = int_np293 x {t1,t4} x {tuning,indep} x {off,on};
one warmup each + 5 rotated rounds = 40 formal records) and
'ab recall <cfg> ...' records (the 4 t1 configs). rrf_np180 is excluded
from the A/B: the engine rejects rabitq_reuse_coarse_dot when a refiner
is configured (ivf_index.cc "RaBitQ scan options require Turbo IVF
without refiner or bf_pks", ret=-12 on every query). The gate requires
the full unique rc=0 grid with finite in-range fields (p95 <= p99), no
duplicates/missing rounds/FAILED/unparseable lines, and identical paired
round sets for on/off -- any violation exits non-zero and suppresses
v14_reuse_ab_analysis.out.

Per-query .knn validation (strict): the set of query ids must equal the
GT query set exactly; every query must return exactly 50 ids, all unique;
every line must parse; a query may appear in only one shard (duplicate
shards or mixed-run remnants are rejected). Recall is recomputed as an
integer hit total and cross-checked against the tool-reported value at
the tool's printed precision (4 decimals, i.e. <= 5e-5pp) -- no wide
tolerance that could mask the 1-2 hit changes under discussion.

Beyond aggregate medians and per-round paired ON/OFF QPS ratios (same
round, same rotation slot), the recall section compares the per-query
t*.knn logs of each off/on pair: how many queries return an identical
ordered id list, the per-query GT@50 hit delta (improved / degraded /
unchanged, max delta) and the integer hit totals.

Writes v14_reuse_ab_analysis.out. Usage: python3 analyze_v14_reuse_ab.py
[RUN_DIR]
"""
import os
import re
import sys
from pathlib import Path
from statistics import median

ROUNDS = 5
RUN = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent
SUMMARY = RUN / 'v14_reuse_ab_summary.txt'

SCHEMES = ('int_np293',)
TIERS = ('t1', 't4')
QSETS = ('tuning', 'indep')
# Overridable for the self-contained test suite; defaults are the real
# production GT files.
GT_FILES = {
    'tuning': Path(os.environ.get(
        'V14_AB_GT_TUNING',
        '/home/xingrui/bench/dataset/cohere/1m/neighbors.txt')),
    'indep': Path(os.environ.get(
        'V14_AB_GT_INDEP',
        '/home/xingrui/bench/integrate_rabitq_in_hnsw/v14_runs/indep/'
        'neighbors_indep_q10m_900.txt'))}
NQ = {'tuning': 1000, 'indep': 900}
KNN_TOPK = 50

FIELDS = ('qps', 'avg', 'p95', 'p99', 'rss', 'load')
BOUNDS = {'qps': (0.0, 1e6, True), 'avg': (0.0, 1e5, True),
          'p95': (0.0, 1e5, True), 'p99': (0.0, 1e5, True),
          'rss': (1024.0, 1e12, True), 'load': (0.0, 1e5, False)}
CFG_RE = re.compile(
    r'^v14_ab_(?P<scheme>int_np293|rrf_np180)_(?P<tier>t[14])_'
    r'(?P<qset>tuning|indep)_(?P<reuse>off|on)$')
LINE = re.compile(
    r'^bench (?P<tag>\S+) rc=(?P<rc>\d+) qps=(?P<qps>[0-9.]+|MISSING) '
    r'avg_ms=(?P<avg>[0-9.]+|MISSING) p95_ms=(?P<p95>[0-9.]+|MISSING) '
    r'p99_ms=(?P<p99>[0-9.]+|MISSING) rss_kb=(?P<rss>[0-9]+|MISSING) '
    r'loadavg=(?P<load>[0-9.]+)$')
AB_RECALL = re.compile(
    r'^ab recall (?P<cfg>\S+) rc=(?P<rc>\d+) '
    r'Recall@50: (?P<v>[0-9.]+|MISSING)$')
KNN_RE = re.compile(r'^query\[(\d+)\]\s+key\[(\d+)\]')


def parse_summary(text, errors):
    formal, warm, recalls = {}, {}, {}
    for lineno, line in enumerate(text.splitlines(), 1):
        s = line.strip()
        m = LINE.match(s)
        if m:
            tag = m.group('tag')
            rec = {'tag': tag, 'line': lineno, 'rc': int(m.group('rc')),
                   'raw': {f: m.group(f) for f in FIELDS}}
            if tag.startswith('warmup_'):
                name = tag[len('warmup_'):]
                if tag in warm:
                    errors.append(f'line {lineno}: duplicate warmup {tag}')
                warm[tag] = rec
            else:
                rm = re.search(r'_r(\d+)$', tag)
                if not rm:
                    errors.append(f'line {lineno}: bench tag {tag} carries '
                                  f'no round suffix')
                    continue
                name, rnd = tag[:rm.start()], int(rm.group(1))
                if (name, rnd) in formal:
                    errors.append(f'line {lineno}: duplicate (config, round) '
                                  f'{tag}')
                formal[(name, rnd)] = rec
            if not CFG_RE.match(name):
                errors.append(f'line {lineno}: unrecognized config {name}')
            continue
        m = AB_RECALL.match(s)
        if m:
            cfg = m.group('cfg')
            if cfg in recalls:
                errors.append(f'line {lineno}: duplicate ab recall {cfg}')
            recalls[cfg] = {'rc': int(m.group('rc')), 'v': m.group('v'),
                            'line': lineno}
            continue
        if s.startswith('bench '):
            errors.append(f'line {lineno}: unparseable bench line: {s}')
        elif s.startswith('ab recall'):
            errors.append(f'line {lineno}: unparseable ab recall line: {s}')
        elif s.startswith('FAILED:'):
            errors.append(f'line {lineno}: aborted run: {s}')
    return formal, warm, recalls


def validate_structure(formal, warm, recalls, errors):
    expected = [f'v14_ab_{s}_{t}_{q}_{r}'
                for s in SCHEMES for t in TIERS for q in QSETS
                for r in ('off', 'on')]
    by_cfg = {}
    for (name, rnd), rec in sorted(formal.items()):
        if rec['rc'] != 0:
            errors.append(f'FAILED round: {rec["tag"]} rc={rec["rc"]}')
        by_cfg.setdefault(name, {})[rnd] = rec
    for name in expected:
        rr = by_cfg.get(name, {})
        missing = [r for r in range(1, ROUNDS + 1) if r not in rr]
        if missing:
            errors.append(f'missing rounds: {name} lacks '
                          f'{", ".join(f"r{r}" for r in missing)}')
        extra = sorted(r for r in rr if not 1 <= r <= ROUNDS)
        if extra:
            errors.append(f'unexpected rounds: {name} has '
                          f'{", ".join(f"r{r}" for r in extra)}')
    for name in sorted(set(by_cfg) - set(expected)):
        errors.append(f'unexpected config: {name}')
    n = sum(len(rr) for rr in by_cfg.values())
    if n != len(expected) * ROUNDS:
        errors.append(f'formal record count {n} != {len(expected) * ROUNDS} '
                      f'({len(expected)} configs x {ROUNDS} rounds)')
    for tag, rec in sorted(warm.items()):
        name = tag[len('warmup_'):]
        if rec['rc'] != 0:
            errors.append(f'FAILED warmup: {tag} rc={rec["rc"]}')
        if name not in by_cfg:
            errors.append(f'unexpected warmup {tag}: no matching formal config')
    for name in expected:
        if f'warmup_{name}' not in warm:
            errors.append(f'missing warmup record: warmup_{name}')
    for cfg, rec in sorted(recalls.items()):
        if not CFG_RE.match(cfg):
            errors.append(f'unrecognized recall config {cfg}')
            continue
        parts = CFG_RE.match(cfg)
        if parts['tier'] != 't1':
            errors.append(f'recall config {cfg} is not a t1 config')
        if rec['rc'] != 0:
            errors.append(f'FAILED recall: {cfg} rc={rec["rc"]}')
        v = rec['v']
        if v == 'MISSING':
            errors.append(f'recall {cfg}: Recall@50 MISSING')
        elif not 0.0 < float(v) <= 100.0:
            errors.append(f'recall {cfg}: Recall@50={v} out of range')
    for s in SCHEMES:
        for q in QSETS:
            for r in ('off', 'on'):
                cfg = f'v14_ab_{s}_t1_{q}_{r}'
                if cfg not in recalls:
                    errors.append(f'missing recall record: {cfg}')


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
                errors.append(f'field out of range: {rec["tag"]} {f}={raw}')
                continue
            v[f] = x
        if 'p95' in v and 'p99' in v and v['p95'] > v['p99']:
            errors.append(f'field inconsistent: {rec["tag"]} '
                          f'p95={rec["raw"]["p95"]} > p99={rec["raw"]["p99"]}')
        vals[(name, rnd)] = v
    return vals


def load_gt(qset):
    gt = {}
    with GT_FILES[qset].open(encoding='utf-8') as f:
        for line in f:
            idx, _, keys = line.partition(';')
            gt[int(idx)] = {int(k) for k in keys.split()[:50]}
    if len(gt) != NQ[qset]:
        sys.exit(f'GT {qset}: {len(gt)} queries, expected {NQ[qset]}')
    return gt


def load_knn(cfg, gt, errors):
    """Strictly validate and load one config's per-query .knn logs.

    Returns {qid: [key, ...]} with exactly KNN_TOPK unique keys per query
    and the query-id set equal to the GT query set, or None when the logs
    for this config are unusable (details are appended to errors).
    """
    d = RUN / f'{cfg}.recall'
    if not d.is_dir():
        errors.append(f'{cfg}: missing RecallLogDir {d.name}')
        return None
    shards = sorted(d.glob('t*.knn'))
    if not shards:
        errors.append(f'{cfg}: no t*.knn shards in {d.name}')
        return None
    qs = {}
    qid_shard = {}
    bad = 0
    for p in shards:
        shard = p.name
        for lineno, line in enumerate(
                p.read_text(encoding='utf-8').splitlines(), 1):
            s = line.strip()
            if not s:
                continue
            m = KNN_RE.match(s)
            if not m:
                bad += 1
                if bad <= 5:
                    errors.append(f'{cfg}/{shard}:{lineno}: unparseable '
                                  f'line: {s[:60]}')
                continue
            q, k = int(m.group(1)), int(m.group(2))
            # Each query legitimately spans KNN_TOPK lines in ONE shard; a
            # second shard carrying the same query means a duplicate shard
            # file or remnants of a previous run mixed into this one.
            if q in qid_shard and qid_shard[q] != shard:
                bad += 1
                if bad <= 5:
                    errors.append(f'{cfg}/{shard}:{lineno}: query {q} already '
                                  f'logged by {qid_shard[q]} (duplicate shard '
                                  f'or mixed-run data)')
                continue
            qid_shard[q] = shard
            qs.setdefault(q, []).append(k)
    if bad > 5:
        errors.append(f'{cfg}: {bad} per-query log problems in total')
    missing = sorted(set(gt) - set(qs))
    if missing:
        errors.append(f'{cfg}: {len(missing)} query id(s) missing from .knn '
                      f'(first: {missing[:5]})')
    extra = sorted(set(qs) - set(gt))
    if extra:
        errors.append(f'{cfg}: {len(extra)} unexpected query id(s) not in GT '
                      f'(first: {extra[:5]})')
    for q in sorted(set(qs) & set(gt)):
        ks = qs[q]
        if len(ks) != KNN_TOPK:
            errors.append(f'{cfg}: query {q} returned {len(ks)} results, '
                          f'expected {KNN_TOPK}')
        elif len(set(ks)) != KNN_TOPK:
            errors.append(f'{cfg}: query {q} returned '
                          f'{KNN_TOPK - len(set(ks))} duplicate id(s)')
    if missing or extra or bad:
        return None
    return qs


def knn_hits(qs, gt):
    return sum(len(gt[q] & set(qs[q])) for q in qs)


def main():
    errors = []
    if not SUMMARY.exists():
        sys.exit(f'missing summary file: {SUMMARY}')
    formal, warm, recalls = parse_summary(
        SUMMARY.read_text(encoding='utf-8'), errors)
    validate_structure(formal, warm, recalls, errors)
    vals = check_fields(formal, errors)

    knn_data = {}
    for s in SCHEMES:
        for q in QSETS:
            gt = None
            for r in ('off', 'on'):
                cfg = f'v14_ab_{s}_t1_{q}_{r}'
                if cfg not in recalls:
                    continue
                if gt is None:
                    gt = load_gt(q)
                qs = load_knn(cfg, gt, errors)
                if qs is None:
                    continue
                hits = knn_hits(qs, gt)
                r_knn = 100.0 * hits / (KNN_TOPK * len(gt))
                v_raw = recalls[cfg]['v']
                if v_raw != 'MISSING':
                    # Tool prints 4 decimals: accept only a recomputation
                    # that rounds/truncates to the reported figure (<=5e-5pp).
                    if abs(r_knn - float(v_raw)) > 5e-5 + 1e-9:
                        errors.append(
                            f'recall mismatch for {cfg}: .knn={r_knn:.6f} '
                            f'({hits}/{KNN_TOPK * len(gt)} hits) vs '
                            f'tool={v_raw}')
                knn_data[cfg] = (qs, gt, hits, r_knn)

    if errors:
        print(f'INTEGRITY FAILURE: {len(errors)} problem(s) in {SUMMARY} -- '
              f'no formal conclusions; v14_reuse_ab_analysis.out NOT written')
        for e in errors:
            print(f'  - {e}', file=sys.stderr)
        return 1

    lines = ['=== v14 reuse OFF/ON production A/B analysis ===']
    n_cfg = len(SCHEMES) * len(TIERS) * len(QSETS) * 2
    lines.append(f'integrity: {len(formal)}/{n_cfg * ROUNDS} formal records OK '
                 f'({n_cfg} configs x {ROUNDS} rounds, unique rc=0; '
                 f'{len(warm)}/{n_cfg} warmups rc=0; fields finite, in range; '
                 f'paired on/off round sets identical; per-query .knn strict: '
                 f'exact query-id set, {KNN_TOPK} unique ids per query, one '
                 f'shard per query, recall cross-checked at tool precision)')

    by_cfg = {}
    for (name, rnd), v in vals.items():
        by_cfg.setdefault(name, {})[rnd] = v

    lines.append('--- QPS medians (p95/p99/RSS/load medians in parentheses) ---')
    for s in SCHEMES:
        for q in QSETS:
            for t in TIERS:
                for r in ('off', 'on'):
                    name = f'v14_ab_{s}_{t}_{q}_{r}'
                    rr = by_cfg[name]
                    rounds = sorted(rr)
                    qps_str = ' '.join(f'{rr[x]["qps"]:.1f}' for x in rounds)
                    qps_med = median(rr[x]['qps'] for x in rounds)
                    p95_med = median(rr[x]['p95'] for x in rounds)
                    p99_med = median(rr[x]['p99'] for x in rounds)
                    rss_med = median(rr[x]['rss'] for x in rounds)
                    load_med = median(rr[x]['load'] for x in rounds)
                    lines.append(
                        f'{name}: rounds={qps_str} '
                        f'| qps med={qps_med:.1f} '
                        f'p95 med={p95_med:.2f}ms '
                        f'p99 med={p99_med:.2f}ms '
                        f'rss med={rss_med / 1024 / 1024:.2f}GB '
                        f'load med={load_med:.2f}')

    lines.append('--- per-round paired QPS ratios ON/OFF (same round, same rotation) ---')
    for s in SCHEMES:
        for q in QSETS:
            for t in TIERS:
                off = by_cfg[f'v14_ab_{s}_{t}_{q}_off']
                on = by_cfg[f'v14_ab_{s}_{t}_{q}_on']
                rounds = sorted(off)
                ratios = [on[r]['qps'] / off[r]['qps'] for r in rounds]
                lines.append(
                    f'{s} {t} {q}: '
                    f'{" ".join(f"{x:.3f}" for x in ratios)} '
                    f'| med x{median(ratios):.3f}')

    lines.append('--- recall: aggregate + per-query comparison (t1 configs) ---')
    for s in SCHEMES:
        for q in QSETS:
            off_cfg = f'v14_ab_{s}_t1_{q}_off'
            on_cfg = f'v14_ab_{s}_t1_{q}_on'
            v_off = float(recalls[off_cfg]['v'])
            v_on = float(recalls[on_cfg]['v'])
            qs_off, gt, hits_off, r_off = knn_data[off_cfg]
            qs_on, _, hits_on, r_on = knn_data[on_cfg]
            total = KNN_TOPK * NQ[q]
            lines.append(f'{s} {q}: Recall@50 off={v_off:.4f} on={v_on:.4f} '
                         f'(delta {v_on - v_off:+.4f}pp); GT@50 hit totals '
                         f'off={hits_off}/{total} on={hits_on}/{total} '
                         f'({hits_on - hits_off:+d})')
            same_order = sum(1 for x in qs_off if qs_off[x] == qs_on[x])
            same_set = sum(1 for x in qs_off
                           if set(qs_off[x]) == set(qs_on[x]))
            hits_off_q = {x: len(gt[x] & set(qs_off[x])) for x in qs_off}
            hits_on_q = {x: len(gt[x] & set(qs_on[x])) for x in qs_on}
            improved = sum(1 for x in hits_off_q if hits_on_q[x] > hits_off_q[x])
            degraded = sum(1 for x in hits_off_q if hits_on_q[x] < hits_off_q[x])
            max_delta = max((abs(hits_on_q[x] - hits_off_q[x])
                             for x in hits_off_q), default=0)
            lines.append(
                f'  per-query: identical ordered ids {same_order}/{NQ[q]}, '
                f'identical id sets {same_set}/{NQ[q]}; GT@50 hits '
                f'improved {improved}, degraded {degraded}, unchanged '
                f'{NQ[q] - improved - degraded}, max |delta| {max_delta}')
            lines.append(f'  recomputed from .knn: off={r_off:.4f} '
                         f'on={r_on:.4f} (tool: {v_off:.4f}/{v_on:.4f})')

    out = '\n'.join(lines)
    print(out)
    (RUN / 'v14_reuse_ab_analysis.out').write_text(out + '\n',
                                                   encoding='utf-8')
    print(f"\nwrote {RUN / 'v14_reuse_ab_analysis.out'}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

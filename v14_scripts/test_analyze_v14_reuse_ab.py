#!/usr/bin/env python3
"""Self-contained smoke test for analyze_v14_reuse_ab.py.

Builds a scratch run dir (next to this script, never /tmp; TMPDIR points
inside it) with fully SYNTHETIC data: GT files (qid;50 gt keys) and
16-shard t*.knn logs (query[q]\\tkey[k], dist[...]) for the 4 t1 configs,
plus a structurally complete v14_reuse_ab_summary.txt (8 configs x 5
rounds + 8 warmups + 4 recall records). GT keys are q*100000+100+i,
filler keys q*100000+1000+i, so hits are exact integers: base 47/50 per
query, ON config flips tuning q7/q13 (+1) and indep q5/q9 (+1) / q21
(-1) -> tuning 47000->47002/50000 (94.0000->94.0040), indep
42300->42301/45000 (94.0000->94.0022), improved 2, degraded 0/1,
identical ordered ids 998/1000 and 897/900. The positive case asserts
rc=0, the written .out and the key report strings; twelve negative cases
(knn defects on query 100 of v14_ab_int_np293_t1_indep_off: missing
query / wrong qid / 49 results / 51 results / duplicate ids / duplicate
shard / unparseable line; recall value mismatch; dropped round; MISSING
recall; duplicate round; failed warmup) must exit non-zero with the
expected stderr keywords and suppress the .out. Scratch is removed when
all cases pass and kept for inspection otherwise.

Usage: python3 test_analyze_v14_reuse_ab.py
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ANALYZER = HERE / 'analyze_v14_reuse_ab.py'
SCRATCH = HERE / 'test_ab_tmp'

QSETS = ('tuning', 'indep')
TIERS = ('t1', 't4')
ROUNDS = 5
NSHARD = 16
QIDS = {'tuning': range(1, 1001), 'indep': range(1, 901)}
BASE_HITS = 47
DELTAS = {'tuning': {7: 1, 13: 1}, 'indep': {5: 1, 9: 1, 21: -1}}
RECALL = {('tuning', 'off'): 94.0000, ('tuning', 'on'): 94.0040,
          ('indep', 'off'): 94.0000, ('indep', 'on'): 94.0022}
BASE_QPS = {('t1', 'tuning'): 130.0, ('t4', 'tuning'): 530.0,
            ('t1', 'indep'): 41.0, ('t4', 'indep'): 165.0}
DEFECT_CFG = 'v14_ab_int_np293_t1_indep_off'


def gt_keys(q):
    return [q * 100000 + 100 + i for i in range(50)]


def result_keys(q, hits):
    return gt_keys(q)[:hits] + [q * 100000 + 1000 + i
                                for i in range(50 - hits)]


def write_gt(path, qset):
    lines = []
    for q in QIDS[qset]:
        lines.append(f"{q};{' '.join(str(k) for k in gt_keys(q))}")
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def write_knn(path, qset, deltas=None):
    deltas = deltas or {}
    shards = {i: [] for i in range(NSHARD)}
    for q in QIDS[qset]:
        for k in result_keys(q, BASE_HITS + deltas.get(q, 0)):
            shards[q % NSHARD].append(f'query[{q}]\tkey[{k}], dist[0.5]')
    path.mkdir(parents=True)
    for i in range(NSHARD):
        (path / f't{i}.knn').write_text(
            '\n'.join(shards[i]) + '\n', encoding='utf-8')


def mutate(d, kind):
    """Corrupt query 100 (shard t4) of the DEFECT_CFG knn dir."""
    t4 = d / 't4.knn'
    lines = t4.read_text(encoding='utf-8').splitlines()
    q100 = [l for l in lines if l.startswith('query[100]\t')]
    other = [l for l in lines if not l.startswith('query[100]\t')]
    if kind == 'missing_query':
        t4.write_text('\n'.join(other) + '\n', encoding='utf-8')
    elif kind == 'wrong_qid':
        t4.write_text(
            '\n'.join(other + [l.replace('query[100]', 'query[99999]', 1)
                               for l in q100]) + '\n',
            encoding='utf-8')
    elif kind == 'short_49':
        t4.write_text('\n'.join(other + q100[:-1]) + '\n', encoding='utf-8')
    elif kind == 'long_51':
        extra = 100 * 100000 + 1000 + 50
        t4.write_text(
            '\n'.join(other + q100 + [f'query[100]\tkey[{extra}], dist[0.5]'])
            + '\n', encoding='utf-8')
    elif kind == 'dup_ids':
        t4.write_text('\n'.join(other + [q100[0]] * 50) + '\n',
                      encoding='utf-8')
    elif kind == 'dup_shard':
        shutil.copyfile(t4, d / 't16.knn')
    elif kind == 'bad_line':
        t4.write_text('\n'.join(lines + ['garbage line']) + '\n',
                      encoding='utf-8')
    else:
        raise ValueError(kind)


def bench_line(tag, qps, rc=0):
    avg = 1000.0 / qps
    return (f'bench {tag} rc={rc} qps={qps:.1f} avg_ms={avg:.3f} '
            f'p95_ms={avg * 1.5:.3f} p99_ms={avg * 2.0:.3f} '
            f'rss_kb=840000 loadavg=1.50')


def build_summary():
    lines = ['=== v14 reuse OFF/ON production A/B ===']
    for q in QSETS:
        for r in ('off', 'on'):
            cfg = f'v14_ab_int_np293_t1_{q}_{r}'
            lines.append(f'ab recall {cfg} rc=0 Recall@50: '
                         f'{RECALL[(q, r)]:.4f}')
    for t in TIERS:
        for q in QSETS:
            for r in ('off', 'on'):
                name = f'v14_ab_int_np293_{t}_{q}_{r}'
                lines.append(f'--- ab warmup {name} ---')
                lines.append(bench_line(
                    f'warmup_{name}', BASE_QPS[(t, q)] - 5.0))
    for rnd in range(1, ROUNDS + 1):
        lines.append(f'=== ab ROUND r{rnd} ===')
        for t in TIERS:
            for q in QSETS:
                for r in ('off', 'on'):
                    name = f'v14_ab_int_np293_{t}_{q}_{r}'
                    qps = BASE_QPS[(t, q)] + 0.5 * (rnd - 1) \
                        + (8.0 if r == 'on' else 0.0)
                    lines.append(f'--- ab r{rnd} {name}_r{rnd} ---')
                    lines.append(bench_line(f'{name}_r{rnd}', qps))
    lines.append('=== v14 reuse OFF/ON production A/B DONE ===')
    return lines


def run_case(name, summary, defect=None, expect_rc0=False, must=None,
             must_stderr=None, must_not=None):
    d = SCRATCH / name
    if d.exists():
        shutil.rmtree(d)
    (d / 'tmp').mkdir(parents=True)
    (d / 'v14_reuse_ab_summary.txt').write_text(
        '\n'.join(summary) + '\n', encoding='utf-8')
    write_gt(d / 'gt_tuning.txt', 'tuning')
    write_gt(d / 'gt_indep.txt', 'indep')
    for q in QSETS:
        for r in ('off', 'on'):
            cfg = f'v14_ab_int_np293_t1_{q}_{r}'
            write_knn(d / f'{cfg}.recall', q,
                      DELTAS[q] if r == 'on' else None)
    if defect:
        mutate(d / f'{DEFECT_CFG}.recall', defect)
    env = dict(os.environ, TMPDIR=str(d / 'tmp'),
               V14_AB_GT_TUNING=str(d / 'gt_tuning.txt'),
               V14_AB_GT_INDEP=str(d / 'gt_indep.txt'))
    p = subprocess.run([sys.executable, str(ANALYZER), str(d)],
                       capture_output=True, text=True, env=env)
    out, err = p.stdout, p.stderr
    problems = []
    if expect_rc0 and p.returncode != 0:
        problems.append(f'rc={p.returncode}, expected 0; stderr:\n{err}')
    if not expect_rc0 and p.returncode == 0:
        problems.append('rc=0, expected non-zero')
    outfile = d / 'v14_reuse_ab_analysis.out'
    if expect_rc0:
        if not outfile.exists():
            problems.append('v14_reuse_ab_analysis.out not written')
        else:
            otext = outfile.read_text(encoding='utf-8')
            if 'integrity: 40/40 formal records OK' not in otext:
                problems.append('integrity line missing from .out')
    else:
        if outfile.exists():
            problems.append('.out written despite failure')
    for kw in (must or []):
        if kw not in out and kw not in err:
            problems.append(f'missing keyword: {kw!r}')
    for kw in (must_stderr or []):
        if kw not in err:
            problems.append(f'missing stderr keyword: {kw!r}')
    for kw in (must_not or []):
        if kw in out or kw in err:
            problems.append(f'unexpected keyword: {kw!r}')
    return problems


def main():
    if SCRATCH.exists():
        shutil.rmtree(SCRATCH)
    SCRATCH.mkdir(parents=True)
    base = build_summary()

    cases = [
        ('complete', dict(
            summary=base, expect_rc0=True,
            must=['integrity: 40/40 formal records OK',
                  'qps med=131.0', 'qps med=139.0', 'qps med=42.0',
                  'qps med=50.0', 'med x1.061', 'med x1.190',
                  'rss med=0.80GB',
                  'Recall@50 off=94.0000 on=94.0040',
                  'Recall@50 off=94.0000 on=94.0022',
                  'GT@50 hit totals off=47000/50000 on=47002/50000 (+2)',
                  'GT@50 hit totals off=42300/45000 on=42301/45000 (+1)',
                  'identical ordered ids 998/1000',
                  'identical ordered ids 897/900',
                  'improved 2, degraded 0, unchanged 998',
                  'improved 2, degraded 1, unchanged 897',
                  'recomputed from .knn: off=94.0000 on=94.0040'],
            must_not=['INTEGRITY FAILURE'])),
        ('missing_query', dict(
            summary=base, defect='missing_query',
            must_stderr=['v14_ab_int_np293_t1_indep_off: 1 query id(s) '
                         'missing from .knn (first: [100])'])),
        ('wrong_qid', dict(
            summary=base, defect='wrong_qid',
            must_stderr=['v14_ab_int_np293_t1_indep_off: 1 unexpected '
                         'query id(s) not in GT (first: [99999])'])),
        ('short_49', dict(
            summary=base, defect='short_49',
            must_stderr=['query 100 returned 49 results, expected 50'])),
        ('long_51', dict(
            summary=base, defect='long_51',
            must_stderr=['query 100 returned 51 results, expected 50'])),
        ('dup_ids', dict(
            summary=base, defect='dup_ids',
            must_stderr=['query 100 returned 49 duplicate id(s)'])),
        ('dup_shard', dict(
            summary=base, defect='dup_shard',
            must_stderr=['duplicate shard or mixed-run data'])),
        ('bad_line', dict(
            summary=base, defect='bad_line',
            must_stderr=['unparseable line: garbage line'])),
        ('recall_mismatch', dict(
            summary=[l.replace('Recall@50: 94.0000', 'Recall@50: 94.0100')
                     if l.startswith('ab recall v14_ab_int_np293_t1_indep_off ')
                     else l for l in base],
            must_stderr=['recall mismatch for v14_ab_int_np293_t1_indep_off: '
                         '.knn=94.000000 (42300/45000 hits) vs '
                         'tool=94.0100'])),
        ('missing_round', dict(
            summary=[l for l in base
                     if not l.startswith(
                         'bench v14_ab_int_np293_t4_indep_on_r3 ')],
            must_stderr=['missing rounds: v14_ab_int_np293_t4_indep_on '
                         'lacks r3'])),
        ('missing_recall_value', dict(
            summary=[l.replace('Recall@50: 94.0000', 'Recall@50: MISSING')
                     if l.startswith('ab recall v14_ab_int_np293_t1_indep_off ')
                     else l for l in base],
            must_stderr=['recall v14_ab_int_np293_t1_indep_off: '
                         'Recall@50 MISSING'])),
        ('duplicate_round', dict(
            summary=base + [bench_line(
                'v14_ab_int_np293_t1_tuning_off_r2', 999.0)],
            must_stderr=['duplicate (config, round) '
                         'v14_ab_int_np293_t1_tuning_off_r2'])),
        ('failed_warmup', dict(
            summary=[bench_line('warmup_v14_ab_int_np293_t1_tuning_off',
                                130.0, rc=1)
                     if l.startswith('bench warmup_v14_ab_int_np293_t1_'
                                     'tuning_off ')
                     else l for l in base],
            must_stderr=['FAILED warmup: warmup_v14_ab_int_np293_t1_'
                         'tuning_off rc=1'])),
    ]

    failures = 0
    for name, kw in cases:
        problems = run_case(name, **kw)
        if problems:
            failures += 1
            print(f'FAIL {name}')
            for p in problems:
                print(f'    {p}')
        else:
            print(f'PASS {name}')

    if failures:
        print(f'{failures}/{len(cases)} case(s) FAILED; scratch kept at '
              f'{SCRATCH}')
        return 1
    shutil.rmtree(SCRATCH)
    print(f'all {len(cases)} cases passed; scratch removed')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

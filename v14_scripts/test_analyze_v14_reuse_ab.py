#!/usr/bin/env python3
"""Smoke test for analyze_v14_reuse_ab.py.

Builds a scratch run dir (next to this script, never /tmp; TMPDIR points
inside it) with a synthetic but structurally complete v14_reuse_ab_summary.txt
(8 configs x 5 rounds + 8 warmups + 4 recall records, real measured recall
values) and per-query .knn fixture dirs copied from the REAL v12 recall logs
(off/on pairs are copies, so the per-query comparison must report
identical ids and zero recall delta). rrf_np180 is not part of the A/B
(reuse is engine-unsupported with a refiner). Runs the analyzer as a
subprocess and asserts rc=0, the written .out and key output strings; then
two negative cases (a dropped round; a MISSING recall value) must exit
non-zero with the expected stderr keywords and suppress the .out. Scratch
is removed when all cases pass and kept for inspection otherwise.

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
RUNS = Path('/home/xingrui/bench/integrate_rabitq_in_hnsw/v14_runs')
V12 = Path('/home/xingrui/bench/integrate_rabitq_in_hnsw/v12_runs/prod')

SCHEMES = ('int_np293',)
TIERS = ('t1', 't4')
QSETS = ('tuning', 'indep')
ROUNDS = 5
# real measured Recall@50 of the source configs (t1, stats off)
REAL_RECALL = {('int_np293', 'tuning'): 99.002,
               ('int_np293', 'indep'): 94.4689}
# knn fixture source per (scheme, qset): real per-query recall logs
KNN_SRC = {('int_np293', 'tuning'): V12 / 'prod50_np293_t1.recall',
           ('int_np293', 'indep'): RUNS / 'v12_prod50_np293_t1_indep.recall'}
# base qps for synthetic bench lines
BASE_QPS = {('int_np293', 't1', 'tuning'): 130.0,
            ('int_np293', 't4', 'tuning'): 530.0,
            ('int_np293', 't1', 'indep'): 41.0,
            ('int_np293', 't4', 'indep'): 165.0}


def bench_line(tag, qps, rc=0):
    avg = 1000.0 / qps
    return (f'bench {tag} rc={rc} qps={qps:.1f} avg_ms={avg:.3f} '
            f'p95_ms={avg * 1.5:.3f} p99_ms={avg * 2.0:.3f} '
            f'rss_kb=840000 loadavg=1.50')


def build_summary():
    lines = ['=== v14 reuse OFF/ON production A/B ===']
    for s in SCHEMES:
        for q in QSETS:
            for r in ('off', 'on'):
                cfg = f'v14_ab_{s}_t1_{q}_{r}'
                v = REAL_RECALL[(s, q)] + (0.0001 if r == 'on' else 0.0)
                lines.append(f'ab recall {cfg} rc=0 Recall@50: {v:.4f}')
    for s in SCHEMES:
        for t in TIERS:
            for q in QSETS:
                for r in ('off', 'on'):
                    name = f'v14_ab_{s}_{t}_{q}_{r}'
                    lines.append(f'--- ab warmup {name} ---')
                    lines.append(bench_line(
                        f'warmup_{name}', BASE_QPS[(s, t, q)] - 5.0))
    for rnd in range(1, ROUNDS + 1):
        lines.append(f'=== ab ROUND r{rnd} ===')
        for s in SCHEMES:
            for t in TIERS:
                for q in QSETS:
                    for r in ('off', 'on'):
                        name = f'v14_ab_{s}_{t}_{q}_{r}'
                        qps = BASE_QPS[(s, t, q)] + 0.5 * (rnd - 1) \
                            + (8.0 if r == 'on' else 0.0)
                        lines.append(f'--- ab r{rnd} {name}_r{rnd} ---')
                        lines.append(bench_line(f'{name}_r{rnd}', qps))
    lines.append('=== v14 reuse OFF/ON production A/B DONE ===')
    return lines


def run_case(name, summary, expect_rc0=False, must=None, must_stderr=None,
             must_not=None):
    d = SCRATCH / name
    if d.exists():
        shutil.rmtree(d)
    (d / 'tmp').mkdir(parents=True)
    (d / 'v14_reuse_ab_summary.txt').write_text(
        '\n'.join(summary) + '\n', encoding='utf-8')
    for (s, q), src in KNN_SRC.items():
        for r in ('off', 'on'):
            dst = d / f'v14_ab_{s}_t1_{q}_{r}.recall'
            shutil.copytree(src, dst)
    env = dict(os.environ, TMPDIR=str(d / 'tmp'))
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
                  'Recall@50 off=94.4689 on=94.4690',
                  'Recall@50 off=99.0020 on=99.0021',
                  'identical ordered ids 900/900',
                  'identical ordered ids 1000/1000',
                  'GT@50 hits improved 0, degraded 0, unchanged 900',
                  'recomputed from .knn: off=94.4689'],
            must_not=['INTEGRITY FAILURE'])),
        ('missing_round', dict(
            summary=[l for l in base
                     if 'bench v14_ab_int_np293_t4_indep_on_r3 ' not in l],
            must_stderr=['missing rounds: v14_ab_int_np293_t4_indep_on lacks r3'])),
        ('missing_recall_value', dict(
            summary=[l.replace('Recall@50: 94.4689', 'Recall@50: MISSING')
                     if l.startswith('ab recall v14_ab_int_np293_t1_indep_off')
                     else l for l in base],
            must_stderr=['recall v14_ab_int_np293_t1_indep_off: '
                         'Recall@50 MISSING'])),
        ('duplicate_round', dict(
            summary=base + [bench_line('v14_ab_int_np293_t1_tuning_off_r2', 999.0)],
            must_stderr=['duplicate (config, round) '
                         'v14_ab_int_np293_t1_tuning_off_r2'])),
        ('failed_warmup', dict(
            summary=[bench_line('warmup_v14_ab_int_np293_t1_tuning_off', 130.0, rc=1)
                     if l.startswith('bench warmup_v14_ab_int_np293_t1_tuning_off ')
                     else l for l in base],
            must_stderr=['FAILED warmup: warmup_v14_ab_int_np293_t1_tuning_off '
                         'rc=1'])),
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
        print(f'{failures}/{len(cases)} case(s) FAILED; scratch kept at {SCRATCH}')
        return 1
    shutil.rmtree(SCRATCH)
    print(f'all {len(cases)} cases passed; scratch removed')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

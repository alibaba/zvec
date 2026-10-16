#!/usr/bin/env python3
"""Positive and negative fixtures for analyze_v13_prod.py.

Each case synthesizes a v13_prod_summary.txt (plus a holdout summary
for the positive case) in a scratch dir next to this script (never
/tmp; TMPDIR is pointed at the scratch dir), runs the analyzer as a
subprocess, and asserts the exit code, whether v13_prod_analysis.out
was written, and error keywords. Scratch dirs are removed when all
cases pass and kept for inspection when a case fails.

Usage: python3 test_analyze_v13_prod.py
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ANALYZER = HERE / 'analyze_v13_prod.py'
SCRATCH = HERE / 'test_tmp'

ROUNDS = 5
# name -> (base qps, rss_kb); qps_r = base + 0.5*(r-1)
CFG = [
    ('v12_prod50_np293_t1', 130.0, 840000),
    ('v12_prod50_np293_t4', 530.0, 3600000),
    ('v13_rrf55_np180_t1', 190.0, 3590000),
    ('v13_rrf55_np180_t4', 800.0, 3590000),
    ('v13_native_np249_t1', 175.0, 730000),
    ('v13_native_np249_t4', 730.0, 3510000),
    ('v13_natref_np200_t1', 195.0, 3500000),
    ('v13_natref_np200_t4', 870.0, 3500000),
]
RECALLS = [
    'recall v12_prod50_np293_t1 rc=0 Recall@50: 99.002',
    'recall v13_rrf55_np180_t1 rc=0 Recall@50: 99.082',
    'recall v13_native_np249_t1 rc=0 Recall@50: 99.002',
    'recall v13_natref_np200_t1 rc=0 Recall@50: 99.258',
]
HOLDOUT = [
    '=== v13 holdout np=180 N=55 natref_np=200 ===',
    'holdout recall v13_int50_np293_t1_holdout rc=0 Recall@50: 98.952',
    'holdout bench v13_int50_np293_t1_holdout rc=0 qps=127.3 '
    'rss_kb=868384 loadavg=1.50',
    'holdout recall v13_rrf55_np180_t1_holdout rc=0 Recall@50: 99.036',
    'holdout bench v13_rrf55_np180_t1_holdout rc=0 qps=188.4 '
    'rss_kb=3585536 loadavg=1.60',
    'holdout recall v13_native_np249_t1_holdout rc=0 Recall@50: 99.000',
    'holdout bench v13_native_np249_t1_holdout rc=0 qps=162.5 '
    'rss_kb=724992 loadavg=1.40',
    'holdout recall v13_natref_np200_holdout rc=0 Recall@50: 99.212',
    'holdout bench v13_natref_np200_holdout rc=0 qps=181.1 '
    'rss_kb=3504384 loadavg=1.70',
    '=== v13 holdout DONE ===',
]


def values(name, r):
    base, rss = next((b, rss) for n, b, rss in CFG if n == name)
    q = base + 0.5 * (r - 1)
    avg = round(1000.0 / q, 3)
    return {'qps': f'{q:.1f}', 'avg': f'{avg:.3f}',
            'p95': f'{avg * 1.5:.3f}', 'p99': f'{avg * 2.0:.3f}',
            'rss': str(rss), 'load': f'{1.0 + 0.1 * r:.2f}'}


def bench_line(name, r, rc=0, **over):
    v = values(name, r)
    v.update(over)
    return (f'bench {name}_r{r} rc={rc} qps={v["qps"]} avg_ms={v["avg"]} '
            f'p95_ms={v["p95"]} p99_ms={v["p99"]} rss_kb={v["rss"]} '
            f'loadavg={v["load"]}')


def warmup_line(name, rc=0, **over):
    v = values(name, 1)
    v.update(over)
    return (f'bench warmup_{name} rc={rc} qps={v["qps"]} avg_ms={v["avg"]} '
            f'p95_ms={v["p95"]} p99_ms={v["p99"]} rss_kb={v["rss"]} '
            f'loadavg={v["load"]}')


def build_summary():
    lines = ['=== v13 prod 4way np=180 N=55 natref_np=200 ===']
    lines.extend(RECALLS)
    for name, _, _ in CFG:
        lines.append(f'--- warmup {name} ---')
        lines.append(warmup_line(name))
    for r in range(1, ROUNDS + 1):
        lines.append(f'=== ROUND r{r} ===')
        for name, _, _ in CFG:
            lines.append(f'--- r{r} {name}_r{r} ---')
            lines.append(bench_line(name, r))
    lines.append('=== v13 prod 4way DONE ===')
    return lines


def drop(lines, sub):
    return [l for l in lines if sub not in l]


def replace_line(lines, sub, new):
    return [new if sub in l else l for l in lines]


def run_case(name, summary, holdout=None, expect_rc0=False,
             must=None, must_stderr=None, must_not=None):
    d = SCRATCH / name
    if d.exists():
        shutil.rmtree(d)
    (d / 'tmp').mkdir(parents=True)
    (d / 'v13_prod_summary.txt').write_text(
        '\n'.join(summary) + '\n', encoding='utf-8')
    if holdout is not None:
        (d / 'v13_holdout_summary.txt').write_text(
            '\n'.join(holdout) + '\n', encoding='utf-8')
    env = dict(os.environ, TMPDIR=str(d / 'tmp'))
    p = subprocess.run([sys.executable, str(ANALYZER), str(d)],
                       capture_output=True, text=True, env=env)
    out, err = p.stdout, p.stderr
    problems = []
    if expect_rc0 and p.returncode != 0:
        problems.append(f'rc={p.returncode}, expected 0; stderr:\n{err}')
    if not expect_rc0 and p.returncode == 0:
        problems.append('rc=0, expected non-zero')
    outfile = d / 'v13_prod_analysis.out'
    if expect_rc0:
        if not outfile.exists():
            problems.append('v13_prod_analysis.out not written')
        else:
            otext = outfile.read_text(encoding='utf-8')
            if 'integrity: 40/40 formal records OK' not in otext:
                problems.append('integrity line missing from .out')
    else:
        if outfile.exists():
            problems.append('v13_prod_analysis.out written despite failure')
        if 'qps med=' in out:
            problems.append('formal medians emitted despite failure')
        if '40/40' in out:
            problems.append('integrity OK line emitted despite failure')
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

    native_r2 = values('v13_native_np249_t1', 2)
    unparseable = (f'bench v13_native_np249_t1_r2 rc=0 qps={native_r2["qps"]} '
                   f'avg_ms={native_r2["avg"]} p99_ms={native_r2["p99"]} '
                   f'rss_kb={native_r2["rss"]} loadavg={native_r2["load"]}')
    failed_r3 = bench_line('v13_natref_np200_t4', 3, rc=1, qps='MISSING',
                           avg='MISSING', p95='MISSING', p99='MISSING',
                           rss='MISSING')
    failed_warm = warmup_line('v13_native_np249_t1', rc=1, qps='MISSING',
                              avg='MISSING', p95='MISSING', p99='MISSING',
                              rss='MISSING')

    cases = [
        ('complete', dict(
            summary=base, holdout=HOLDOUT, expect_rc0=True,
            must=['integrity: 40/40 formal records OK',
                  'qps med=191.0', 'qps med=131.0', 'qps med=871.0',
                  'rss med=3.42GB', 'rss med=0.80GB',
                  't1 rrf/int:', 'x1.458',
                  'recall v13_natref_np200_t1 rc=0 Recall@50: 99.258',
                  '调参数据的分组检查',
                  'holdout bench v13_rrf55_np180_t1_holdout rc=0 qps=188.4'],
            must_not=['INTEGRITY FAILURE', 'independent confirmation'])),
        ('missing_round', dict(
            summary=drop(base, 'bench v13_natref_np200_t4_r3 '),
            must_stderr=['missing rounds: v13_natref_np200_t4 lacks r3'])),
        ('failed_round', dict(
            summary=replace_line(base, 'bench v13_natref_np200_t4_r3 ', failed_r3),
            must_stderr=['FAILED round: v13_natref_np200_t4_r3 rc=1',
                         'field MISSING: v13_natref_np200_t4_r3 qps'])),
        ('duplicate_round', dict(
            summary=base + [bench_line('v13_rrf55_np180_t1', 4, qps='999.0')],
            must_stderr=['duplicate (config, round) record '
                         'v13_rrf55_np180_t1_r4'])),
        ('missing_field_value', dict(
            summary=replace_line(
                base, 'bench v13_rrf55_np180_t4_r5 ',
                bench_line('v13_rrf55_np180_t4', 5, qps='MISSING')),
            must_stderr=['field MISSING: v13_rrf55_np180_t4_r5 qps'])),
        ('missing_field_unparseable', dict(
            summary=replace_line(base, 'bench v13_native_np249_t1_r2 ', unparseable),
            must_stderr=['unparseable bench line',
                         'missing rounds: v13_native_np249_t1 lacks r2'])),
        ('missing_config', dict(
            summary=drop(base, 'v13_natref_np200_t4'),
            must_stderr=['missing config: scheme natref (t4) has no records',
                         'formal record count 35 != 40'])),
        ('failed_warmup', dict(
            summary=replace_line(
                base, 'bench warmup_v13_native_np249_t1 ', failed_warm),
            must_stderr=['FAILED warmup: warmup_v13_native_np249_t1 rc=1'])),
        ('missing_warmup', dict(
            summary=drop(base, 'bench warmup_v13_rrf55_np180_t4 '),
            must_stderr=['missing warmup record: warmup_v13_rrf55_np180_t4'])),
        ('unexpected_round', dict(
            summary=base + [bench_line('v12_prod50_np293_t1', 6)],
            must_stderr=['unexpected rounds: v12_prod50_np293_t1 has r6',
                         'formal record count 41 != 40'])),
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

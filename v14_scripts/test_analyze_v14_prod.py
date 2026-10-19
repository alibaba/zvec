#!/usr/bin/env python3
"""Positive and negative fixtures for analyze_v14_prod.py.

Each case synthesizes a v14_prod_summary.txt (plus a v14_indep_summary.txt
when the case exercises the independent-set section) in a scratch dir next
to this script (never /tmp; TMPDIR is pointed at the scratch dir), runs the
analyzer as a subprocess, and asserts the exit code, whether
v14_prod_analysis.out was written, and error keywords. The positive indep
fixture mirrors the real run_v14_indep.sh output byte-for-byte in shape
(gtgen header, per-config recall records, warmup_<cfg> bench records and
<cfg>_r{1,2,3} rotated formal rounds). Scratch dirs are removed when all
cases pass and kept for inspection when a case fails.

Usage: python3 test_analyze_v14_prod.py
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ANALYZER = HERE / 'analyze_v14_prod.py'
SCRATCH = HERE / 'test_tmp'

ROUNDS = 5
INDEP_ROUNDS = 3
# name -> (base qps, rss_kb); qps_r = base + 0.5*(r-1)
CFG = [
    ('v12_prod50_np293_t1', 130.0, 840000),
    ('v12_prod50_np293_t4', 530.0, 3600000),
    ('v13_rrf55_np180_t1', 190.0, 3590000),
    ('v13_rrf55_np180_t4', 800.0, 3590000),
    ('v13_native_np249_t1', 175.0, 730000),
    ('v13_native_np249_t4', 730.0, 3510000),
    ('v14_natref_np170_top55_t1', 195.0, 3472000),
    ('v14_natref_np170_top55_t4', 870.0, 3472000),
]
RECALLS = [
    'recall v12_prod50_np293_t1 rc=0 Recall@50: 99.002',
    'recall v13_rrf55_np180_t1 rc=0 Recall@50: 99.082',
    'recall v13_native_np249_t1 rc=0 Recall@50: 99.002',
    'recall v14_natref_np170_top55_t1 rc=0 Recall@50: 99.058',
]
# frozen t1 configs -> real measured independent-set values (run_v14_indep.sh)
INDEP_CFGS = [
    # name, Recall@50, warmup qps/rss/load, r1..r3 qps, rss, load
    ('v12_prod50_np293_t1_indep', 94.4689, 42.9, 835192, 5.86,
     [(42.4, 835576, 2.47), (40.8, 835448, 1.51), (41.4, 835444, 1.47)]),
    ('v13_rrf55_np180_t1_indep', 91.4422, 62.4, 3383640, 4.14,
     [(59.1, 3383876, 2.44), (59.9, 3384016, 1.72), (58.2, 3384188, 2.71)]),
    ('v13_native_np249_t1_indep', 92.74, 85.0, 719460, 3.07,
     [(79.0, 719332, 2.10), (75.1, 719460, 1.58), (78.4, 719208, 1.62)]),
    ('v14_natref_np170_top55_t1_indep', 89.6778, 111.2, 3267884, 2.33,
     [(104.3, 3268236, 1.96), (99.9, 3268128, 1.47), (102.6, 3267980, 1.77)]),
]
TS = '2026-10-16T18:30:15+08:00'


def build_indep():
    lines = [
        '=== v14 independent GT generation 2026-10-16T18:28:36+08:00 '
        'commit=423423c1 ===',
        '--- gtgen flat100 2026-10-16T18:28:36+08:00 ---',
        'gtgen_recall v14_gtgen_flat100 rc=0 Recall@100: 100',
        'wrote GT: 900 queries x 100 keys -> '
        'indep/neighbors_indep_q10m_900.txt',
        'brute-force check on 10 sampled queries (keys mapped): top100 set '
        'match 10/10, top50 set match 10/10',
        '=== v14 independent GT generation DONE 2026-10-16T18:29:54+08:00 ===',
        f'=== v14 independent-set validation np=170 N=55 {TS} '
        'commit=423423c1 ===',
    ]
    for name, v, *_ in INDEP_CFGS:
        lines.append(f'--- indep recall {name} {TS} ---')
        lines.append(f'indep recall {name} rc=0 Recall@50: {v}')
    for name, _, wq, wr, wl, _ in INDEP_CFGS:
        lines.append(f'--- indep warmup {name} {TS} ---')
        lines.append(f'indep bench warmup_{name} rc=0 qps={wq} '
                     f'rss_kb={wr} loadavg={wl}')
    order = {1: [0, 1, 2, 3], 2: [1, 2, 3, 0], 3: [2, 3, 0, 1]}
    for r in range(1, INDEP_ROUNDS + 1):
        lines.append(f'=== indep ROUND r{r} {TS} loadavg=2.0 2.0 2.0 ===')
        for i in order[r]:
            name, _, _, _, _, rr = INDEP_CFGS[i]
            q, rss, load = rr[r - 1]
            lines.append(f'--- indep r{r} {name} {TS} ---')
            lines.append(f'indep bench {name}_r{r} rc=0 qps={q} '
                         f'rss_kb={rss} loadavg={load}')
    lines.append(f'=== v14 independent-set validation DONE {TS} ===')
    return lines


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
    lines = ['=== v14 prod 4way natref np=170 N=55 ===']
    lines.extend(RECALLS)
    for name, _, _ in CFG:
        lines.append(f'--- warmup {name} ---')
        lines.append(warmup_line(name))
    for r in range(1, ROUNDS + 1):
        lines.append(f'=== ROUND r{r} ===')
        for name, _, _ in CFG:
            lines.append(f'--- r{r} {name}_r{r} ---')
            lines.append(bench_line(name, r))
    lines.append('=== v14 prod 4way DONE ===')
    return lines


def drop(lines, sub):
    return [l for l in lines if sub not in l]


def replace_line(lines, sub, new):
    return [new if sub in l else l for l in lines]


def run_case(name, summary, indep=None, expect_rc0=False,
             must=None, must_stderr=None, must_not=None):
    d = SCRATCH / name
    if d.exists():
        shutil.rmtree(d)
    (d / 'tmp').mkdir(parents=True)
    (d / 'v14_prod_summary.txt').write_text(
        '\n'.join(summary) + '\n', encoding='utf-8')
    if indep is not None:
        (d / 'v14_indep_summary.txt').write_text(
            '\n'.join(indep) + '\n', encoding='utf-8')
    env = dict(os.environ, TMPDIR=str(d / 'tmp'))
    p = subprocess.run([sys.executable, str(ANALYZER), str(d)],
                       capture_output=True, text=True, env=env)
    out, err = p.stdout, p.stderr
    problems = []
    if expect_rc0 and p.returncode != 0:
        problems.append(f'rc={p.returncode}, expected 0; stderr:\n{err}')
    if not expect_rc0 and p.returncode == 0:
        problems.append('rc=0, expected non-zero')
    outfile = d / 'v14_prod_analysis.out'
    if expect_rc0:
        if not outfile.exists():
            problems.append('v14_prod_analysis.out not written')
        else:
            otext = outfile.read_text(encoding='utf-8')
            if 'integrity: 40/40 formal records OK' not in otext:
                problems.append('integrity line missing from .out')
    else:
        if outfile.exists():
            problems.append('v14_prod_analysis.out written despite failure')
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
    indep = build_indep()

    native_r2 = values('v13_native_np249_t1', 2)
    unparseable = (f'bench v13_native_np249_t1_r2 rc=0 qps={native_r2["qps"]} '
                   f'avg_ms={native_r2["avg"]} p99_ms={native_r2["p99"]} '
                   f'rss_kb={native_r2["rss"]} loadavg={native_r2["load"]}')
    failed_r3 = bench_line('v14_natref_np170_top55_t4', 3, rc=1, qps='MISSING',
                           avg='MISSING', p95='MISSING', p99='MISSING',
                           rss='MISSING')
    failed_warm = warmup_line('v13_native_np249_t1', rc=1, qps='MISSING',
                              avg='MISSING', p95='MISSING', p99='MISSING',
                              rss='MISSING')
    bad_indep = replace_line(
        indep, 'indep recall v13_native_np249_t1_indep rc=0 Recall@50: 92.74',
        'indep recall v13_native_np249_t1_indep rc=1 Recall@50: 92.74')
    bad_indep = replace_line(
        bad_indep,
        'indep recall v14_natref_np170_top55_t1_indep rc=0 '
        'Recall@50: 89.6778',
        'indep recall v14_natref_np170_top55_t1_indep rc=0 '
        'Recall@50: MISSING')
    r1_v12 = next(l for l in indep
                  if l == 'indep bench v12_prod50_np293_t1_indep_r1 rc=0 '
                          'qps=42.4 rss_kb=835576 loadavg=2.47')
    no_round = ('indep bench v12_prod50_np293_t1_indep rc=0 qps=42.4 '
                'rss_kb=835576 loadavg=2.47')
    gtgen_only = [l for l in indep if 'indep recall' not in l
                  and 'indep bench' not in l
                  and '--- indep' not in l and 'indep ROUND' not in l
                  and 'independent-set validation' not in l]
    mismatched = [l.replace('np170_top55', 'np175_top55') for l in indep]

    cases = [
        ('complete', dict(
            summary=base, indep=indep, expect_rc0=True,
            must=['integrity: 40/40 formal records OK',
                  'qps med=191.0', 'qps med=131.0', 'qps med=871.0',
                  'rss med=3.42GB', 'rss med=3.31GB', 'rss med=0.80GB',
                  't1 rrf/int:', 'x1.458',
                  'recall v14_natref_np170_top55_t1 rc=0 Recall@50: 99.058',
                  'independent query set (cohere 10m test split',
                  'indep integrity: 4 recall + 4 warmup + 12 bench records',
                  'indep recall v12_prod50_np293_t1_indep rc=0 '
                  'Recall@50: 94.4689',
                  'indep bench warmup_v12_prod50_np293_t1_indep rc=0 qps=42.9',
                  'indep bench v12_prod50_np293_t1_indep_r1 rc=0 qps=42.4',
                  'indep bench v14_natref_np170_top55_t1_indep_r3 rc=0 '
                  'qps=102.6'],
            must_not=['INTEGRITY FAILURE'])),
        ('no_indep_file', dict(
            summary=base, expect_rc0=True,
            must=['integrity: 40/40 formal records OK',
                  'independent validation not completed'],
            must_not=['indep recall', 'INTEGRITY FAILURE'])),
        ('bad_indep_line', dict(
            summary=base, indep=bad_indep,
            must_stderr=['rc=1', 'Recall@50 MISSING'])),
        ('indep_missing_round', dict(
            summary=base, indep=drop(indep, '_r2 rc=0 qps=40.8'),
            must_stderr=['indep missing rounds: v12_prod50_np293_t1_indep '
                         'lacks r2',
                         'indep record count 19 != 20'])),
        ('indep_duplicate_round', dict(
            summary=base, indep=indep + [r1_v12],
            must_stderr=['duplicate (config, round) record '
                         'v12_prod50_np293_t1_indep_r1'])),
        ('indep_failed_marker', dict(
            summary=base,
            indep=[l for l in indep if 'r1 rc=0 qps=42.4' not in l] +
                  ['FAILED: indep bench v12_prod50_np293_t1_indep_r1 rc=1 '
                   '-- aborting'],
            must_stderr=['indep line ', 'aborted run',
                         'indep missing rounds: v12_prod50_np293_t1_indep '
                         'lacks r1'])),
        ('indep_missing_recall', dict(
            summary=base,
            indep=drop(indep,
                       'indep recall v13_rrf55_np180_t1_indep rc=0 '
                       'Recall@50: 91.4422'),
            must_stderr=['indep missing recall record: '
                         'v13_rrf55_np180_t1_indep',
                         'indep record count 19 != 20'])),
        ('indep_one_recall_only', dict(
            summary=base,
            indep=[l for l in indep if not l.startswith('indep recall')
                   or 'v12_prod50_np293_t1_indep' in l],
            must_stderr=['indep missing recall record: '
                         'v13_rrf55_np180_t1_indep',
                         'indep missing recall record: '
                         'v13_native_np249_t1_indep',
                         'indep missing recall record: '
                         'v14_natref_np170_top55_t1_indep'])),
        ('indep_missing_warmup', dict(
            summary=base,
            indep=drop(indep, 'warmup_v13_native_np249_t1_indep rc=0'),
            must_stderr=['indep missing warmup record: '
                         'warmup_v13_native_np249_t1_indep',
                         'indep record count 19 != 20'])),
        ('indep_unexpected_round', dict(
            summary=base,
            indep=indep + ['indep bench v12_prod50_np293_t1_indep_r4 rc=0 '
                           'qps=43.0 rss_kb=835576 loadavg=2.0'],
            must_stderr=['indep unexpected rounds: '
                         'v12_prod50_np293_t1_indep has r4',
                         'indep record count 21 != 20'])),
        ('indep_config_mismatch', dict(
            summary=base, indep=mismatched,
            must_stderr=['indep config identity mismatch for scheme natref: '
                         "got ['v14_natref_np175_top55_t1'], expected "
                         'v14_natref_np170_top55_t1'])),
        ('indep_run_mixing', dict(
            summary=base, indep=indep + indep,
            must_stderr=['duplicate recall record', 'duplicate warmup record',
                         'duplicate (config, round) record'])),
        ('indep_no_round_suffix', dict(
            summary=base,
            indep=replace_line(indep, 'v12_prod50_np293_t1_indep_r1 rc=0 '
                                      'qps=42.4', no_round),
            must_stderr=['bench tag v12_prod50_np293_t1_indep is neither a '
                         'warmup nor a formal _r<round> record',
                         'indep missing rounds: v12_prod50_np293_t1_indep '
                         'lacks r1'])),
        ('indep_gtgen_only', dict(
            summary=base, indep=gtgen_only,
            must_stderr=['indep summary exists but contains no validation '
                         'records'])),
        ('missing_round', dict(
            summary=drop(base, 'bench v14_natref_np170_top55_t4_r3 '),
            must_stderr=['missing rounds: v14_natref_np170_top55_t4 lacks r3'])),
        ('failed_round', dict(
            summary=replace_line(base, 'bench v14_natref_np170_top55_t4_r3 ',
                                 failed_r3),
            must_stderr=['FAILED round: v14_natref_np170_top55_t4_r3 rc=1',
                         'field MISSING: v14_natref_np170_top55_t4_r3 qps'])),
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
            summary=replace_line(base, 'bench v13_native_np249_t1_r2 ',
                                 unparseable),
            must_stderr=['unparseable bench line',
                         'missing rounds: v13_native_np249_t1 lacks r2'])),
        ('missing_config', dict(
            summary=drop(base, 'v14_natref_np170_top55_t4'),
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

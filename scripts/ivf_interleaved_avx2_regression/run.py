#!/usr/bin/env python3
"""Reproduce baseline/candidate correctness and IVF timings from the handoff."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import shlex
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

TOOLS = Path(__file__).resolve().parent
LIBS = ['zvec_ailego', 'core_framework', 'core_utility', 'core_metric',
        'core_quantizer', 'core_knn_cluster', 'core_knn_flat', 'core_knn_ivf', 'zvec_turbo']
KEYS = ['type', 'metric', 'dim', 'distribution', 'nprobe', 'count']
HEADER = ','.join(KEYS + ['ns_per_query', 'distances']) + '\n'

def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')

def verify_manifest(bundle):
    for line in (bundle / 'SHA256SUMS').read_text().splitlines():
        digest, name = line.split('  ', 1)
        p = bundle / name
        if hashlib.sha256(p.read_bytes()).hexdigest() != digest:
            raise RuntimeError('SHA256 mismatch: ' + name)

def run(cmd, log, cwd=None, env=None, cpu=None):
    print('RUN', ' '.join(map(str, cmd)), flush=True)
    pin = (lambda: os.sched_setaffinity(0, {cpu})) if cpu is not None else None
    with log.open('a') as f:
        subprocess.run(list(map(str, cmd)), cwd=cwd, env=env, stdout=f,
                       stderr=subprocess.STDOUT, check=True, preexec_fn=pin)

def counts(xml):
    result = dict(passed=0, skipped=0, disabled=0, failed=0)
    for c in ET.parse(xml).getroot().iter('testcase'):
        key = ('failed' if c.find('failure') is not None else
               'skipped' if c.get('result') == 'skipped' or c.find('skipped') is not None else
               'disabled' if c.get('status') == 'notrun' or c.get('result') == 'suppressed' else 'passed')
        result[key] += 1
    return result

def static_link(build, target, destination, log):
    lines = subprocess.check_output(['ninja', '-t', 'commands', target], cwd=build, text=True).splitlines()
    args = shlex.split(lines[-1])
    if args[:2] == [':', '&&']:
        args = args[2:]
    if '&&' in args:
        args = args[:args.index('&&')]
    args[args.index('-o') + 1] = str(destination)
    cmd = []
    for arg in args:
        if arg.startswith('-Wl,-rpath,'):
            continue
        if re.search(r'\.so(?:\.\d+)*$|\.dylib$', arg):
            lib = re.sub(r'\.so(?:\.\d+)*$|(?:\.\d+)*\.dylib$', '.a', arg)
            if not (build / lib).exists():
                raise RuntimeError('Missing static archive: ' + lib)
            if sys.platform == 'darwin':
                cmd += ['-Wl,-force_load', lib]
            else:
                cmd += ['-Wl,--whole-archive', lib, '-Wl,--no-whole-archive']
        else:
            cmd.append(arg)
    run(cmd, log, cwd=build)

def configure(source, build, deps, log, sanitize=False):
    flags = '-fsanitize=address,undefined -fno-omit-frame-pointer' if sanitize else ''
    run(['cmake', '-S', TOOLS, '-B', build, '-G', 'Ninja',
         '-DPROJECT_ROOT_DIR=' + str(source), '-DVALIDATION_DEPS=' + str(deps),
         '-DVALIDATION_FIXED=ON', '-DCMAKE_BUILD_TYPE=Release',
         '-DCMAKE_INTERPROCEDURAL_OPTIMIZATION=OFF', '-DCMAKE_EXPORT_COMPILE_COMMANDS=ON',
         '-DCMAKE_CXX_FLAGS=' + flags, '-DCMAKE_EXE_LINKER_FLAGS=' + flags,
         '-DCMAKE_SHARED_LINKER_FLAGS=' + flags], log)

def test_binary(binary, directory, env=None, args=None):
    directory.mkdir(parents=True, exist_ok=True)
    xml = directory / 'results.xml'
    run([binary, '--gtest_output=xml:' + str(xml), *(args or [])], directory / 'run.log', directory, env)
    result = counts(xml)
    if result['failed']:
        raise RuntimeError('Correctness failure: ' + str(binary))
    return result

def compare(out, prefix, cases):
    rounds, correctness = [], []
    for rnd in [1, 2]:
        bd, ad = out / f'{prefix}before-round{rnd}', out / f'{prefix}after-round{rnd}'
        before = list(csv.DictReader((bd / 'timings.csv').open()))
        after = list(csv.DictReader((ad / 'timings.csv').open()))
        if len(before) != cases or len(after) != cases:
            raise RuntimeError('Missing benchmark cases')
        rows = []
        for a, b in zip(before, after):
            if any(a[k] != b[k] for k in KEYS) or a['distances'] != b['distances']:
                raise RuntimeError('Case ordering or scan counts changed')
            if any(not math.isfinite(float(z['ns_per_query'])) or float(z['ns_per_query']) <= 0 for z in [a, b]):
                raise RuntimeError('Non-finite or non-positive timing')
            rows.append({**{k: a[k] for k in KEYS}, 'before_ns': float(a['ns_per_query']),
                         'after_ns': float(b['ns_per_query']),
                         'speedup': float(a['ns_per_query']) / float(b['ns_per_query'])})
        rounds.append(rows)
        a = (bd / 'query_results.txt').read_text().splitlines()
        b = (ad / 'query_results.txt').read_text().splitlines()
        if len(a) != len(b) or (cases and not a):
            raise RuntimeError('Missing query results')
        error = 0.0
        for x, y in zip(a, b):
            xx, yy = x.rsplit(',', 1), y.rsplit(',', 1)
            if xx[0] != yy[0]:
                raise RuntimeError('Top-K ID/order changed: ' + x + ' vs ' + y)
            if not math.isfinite(float(xx[1])) or not math.isfinite(float(yy[1])):
                raise RuntimeError('Non-finite score')
            err = abs(float(xx[1]) - float(yy[1]))
            if err > 1e-4 * max(1, abs(float(xx[1]))):
                raise RuntimeError('Score mismatch: ' + x + ' vs ' + y)
            error = max(error, err)
        correctness.append(dict(round=rnd, records=len(a), ids_identical=True, max_score_error=error))
    sustained = [dict(a, speedup_round2=b['speedup']) for a, b in zip(*rounds)
                 if a['speedup'] < 1 / 1.05 and b['speedup'] < 1 / 1.05]
    result = dict(rounds=rounds, correctness=correctness, sustained_regressions=sustained)
    dump(out / (prefix + 'performance.json'), result)
    table = ['type,metric,dim,distribution,nprobe,count,before_r1_ns,after_r1_ns,speedup_r1,before_r2_ns,after_r2_ns,speedup_r2']
    for a, b in zip(*rounds):
        table.append(','.join([*[a[k] for k in KEYS], *[str(x) for x in
                     [a['before_ns'], a['after_ns'], a['speedup'], b['before_ns'], b['after_ns'], b['speedup']]]]))
    (out / (prefix + 'PERFORMANCE.csv')).write_text('\n'.join(table) + '\n')
    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bundle', type=Path, default=TOOLS.parent)
    parser.add_argument('--deps', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--jobs', type=int, default=4)
    parser.add_argument('--cpu', type=int)
    parser.add_argument('--sample-ms', type=float, default=50)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--allow-non-x86', action='store_true', help='Local tooling smoke check only, never Linux acceptance')
    args = parser.parse_args()
    if args.jobs < 1 or not math.isfinite(args.sample_ms) or args.sample_ms < 20:
        parser.error('jobs >= 1 and sample-ms >= 20 required')
    bundle, deps, out = args.bundle.resolve(), args.deps.resolve(), args.out.resolve()
    verify_manifest(bundle)
    native = sys.platform.startswith('linux') and platform.machine().lower() in ('x86_64', 'amd64')
    if not native and not args.allow_non_x86:
        parser.error('Acceptance requires native Linux x86_64; --allow-non-x86 is only a tooling smoke check')
    for p in ['thirdparty/googletest/googletest-1.10.0/CMakeLists.txt',
              'thirdparty/RaBitQ-Library/RaBitQ-Library-0.7.0/include']:
        if not (deps / p).exists():
            parser.error('Missing dependency: ' + str(deps / p))
    out.mkdir(parents=True, exist_ok=False)
    summary = dict(status='RUNNING', native_linux_x86=native)
    dump(out / 'environment.json', dict(platform=platform.platform(), python=sys.version,
         args={k: str(v) for k, v in vars(args).items()}, affinity=sorted(os.sched_getaffinity(0)) if native else None))
    try:
        if native:
            run(['lscpu'], out / 'lscpu.txt')
        sources = {}
        for name in ['before', 'after']:
            src = out / 'sources' / name
            src.mkdir(parents=True)
            run(['tar', '-xzf', bundle / 'source-base.tar.gz', '-C', src], out / 'prepare.log')
            run(['git', 'init', '--quiet'], out / 'prepare.log', src)
            run(['git', 'apply', bundle / 'baseline-vs-head.patch'], out / 'prepare.log', src)
            if name == 'after':
                run(['git', 'apply', bundle / 'candidate-vs-baseline.patch'], out / 'prepare.log', src)
            expected = json.loads((bundle / (name + '-source-sha256.json')).read_text())
            for f, h in expected.items():
                if hashlib.sha256((src / f).read_bytes()).hexdigest() != h:
                    raise RuntimeError('Source hash mismatch: ' + name + '/' + f)
            sources[name] = src
            configure(src, out / ('build-' + name), deps, out / ('configure-' + name + '.log'))
        if args.prepare_only:
            summary['status'] = 'PREPARED_ONLY'
            return 0
        paths = ['tests/turbo', 'tests/core/metric', 'tests/core/algorithm/flat', 'tests/core/algorithm/ivf']
        tests = sorted(p.stem for folder in paths for p in (sources['after'] / folder).glob('*_test.cc'))
        extras = ['ivf_interleaved_bench', 'legacy_ivf_comparison', 'interleaved_dispatch_probe']
        executables = {}
        for name in ['before', 'after']:
            build = out / ('build-' + name)
            run(['cmake', '--build', build, '--parallel', args.jobs, '--target', *LIBS, *tests, *extras], out / ('build-' + name + '.log'))
            bindir = out / ('bin-' + name)
            bindir.mkdir()
            for target in tests + extras:
                static_link(build, target, bindir / target, out / ('link-' + name + '.log'))
            executables[name] = bindir
            probe_args = ['--require-avx2'] if name == 'after' and native else []
            run([bindir / 'interleaved_dispatch_probe', *probe_args], out / ('dispatch-' + name + '.log'))
            results = {}
            for target in tests:
                results[target] = test_binary(bindir / target, out / ('correctness-' + name) / target)
            env = dict(os.environ, ZVEC_LEGACY_READ_FIXTURES=str(bundle / 'old-index-fixtures'))
            results['old_index_compatibility'] = test_binary(bindir / 'legacy_ivf_comparison', out / ('compatibility-' + name), env)
            dump(out / ('correctness-' + name + '.json'), results)
        if native:
            xml = ET.parse(out / 'correctness-after/turbo_matrix_test/results.xml').getroot()
            required = [c for c in xml.iter('testcase') if c.get('name') == 'Avx2JointInterleavedQueriesAndNoAllocations']
            if len(required) != 1 or required[0].get('result') != 'completed':
                raise RuntimeError('AVX2 numerical test missing or skipped: cannot accept fallback-only validation')
        san = out / 'build-sanitized'
        configure(sources['after'], san, deps, out / 'configure-sanitized.log', True)
        san_tests = ['turbo_matrix_test', 'ivf_legacy_matrix_test', 'ivf_searcher_test']
        run(['cmake', '--build', san, '--parallel', args.jobs, '--target', *LIBS, *san_tests], out / 'build-sanitized.log')
        san_bin = out / 'bin-sanitized'
        san_bin.mkdir()
        for target in san_tests:
            static_link(san, target, san_bin / target, out / 'link-sanitized.log')
            test_binary(san_bin / target, out / 'sanitized' / target,
                        dict(os.environ, ASAN_OPTIONS='detect_leaks=0:halt_on_error=1', UBSAN_OPTIONS='halt_on_error=1:print_stacktrace=1'))
        (out / 'indexes').mkdir()
        prepare = out / 'prepare-indexes'
        prepare.mkdir()
        run([executables['before'] / 'ivf_interleaved_bench', '--gtest_filter=*ManyBenchmark*'], prepare / 'run.log', prepare, dict(os.environ, PREPARE='1'))
        cpu = args.cpu if args.cpu is not None else (min(os.sched_getaffinity(0)) if native else None)
        if native and cpu not in os.sched_getaffinity(0):
            raise RuntimeError('Selected CPU outside allowed affinity')
        summary['benchmark_cpu'] = cpu
        summary['sample_ms'] = args.sample_ms
        time.sleep(10)
        env = dict(os.environ, BENCH_MIN_MS=str(args.sample_ms))
        env.pop('BENCH_CASE', None)
        env.pop('PREPARE', None)
        def init_dirs(prefix):
            for rnd in [1, 2]:
                for name in ['before', 'after']:
                    d = out / f'{prefix}{name}-round{rnd}'
                    d.mkdir(); (d / 'timings.csv').write_text(HEADER); (d / 'query_results.txt').write_text('')
        def measure(name, rnd, prefix='', case=None):
            d = out / f'{prefix}{name}-round{rnd}'
            e = dict(env, BENCH_CASE=case) if case else env
            run([executables[name] / 'ivf_interleaved_bench', '--gtest_filter=*ManyBenchmark*'], d / 'run.log', d, e, cpu)
        init_dirs('')
        for rnd in [1, 2]:
            for name in (['before', 'after'] if rnd == 1 else ['after', 'before']):
                measure(name, rnd)
        perf = compare(out, '', 288)
        cases = sorted({','.join(x[k] for k in KEYS) for rows in perf['rounds'] for x in rows if x['speedup'] < 1 / 1.05})
        init_dirs('retake-')
        for rnd in [1, 2]:
            for case in cases:
                for name in (['before', 'after'] if rnd == 1 else ['after', 'before']):
                    measure(name, rnd, 'retake-', case)
        retake = compare(out, 'retake-', len(cases))
        summary.update(correctness='PASS', cases_per_round=288, retake_cases=len(cases),
                       original_sustained_regressions=perf['sustained_regressions'],
                       retake_sustained_regressions=retake['sustained_regressions'])
        summary['status'] = ('PERFORMANCE_REVIEW_REQUIRED' if retake['sustained_regressions'] else
                             'PASS_NATIVE_X86' if native else 'SMOKE_ONLY')
        return 1 if retake['sustained_regressions'] else 0
    except Exception as e:
        summary.update(status='FAIL', error=str(e))
        raise
    finally:
        dump(out / 'SUMMARY.json', summary)
        print(json.dumps(summary, indent=2), flush=True)

if __name__ == '__main__':
    sys.exit(main())

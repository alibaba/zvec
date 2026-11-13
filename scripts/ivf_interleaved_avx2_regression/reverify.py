#!/usr/bin/env python3
"""Reverify the frozen AVX2 handoff with identical test repairs on both sides."""
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import xml.etree.ElementTree as ET

import run as kit

TOOLS = Path(__file__).resolve().parent
TEST_FILES = {'tests/core/algorithm/ivf/ivf_legacy_matrix_test.cc',
              'tests/turbo/turbo_many_to_many_distance_test.cc',
              'tests/turbo/turbo_matrix_test.cc'}
SAN_TESTS = ['turbo_matrix_test', 'turbo_many_to_many_distance_test',
             'ivf_legacy_matrix_test', 'ivf_searcher_test']


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(bundle, out):
    kit.verify_manifest(bundle)
    patch = TOOLS / 'validation-fixes.patch'
    meta = json.loads((TOOLS / 'validation-fixes.json').read_text())
    if digest(patch) != meta['patch_sha256']:
        raise RuntimeError('Validation patch checksum mismatch')
    sources = {}
    for version in ['before', 'after']:
        source = out / 'sources' / version
        source.mkdir(parents=True)
        with tarfile.open(bundle / 'source-base.tar.gz') as archive:
            for member in archive.getmembers():
                if (Path(member.name).is_absolute() or '..' in Path(member.name).parts
                        or not (member.isdir() or member.isfile())):
                    raise RuntimeError('Unsafe source archive member: ' + member.name)
            archive.extractall(source, **({'filter': 'data'} if hasattr(tarfile, 'data_filter') else {}))
        log = out / 'prepare.log'
        kit.run(['git', 'init', '--quiet'], log, source)
        kit.run(['git', 'apply', bundle / 'baseline-vs-head.patch'], log, source)
        if version == 'after':
            kit.run(['git', 'apply', bundle / 'candidate-vs-baseline.patch'], log, source)
        expected = json.loads((bundle / (version + '-source-sha256.json')).read_text())
        for name, value in expected.items():
            if digest(source / name) != value:
                raise RuntimeError('Frozen source mismatch: ' + version + '/' + name)
        if set(meta[version]) != TEST_FILES:
            raise RuntimeError('Only the three test files may be repaired')
        kit.run(['git', 'apply', '--check', patch], log, source)
        kit.run(['git', 'apply', patch], log, source)
        expected.update(meta[version])
        for name, value in expected.items():
            if digest(source / name) != value:
                raise RuntimeError('Repaired source mismatch: ' + version + '/' + name)
        kit.dump(out / (version + '-source-sha256.json'), expected)
        sources[version] = source
    return sources


def test(binary, directory, sanitizer=False):
    directory.mkdir(parents=True)
    xml, log = directory / 'results.xml', directory / 'run.log'
    env = dict(os.environ)
    for key in list(env):
        if key.startswith('GTEST_'):
            del env[key]
    if sanitizer:
        # Collect all UB sites on BOTH sides, but never treat a diagnostic as
        # a pass. ASan remains fatal; no mismatch/alignment/vptr suppression.
        env.update(ASAN_OPTIONS='detect_leaks=0:halt_on_error=1',
                   UBSAN_OPTIONS='halt_on_error=0:print_stacktrace=1')
    code = 0
    try:
        kit.run([binary, '--gtest_filter=*', '--gtest_output=xml:' + str(xml)], log,
                directory, env)
    except subprocess.CalledProcessError as exc:
        code = exc.returncode
    result = kit.counts(xml) if xml.exists() else dict(passed=0, skipped=0, disabled=0, failed=0)
    content = log.read_text(errors='replace')
    # Normalize just addresses and source roots. Keep line, type and operation.
    diagnostics = []
    for line in content.splitlines():
        if 'runtime error:' in line or 'ERROR: AddressSanitizer:' in line:
            line = re.sub(r'^.*?/sources/(?:before|after)/', '', line)
            line = re.sub(r'==\d+==', '', line)
            diagnostics.append(re.sub(r'0x[0-9a-fA-F]+', '<address>', line))
    result.update(exit_code=code, missing_xml=not xml.exists(), diagnostics=diagnostics)
    result['ok'] = code == 0 and xml.exists() and not result['failed'] and not diagnostics
    kit.dump(directory / 'summary.json', result)
    return result


def require_test(xml, name):
    cases = [c for c in ET.parse(xml).getroot().iter('testcase') if c.get('name') == name]
    if (len(cases) != 1 or cases[0].get('result') != 'completed'
            or cases[0].find('failure') is not None):
        raise RuntimeError('Required test failed, missing or skipped: ' + name)


def check_instrumentation(build, out):
    commands = json.loads((build / 'compile_commands.json').read_text())
    missing = [x['file'] for x in commands if '-fsanitize=address,undefined' not in x['command']
               or '-fno-sanitize=' in x['command']]
    kit.dump(out, dict(translation_units=len(commands), missing_or_suppressed=missing))
    if missing:
        raise RuntimeError('Incomplete sanitizer instrumentation: ' + str(missing))


def benchmark(out, binaries, cpu, sample_ms):
    (out / 'indexes').mkdir()
    prep = out / 'prepare-indexes'
    prep.mkdir()
    kit.run([binaries['before'] / 'ivf_interleaved_bench', '--gtest_filter=*ManyBenchmark*'],
            prep / 'run.log', prep, dict(os.environ, PREPARE='1'))
    env = dict(os.environ, BENCH_MIN_MS=str(sample_ms))
    env.pop('PREPARE', None)
    env.pop('BENCH_CASE', None)

    def measure(prefix, cases):
        for rnd in [1, 2]:
            for version in ['before', 'after']:
                d = out / f'{prefix}{version}-round{rnd}'
                d.mkdir()
                (d / 'timings.csv').write_text(kit.HEADER)
                (d / 'query_results.txt').write_text('')
            for case in cases:
                for version in (['before', 'after'] if rnd == 1 else ['after', 'before']):
                    d = out / f'{prefix}{version}-round{rnd}'
                    kit.run([binaries[version] / 'ivf_interleaved_bench', '--gtest_filter=*ManyBenchmark*'],
                            d / 'run.log', d, dict(env, BENCH_CASE=case) if case else env, cpu)
    measure('', [None])
    full = kit.compare(out, '', 288)
    cases = sorted({','.join(row[k] for k in kit.KEYS) for rows in full['rounds']
                    for row in rows if row['speedup'] < 1 / 1.05})
    measure('retake-', cases)
    retake = kit.compare(out, 'retake-', len(cases))
    return dict(cases_per_round=288, retake_cases=len(cases),
                sustained_regressions=retake['sustained_regressions'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--deps', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--jobs', type=int, default=4)
    parser.add_argument('--cpu', type=int)
    parser.add_argument('--sample-ms', type=float, default=50)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--allow-non-x86', action='store_true')
    args = parser.parse_args()
    native = sys.platform.startswith('linux') and platform.machine().lower() in ('x86_64', 'amd64')
    if not native and not args.allow_non_x86:
        parser.error('Native Linux x86_64 required for acceptance')
    if args.jobs < 1 or not math.isfinite(args.sample_ms) or args.sample_ms < 20:
        parser.error('jobs >= 1, sample-ms >= 20 required')
    out, bundle, deps = args.out.resolve(), args.bundle.resolve(), args.deps.resolve()
    out.mkdir(parents=True, exist_ok=False)
    summary = dict(status='RUNNING', native_linux_x86=native, test_filters='none',
                   sanitizer_checks='ASan+UBSan including alignment/vptr; leak detection disabled')
    kit.dump(out / 'environment.json', dict(platform=platform.platform(), python=sys.version,
             args={k: str(v) for k, v in vars(args).items()}))
    try:
        tool_copy = out / 'tools'
        tool_copy.mkdir()
        for name in ['reverify.py', 'run.py', 'CMakeLists.txt', 'bench.cc',
                     'benchmark_clock.h', 'dispatch_probe.cc', 'legacy_compatibility.cc',
                     'validation-fixes.patch', 'validation-fixes.json', 'REVERIFY_zh.txt']:
            shutil.copy2(TOOLS / name, tool_copy / name)
        kit.dump(out / 'tools-source-sha256.json', {p.name: digest(p) for p in sorted(tool_copy.iterdir())})
        kit.run(['git', 'rev-parse', 'HEAD'], out / 'dependency-checkout.log', deps)
        kit.run(['git', 'submodule', 'status', '--', 'thirdparty/googletest/googletest-1.10.0',
                 'thirdparty/RaBitQ-Library/RaBitQ-Library-0.7.0'], out / 'dependency-checkout.log', deps)
        if native:
            kit.run(['lscpu'], out / 'lscpu.txt')
        sources = prepare(bundle, out)
        if args.prepare_only:
            summary['status'] = 'PREPARED_ONLY'
            return 0
        paths = ['tests/turbo', 'tests/core/metric', 'tests/core/algorithm/flat', 'tests/core/algorithm/ivf']
        targets = sorted(p.stem for folder in paths for p in (sources['after'] / folder).glob('*_test.cc'))
        extras = ['ivf_interleaved_bench', 'legacy_ivf_comparison', 'interleaved_dispatch_probe']
        bins, results, sanitized = {}, {}, {}
        for version, source in sources.items():
            build = out / ('build-' + version)
            kit.configure(source, build, deps, out / ('configure-' + version + '.log'))
            kit.run(['cmake', '--build', build, '--parallel', args.jobs, '--target', *kit.LIBS, *targets, *extras],
                    out / ('build-' + version + '.log'))
            bindir = out / ('bin-' + version)
            bindir.mkdir()
            bins[version] = bindir
            for target in targets + extras:
                kit.static_link(build, target, bindir / target, out / ('link-' + version + '.log'))
            kit.run([bindir / 'interleaved_dispatch_probe', *(['--require-avx2'] if native and version == 'after' else [])],
                    out / ('dispatch-' + version + '.log'))
            results[version] = {target: test(bindir / target, out / ('correctness-' + version) / target)
                                for target in targets}
            os.environ['ZVEC_LEGACY_READ_FIXTURES'] = str(bundle / 'old-index-fixtures')
            results[version]['old_index_compatibility'] = test(bindir / 'legacy_ivf_comparison', out / ('compatibility-' + version))
            os.environ.pop('ZVEC_LEGACY_READ_FIXTURES')
            kit.dump(out / ('correctness-' + version + '.json'), results[version])
            many_xml = out / ('correctness-' + version) / 'turbo_many_to_many_distance_test/results.xml'
            require_test(many_xml, 'RecordInt8PreprocessAndMetadataFallback')
            if native:
                require_test(many_xml, 'Fp16Avx2')
                if version == 'after':
                    require_test(out / 'correctness-after/turbo_matrix_test/results.xml',
                                 'Avx2JointInterleavedQueriesAndNoAllocations')
        for version, source in sources.items():
            build = out / ('build-sanitized-' + version)
            kit.configure(source, build, deps, out / ('configure-sanitized-' + version + '.log'), True)
            check_instrumentation(build, out / ('instrumentation-' + version + '.json'))
            kit.run(['cmake', '--build', build, '--parallel', args.jobs, '--target', *kit.LIBS, *SAN_TESTS],
                    out / ('build-sanitized-' + version + '.log'))
            bindir = out / ('bin-sanitized-' + version)
            bindir.mkdir()
            sanitized[version] = {}
            for target in SAN_TESTS:
                kit.static_link(build, target, bindir / target, out / ('link-sanitized-' + version + '.log'))
                sanitized[version][target] = test(bindir / target, out / ('sanitized-' + version) / target, True)
            kit.dump(out / ('sanitized-' + version + '.json'), sanitized[version])
        new_diagnostics = {}
        for target in SAN_TESTS:
            old, new = (Counter(sanitized[v][target]['diagnostics']) for v in ['before', 'after'])
            new_diagnostics[target] = dict(new - old)
        kit.dump(out / 'sanitizer-comparison.json', dict(new_or_increased=new_diagnostics, results=sanitized))
        cpu = args.cpu if args.cpu is not None else (min(os.sched_getaffinity(0)) if native else None)
        summary['performance'] = benchmark(out, bins, cpu, args.sample_ms)
        correctness_ok = all(r['ok'] for tests in results.values() for r in tests.values())
        san_fail = any(r['exit_code'] or r['missing_xml'] or r['failed'] for tests in sanitized.values() for r in tests.values())
        san_reports = any(r['diagnostics'] for tests in sanitized.values() for r in tests.values())
        summary.update(correctness='PASS' if correctness_ok else 'FAIL',
                       sanitizer_new_or_increased=new_diagnostics)
        summary['status'] = ('CORRECTNESS_FAILED' if not correctness_ok else
                             'SANITIZER_FAILED' if san_fail or any(new_diagnostics.values()) else
                             'BASELINE_SANITIZER_REVIEW_REQUIRED' if san_reports else
                             'PERFORMANCE_REVIEW_REQUIRED' if summary['performance']['sustained_regressions'] else
                             'PASS_NATIVE_X86' if native else 'SMOKE_ONLY')
        return 0 if summary['status'] in ('PASS_NATIVE_X86', 'SMOKE_ONLY') else 1
    except Exception as exc:
        summary.update(status='FAIL', error=str(exc))
        raise
    finally:
        kit.dump(out / 'SUMMARY.json', summary)
        print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    sys.exit(main())

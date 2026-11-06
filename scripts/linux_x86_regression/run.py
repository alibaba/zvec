#!/usr/bin/env python3
"""Focused Linux regression; no git mutation, commit, push or dependency install."""
import argparse
import csv
import hashlib
import json
import os
import pathlib
import platform
import re
import statistics
import subprocess
import sys
import xml.etree.ElementTree as ET

BUNDLE = pathlib.Path(__file__).resolve().parent
REVISIONS = {'parent': '499eab892c750ecf60f314b3d24cccad888f5af2',
             'before': 'f7bd575f09b622a6dd387bcc266d2f3b53fd47c7',
             'fixed': '955ae841'}
IVF_TESTS = ['TestInit', 'LegacyQuantizerDescriptorDoesNotSelectTurboFormat',
             'TestColumnMajorFloatWithBuildMemory', 'TestColumnMajorFloatWithFilter',
             'TestRowMajorFloatWithBuildMemory', 'TestRowMajorFloatWithFilter',
             'TestColumnMajorFloatWith1LevelAndBuildMemory',
             'TestColumnMajorInt8WithBuildMemory', 'TestColumnMajorFp16WithBuildMemory',
             'TestSearchWithEmptyCentroid', 'TestNprobeScansAllSelectedLists',
             'TestNprobeClampToListCount', 'TestMipsEuclideanMetric']


def command(args, cwd, log, extra_env=None, strict=True):
    env = dict(os.environ)
    for key in ['ZVEC_LEGACY_READ_FIXTURES', 'ZVEC_LEGACY_WRITE_FIXTURES',
                'GTEST_FILTER', 'GTEST_ALSO_RUN_DISABLED_TESTS', 'GTEST_REPEAT',
                'GTEST_TOTAL_SHARDS', 'GTEST_SHARD_INDEX', 'GTEST_OUTPUT']:
        env.pop(key, None)
    env.update({'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'})
    env.update(extra_env or {})
    with log.open('w') as out:
        result = subprocess.run([str(x) for x in args], cwd=cwd, env=env,
                                stdout=out, stderr=subprocess.STDOUT)
    print(f'{log.name}: exit={result.returncode}', flush=True)
    if result.returncode and strict:
        raise RuntimeError(f'Command failed; inspect {log}')
    return result.returncode


def test_counts(xml):
    counts = dict(passed=0, skipped=0, disabled=0, failed=0)
    if not xml.exists():
        return {'missing_xml': True}
    for test in ET.parse(xml).getroot().iter('testcase'):
        result = test.attrib.get('result')
        key = ('failed' if test.find('failure') is not None else
               'skipped' if result == 'skipped' or test.find('skipped') is not None else
               'disabled' if result == 'suppressed' else 'passed')
        counts[key] += 1
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['build', 'correctness', 'perf', 'all'])
    parser.add_argument('--root', type=pathlib.Path, required=True,
                        help='Absolute directory containing parent/, before/, fixed/ worktrees')
    parser.add_argument('--jobs', type=int, default=8)
    parser.add_argument('--cpu', type=int, help='Allowed logical CPU for serial benchmarks')
    parser.add_argument('--deps', type=pathlib.Path,
                        help='Checkout with initialized submodules (default: fixed/)')
    args = parser.parse_args()
    if platform.system() != 'Linux' or platform.machine().lower() not in ('x86_64', 'amd64'):
        parser.error('Run on Linux x86_64; this script does not cross-compile.')
    root = args.root.resolve()
    results = root / 'results'
    results.mkdir(parents=True, exist_ok=True)
    trees = {name: root / name for name in REVISIONS}
    deps = args.deps.resolve() if args.deps else trees['fixed']
    builds = {name: root / ('build-' + name) for name in REVISIONS}
    runtime = root / 'runtime'
    runtime.mkdir(exist_ok=True)
    versions = {}
    for name, tree in trees.items():
        revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=tree, text=True).strip()
        if not revision.startswith(REVISIONS[name]):
            raise RuntimeError(f'{name}: expected {REVISIONS[name]}, got {revision}')
        diff = subprocess.check_output(['git', 'diff', 'HEAD', '--'], cwd=tree, text=True)
        patch = results / (name + '_worktree.patch')
        if diff:
            patch.write_text(diff)
            print(f'WARNING: {name} worktree has local changes; recorded in {patch}', flush=True)
        elif patch.exists():
            patch.unlink()
        versions[name] = {'revision': revision, 'worktree_patch': patch.name if diff else None}
    targets = [p.stem for directory in ['tests/core/metric', 'tests/turbo']
               for p in sorted((trees['fixed'] / directory).glob('*_test.cc'))]
    targets += ['legacy_quantized_index_test', 'flat_quantizer_distance_test',
                'mips_callback_lifetime_test', 'multi_chunk_cluster_test',
                'opt_kmeans_cluster_test', 'ivf_legacy_matrix_test', 'ivf_searcher_test']
    assert len(targets) == 44, targets
    metadata = {'versions': versions, 'uname': list(platform.uname()),
                'dependency_checkout': str(deps),
                'bundle_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in BUNDLE.iterdir() if p.is_file()},
                'scope': 'Focused metric/Turbo/Flat/clustering/legacy IVF; not full repository.',
                'excluded_backends': ['RABITQ_SUPPORTED=0', 'DISKANN_SUPPORTED=0'],
                'micro_unit': 'thread CPU ns/matrix', 'ivf_unit': 'wall-clock us/query',
                'parent_fp16_note': 'Parent FP16 1x1 matrix oracle is skipped; parent FP16 adapter pair tolerance is 1%. Fixed code retains strict checks.'}
    (results / 'environment.json').write_text(json.dumps(metadata, indent=2) + '\n')
    command(['lscpu'], root, results / 'lscpu.txt')
    if args.stage in ('build', 'all'):
        for name, tree in trees.items():
            command(['cmake', '-S', BUNDLE, '-B', builds[name], '-G', 'Ninja',
                     '-DCMAKE_BUILD_TYPE=Release', '-DCMAKE_EXPORT_COMPILE_COMMANDS=ON',
                     '-DCMAKE_INTERPROCEDURAL_OPTIMIZATION=OFF', '-DENABLE_OPENMP=OFF',
                     '-DAUTO_DETECT_ARCH=ON', '-DENABLE_NATIVE=OFF',
                     '-DPROJECT_ROOT_DIR=' + str(tree),
                     '-DVALIDATION_DEPS=' + str(deps),
                     '-DVALIDATION_LEGACY=' + ('ON' if name == 'parent' else 'OFF'),
                     '-DVALIDATION_FIXED=' + ('ON' if name == 'fixed' else 'OFF')],
                    root, results / (name + '_configure.log'))
            selected = ['matrix_sweep', 'adapter_sweep', 'legacy_ivf_comparison']
            if name == 'fixed':
                selected += targets
            command(['cmake', '--build', builds[name], '--parallel', str(args.jobs),
                     '--target', *selected], root, results / (name + '_build.log'))

    if args.stage in ('correctness', 'all'):
        outcomes = {}

        def run_test(label, binary, options=(), env=None):
            directory = runtime / label
            directory.mkdir(exist_ok=True)
            xml = results / (label + '.xml')
            if xml.exists():
                xml.unlink()  # Never reuse a stale success after a crash.
            code = command([binary, '--gtest_output=xml:' + str(xml), *options],
                           directory, results / (label + '.log'), env, strict=False)
            outcomes[label] = {'exit_code': code, **test_counts(xml)}
            (results / 'correctness.json').write_text(json.dumps(outcomes, indent=2) + '\n')

        for target in targets:
            options = []
            if target == 'ivf_searcher_test':
                options = ['--gtest_filter=' + ':'.join('IVFSearcherTest.' + x for x in IVF_TESTS)]
            run_test(target, builds['fixed'] / 'bin' / target, options)
        fixtures = root / 'parent-fixtures'
        fixtures.mkdir(exist_ok=True)
        run_test('parent_write', builds['parent'] / 'bin/legacy_ivf_comparison',
                 env={'ZVEC_LEGACY_WRITE_FIXTURES': str(fixtures)})
        if outcomes['parent_write']['exit_code'] == 0:
            assert len(list(fixtures.glob('*.index'))) == 200
            run_test('fixed_read_parent', builds['fixed'] / 'bin/legacy_ivf_comparison',
                     env={'ZVEC_LEGACY_READ_FIXTURES': str(fixtures)})
        run_test('before_ivf', builds['before'] / 'bin/legacy_ivf_comparison')
        if any(v['exit_code'] or v.get('failed') or v.get('missing_xml') for v in outcomes.values()):
            raise RuntimeError('Correctness failed; inspect correctness.json and individual logs.')

    if args.stage in ('perf', 'all'):
        allowed = sorted(os.sched_getaffinity(0))
        cpu = args.cpu if args.cpu is not None else allowed[0]
        if cpu not in allowed:
            raise RuntimeError(f'CPU {cpu} is not allowed; choose one of {allowed}')
        (results / 'benchmark_cpu.json').write_text(json.dumps({'cpu': cpu, 'allowed': allowed}) + '\n')
        summaries = {}
        for group, count in [('matrix', 504), ('adapter', 32), ('ivf', 24)]:
            names = list(REVISIONS)
            values = {}
            keys = ['type', 'metric', 'dim', 'record'] if group == 'ivf' else ['type', 'metric', 'rows', 'columns', 'dim']
            for repetition in range(3):
                for name in names[repetition:] + names[:repetition]:
                    binary = builds[name] / 'bin' / ('legacy_ivf_comparison' if group == 'ivf' else group + '_sweep')
                    options = ['--gtest_also_run_disabled_tests', '--gtest_filter=*Warm*Benchmark*'] if group == 'ivf' else []
                    log = results / f'{group}_{name}_{repetition}.log'
                    command(['taskset', '-c', str(cpu), binary, *options], runtime, log)
                    if group == 'ivf':
                        rows = [dict(re.findall(r'(\w+)=([^,\s]+)', line))
                                for line in log.read_text().splitlines() if line.startswith('LEGACY_BENCH,')]
                    else:
                        rows = list(csv.DictReader(log.open()))
                    assert len(rows) == count, (log, len(rows), count)
                    for row in rows:
                        key = tuple(row[k] for k in keys)
                        value = float(row['median_us' if group == 'ivf' else 'ns_per_matrix'])
                        values.setdefault(key, {}).setdefault(name, []).append(value)
            output = []
            for key, samples in values.items():
                row = dict(zip(keys, key))
                for name, times in samples.items():
                    assert len(times) == 3
                    row[name] = statistics.median(times)
                    row[name + '_min'] = min(times)
                    row[name + '_max'] = max(times)
                for name in ['parent', 'before']:
                    row['fixed_vs_' + name] = row['fixed'] / row[name]
                    row['change_vs_' + name + '_pct'] = (row['fixed'] / row[name] - 1) * 100
                output.append(row)
            assert len(output) == count
            with (results / (group + '_comparison.csv')).open('w') as out:
                writer = csv.DictWriter(out, fieldnames=list(output[0]))
                writer.writeheader()
                writer.writerows(output)
            summaries[group] = {'cases': count, 'median_over_parent_5pct':
                                [r for r in output if r['fixed_vs_parent'] > 1.05]}
        (results / 'performance.json').write_text(json.dumps(summaries, indent=2) + '\n')
    print(f'Results: {results}')


if __name__ == '__main__':
    main()

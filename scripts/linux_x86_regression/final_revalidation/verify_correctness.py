#!/usr/bin/env python3
"""Audit a run.py correctness run and complete a blocked legacy-fixture read."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import xml.etree.ElementTree as ET

BUNDLE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('regression_driver', BUNDLE / 'run.py')
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)
KNOWN_PARENT = {
    'RawTypes/IVFLegacyMatrixTest.PersistedColumnBlocksMatchIndependentReference/5',
    'RawTypes/IVFLegacyMatrixTest.PersistedRecordQuantizedBlocksMatchIndependentReference/5',
}


def audit(root, allow_known=False):
    results = root / 'results'
    (results / 'final_correctness_audit.json').unlink(missing_ok=True)
    outcomes = json.loads((results / 'correctness.json').read_text())
    targets = [p.stem for d in ['tests/core/metric', 'tests/turbo']
               for p in sorted((root / 'fixed' / d).glob('*_test.cc'))]
    targets += ['legacy_quantized_index_test', 'flat_quantizer_distance_test',
                'mips_callback_lifetime_test', 'multi_chunk_cluster_test',
                'opt_kmeans_cluster_test', 'ivf_legacy_matrix_test', 'ivf_searcher_test']
    if len(targets) != 44 or len(set(targets)) != 44:
        raise RuntimeError('Expected exactly 44 fixed test programs')

    def inspect(name):
        xml = results / (name + '.xml')
        entry = outcomes[name]
        if not xml.exists():
            raise RuntimeError(f'{name}: missing XML')
        tree = ET.parse(xml).getroot()
        counts = driver.test_counts(xml)
        if int(tree.get('errors', '0')) or any(
                int(x.get('errors', '0')) for x in tree.iter('testsuite')):
            raise RuntimeError(f'{name}: XML reports errors')
        if any(entry.get(k) != value for k, value in counts.items()):
            raise RuntimeError(f'{name}: JSON/XML counts disagree')
        failures = {f'{t.get("classname")}.{t.get("name")}'
                    for t in tree.iter('testcase') if t.find('failure') is not None}
        return entry, counts, failures

    def require_pass(name):
        entry, counts, failures = inspect(name)
        if entry['exit_code'] != 0 or counts['failed'] or failures or not counts['passed']:
            raise RuntimeError(f'{name}: not a successful test run')
        return counts

    totals = dict(passed=0, skipped=0, disabled=0, failed=0)
    for name in targets:
        for k, value in require_pass(name).items():
            totals[k] += value
    require_pass('before_ivf')
    parent, counts, failures = inspect('parent_write')
    known_exception = bool(parent['exit_code'] or counts['failed'])
    if known_exception:
        if not (allow_known and parent['exit_code'] == 1 and failures
                and failures <= KNOWN_PARENT and counts['failed'] == len(failures)):
            raise RuntimeError('Unexpected parent failure, or known exception not explicitly allowed')
    else:
        require_pass('parent_write')
    fixtures = root / 'parent-fixtures'
    paths = sorted(fixtures.glob('*.index'))
    if len(paths) != 200 or any(not p.is_file() or not p.stat().st_size for p in paths):
        raise RuntimeError('Expected 200 nonempty parent-generated .index fixtures')

    # run.py skips this stage when parent_write returns nonzero. Never change
    # or hide the parent failure; record a separate successful read instead.
    if 'fixed_read_parent' not in outcomes:
        directory = root / 'runtime' / 'fixed_read_parent'
        directory.mkdir(parents=True, exist_ok=True)
        xml = results / 'fixed_read_parent.xml'
        xml.unlink(missing_ok=True)
        code = driver.command(
            [root / 'build-fixed/bin/legacy_ivf_comparison', '--gtest_output=xml:' + str(xml)],
            directory, results / 'fixed_read_parent.log',
            {'ZVEC_LEGACY_READ_FIXTURES': str(fixtures)}, strict=False)
        outcomes['fixed_read_parent'] = {'exit_code': code, **driver.test_counts(xml)}
        (results / 'correctness.json').write_text(json.dumps(outcomes, indent=2) + '\n')
    read_counts = require_pass('fixed_read_parent')
    if read_counts != dict(passed=12, skipped=4, disabled=16, failed=0):
        raise RuntimeError(f'Unexpected fixture read scope: {read_counts}')
    (results / 'fixture_sha256.txt').write_text(''.join(
        hashlib.sha256(p.read_bytes()).hexdigest() + '  ' + p.name + '\n' for p in paths))
    report = {'fixed_programs': 44, 'fixed_totals': totals, 'fixture_count': 200,
              'fixed_read_parent': read_counts, 'parent_exit_code': parent['exit_code'],
              'known_parent_exception': known_exception,
              'parent_failed_cases': sorted(failures),
              'status': 'fixed_pass_with_known_parent_failures' if known_exception else 'pass'}
    (results / 'final_correctness_audit.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--allow-known-parent-int8-ip', action='store_true',
                        help='Use only after reviewing both historical INT8 IP failure logs')
    args = parser.parse_args()
    print(json.dumps(audit(args.root.resolve(), args.allow_known_parent_int8_ip), indent=2))


if __name__ == '__main__':
    main()

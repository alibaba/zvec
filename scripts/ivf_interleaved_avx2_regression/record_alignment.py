#!/usr/bin/env python3
"""Reverify packed record alignment without changing the previous AVX2 baseline."""
import json
from pathlib import Path
import sys
import tarfile

import reverify as verify
import run as kit

TEST = 'turbo_record_alignment_test'


def prepare(bundle, out):
    kit.verify_manifest(bundle)
    meta = json.loads((bundle / 'alignment-manifest.json').read_text())
    if meta['baseline_commit'] != 'e2ac9ee02a72b2da2265003c17b27357942a2d0d':
        raise RuntimeError('Unexpected baseline: use the already verified AVX2 version')
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
        kit.run(['git', 'init', '--quiet'], out / 'prepare.log', source)
        for patch in (['baseline-vs-head.patch'] +
                      (['candidate-vs-baseline.patch'] if version == 'after' else [])):
            kit.run(['git', 'apply', '--check', bundle / patch], out / 'prepare.log', source)
            kit.run(['git', 'apply', bundle / patch], out / 'prepare.log', source)
        expected = json.loads((bundle / (version + '-source-sha256.json')).read_text())
        actual = {str(p.relative_to(source)): verify.digest(p) for p in source.rglob('*')
                  if p.is_file() and '.git' not in p.relative_to(source).parts}
        if expected != actual:
            raise RuntimeError('Frozen source set/hash mismatch: ' + version)
        kit.dump(out / (version + '-source-sha256.json'), expected)
        sources[version] = source
    test_path = 'tests/turbo/' + TEST + '.cc'
    if (sources['before'] / test_path).read_bytes() != (sources['after'] / test_path).read_bytes():
        raise RuntimeError('Regression test must be identical on both sides')
    kit.dump(out / 'alignment-manifest.json', meta)
    return sources


def finalize(out, summary):
    # Previous diagnostics must be reproduced, and NONE may survive in candidate.
    # Do not turn "same as baseline" into acceptance of alignment UB.
    results = {v: json.loads((out / ('sanitized-' + v + '.json')).read_text())
               for v in ['before', 'after']}
    baseline = [d for r in results['before'].values() for d in r['diagnostics']]
    candidate = [d for r in results['after'].values() for d in r['diagnostics']]
    summary['alignment'] = dict(baseline_reports=len(baseline),
                                baseline_misaligned=sum('misaligned address' in d for d in baseline),
                                candidate_reports=len(candidate),
                                candidate_misaligned=sum('misaligned address' in d for d in candidate))
    if summary['native_linux_x86']:
        required = ['QuantizerPreservesPackedLayoutAtEveryOffset'] + [
            'SingleBatchAndManyMatchDecodedValues/' + str(i) for i in [0, 1, 3, 4, 5, 6]]
        for version in ['before', 'after']:
            for mode in ['correctness', 'sanitized']:
                for name in required:
                    verify.require_test(out / (mode + '-' + version) / TEST / 'results.xml', name)
    failures = any(r['exit_code'] or r['missing_xml'] or r['failed']
                   for tests in results.values() for r in tests.values())
    summary['status'] = (
        'CORRECTNESS_FAILED' if summary['correctness'] != 'PASS' else
        'SANITIZER_FAILED' if failures or candidate else
        'BASELINE_REPRODUCTION_FAILED' if not summary['alignment']['baseline_misaligned'] else
        'BASELINE_SANITIZER_REVIEW_REQUIRED' if any('misaligned address' not in d for d in baseline) else
        'PERFORMANCE_REVIEW_REQUIRED' if summary['performance']['sustained_regressions'] else
        'PASS_ALIGNMENT_FIX_NATIVE_X86' if summary['native_linux_x86'] else 'SMOKE_ONLY')


if __name__ == '__main__':
    sys.exit(verify.main(prepare_sources=prepare,
                         sanitizer_tests=[*verify.SAN_TESTS, TEST], finalize=finalize,
                         extra_tools=['record_alignment.py', 'RECORD_ALIGNMENT_zh.txt']))

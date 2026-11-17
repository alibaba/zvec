#!/usr/bin/env python3
"""Verify FP16 IP AUTO prefers AVX2 while explicit AVX512 stays available."""
import json
from pathlib import Path
import sys
import tarfile

import reverify as verify
import run as kit

BASELINE = '5f489fd4debbf4efc0b32d4e0e90d0741dc65a02'


def prepare(bundle, out):
    kit.verify_manifest(bundle)
    meta = json.loads((bundle / 'avx512-manifest.json').read_text())
    if meta['baseline_commit'] != BASELINE:
        raise RuntimeError('Use the post-record-alignment baseline 5f489fd4')
    if meta.get('dispatch_policy') != 'fp16_ip_auto_avx2':
        raise RuntimeError('Use the 20261117 FP16 IP AUTO policy handoff')
    sources = {}
    for version in ['before', 'after']:
        source = out / 'sources' / version
        source.mkdir(parents=True)
        with tarfile.open(bundle / 'source-base.tar.gz') as archive:
            for member in archive.getmembers():
                if (Path(member.name).is_absolute() or '..' in Path(member.name).parts
                        or not (member.isdir() or member.isfile())):
                    raise RuntimeError('Unsafe archive member: ' + member.name)
            archive.extractall(source, **({'filter': 'data'} if hasattr(tarfile, 'data_filter') else {}))
        kit.run(['git', 'init', '--quiet'], out / 'prepare.log', source)
        patches = ['baseline-vs-head.patch']
        if version == 'after':
            patches.append('candidate-vs-baseline.patch')
        for patch in patches:
            kit.run(['git', 'apply', '--check', bundle / patch], out / 'prepare.log', source)
            kit.run(['git', 'apply', bundle / patch], out / 'prepare.log', source)
        expected = json.loads((bundle / (version + '-source-sha256.json')).read_text())
        actual = {str(p.relative_to(source)): verify.digest(p) for p in source.rglob('*')
                  if p.is_file() and '.git' not in p.relative_to(source).parts}
        if actual != expected:
            raise RuntimeError('Frozen source set/hash mismatch: ' + version)
        kit.dump(out / (version + '-source-sha256.json'), actual)
        sources[version] = source
    tests = Path('tests/turbo/turbo_matrix_test.cc')
    if (sources['before'] / tests).read_bytes() != (sources['after'] / tests).read_bytes():
        raise RuntimeError('Both versions must use identical tests')
    kit.dump(out / 'avx512-manifest.json', meta)
    return sources


def finalize(out, summary):
    if summary['native_linux_x86']:
        kit.run([out / 'bin-after/interleaved_dispatch_probe', '--require-avx512-policy'],
                out / 'dispatch-avx512-policy-required.log')
        for mode in ['correctness', 'sanitized']:
            xml = out / (mode + '-after') / 'turbo_matrix_test/results.xml'
            for name in ['Avx512JointInterleavedQueriesAndNoAllocations',
                         'Avx2JointInterleavedQueriesAndNoAllocations',
                         'Avx512Fp16IpAutoUsesAvx2',
                         'InterleavedDispatchHonorsArchitectureAndLayout',
                         'InterleavedAvx512FeatureGatesAndFallback',
                         'InterleavedFp16AccumulatesInFp32']:
                verify.require_test(xml, name)
    # No sanitizer diagnostic, including old record alignment UB, is accepted.
    both = [json.loads((out / ('sanitized-' + v + '.json')).read_text())
            for v in ['before', 'after']]
    if summary.get('correctness') != 'PASS':
        summary['status'] = 'CORRECTNESS_FAILED'
    elif any(not r['ok'] for tests in both for r in tests.values()):
        summary['status'] = 'SANITIZER_FAILED'
    summary['avx512_policy_native_verified'] = summary['native_linux_x86']


if __name__ == '__main__':
    sys.exit(verify.main(prepare_sources=prepare,
                         sanitizer_tests=[*verify.SAN_TESTS, 'turbo_record_alignment_test'],
                         finalize=finalize,
                         extra_tools=['avx512.py', 'focused_fp16_ip.py',
                                      'AVX512_FP16_IP_REVERIFY_zh.txt']))

#!/usr/bin/env python3
"""Long paired ABBA/BAAB + A/A recheck of the five FP16 IP regressions."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import sys

import run as kit

CASES = ['Raw-FP16,IP,128,shared,8,32', 'Raw-FP16,IP,768,shared,1,32',
         'Raw-FP16,IP,768,shared,1,8', 'Raw-FP16,IP,768,shared,8,32',
         'Raw-FP16,IP,768,shared,8,8']


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def interval(values):
    mean = statistics.mean(values)
    half = 2.571 * statistics.stdev(values) / math.sqrt(6)
    return dict(mean=mean, lo=mean-half, hi=mean+half)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--cpu', type=int, required=True)
    p.add_argument('--sample-ms', type=float, default=1000)
    args = p.parse_args()
    if not sys.platform.startswith('linux') or platform.machine().lower() not in ('x86_64', 'amd64'):
        p.error('Native Linux x86_64 required')
    if args.cpu not in os.sched_getaffinity(0):
        p.error('CPU must be in allowed affinity')
    if not math.isfinite(args.sample_ms) or args.sample_ms < 1000:
        p.error('sample-ms must be >=1000')
    results, out = args.results.resolve(), args.out.resolve()
    bins = {'A': results/'bin-before/ivf_interleaved_bench',
            'B': results/'bin-after/ivf_interleaved_bench'}
    if not (results/'indexes').is_dir() or not all(x.is_file() for x in bins.values()):
        p.error('Run tools/avx512.py first; both binaries and indexes are required')
    out.mkdir(parents=True, exist_ok=False)
    runs = out/'runs'
    runs.mkdir()
    (runs/'indexes').symlink_to(results/'indexes', target_is_directory=True)
    meta = dict(sample_ms=args.sample_ms, cpu=args.cpu,
                binaries={k: dict(path=str(v), sha256=digest(v)) for k, v in bins.items()},
                indexes={x.name: digest(x) for x in (results/'indexes').glob('Raw-FP16_IP_*.index')},
                runs=[])
    summary = dict(status='RUNNING', cases={})
    env = {k: v for k, v in os.environ.items() if not k.startswith('GTEST_') and k != 'PREPARE'}
    env['BENCH_MIN_MS'] = str(args.sample_ms)
    blocks = {case: [] for case in CASES}
    try:
        for block in range(6):
            pattern = 'ABBA' if block % 2 == 0 else 'BAAB'
            order = CASES[block % 5:] + CASES[:block % 5]
            for case in order:
                slots = []
                for slot, side in enumerate(pattern + 'AA'):
                    directory = runs / f'r{len(meta["runs"])+1:03d}'
                    directory.mkdir()
                    (directory/'timings.csv').write_text(kit.HEADER)
                    entry = dict(case=case, block=block+1, pattern=pattern,
                                 slot=slot+1, side=side, run=directory.name)
                    kit.run([bins[side], '--gtest_filter=*ManyBenchmark*'],
                            directory/'run.log', directory,
                            dict(env, BENCH_CASE=case), args.cpu)
                    with (directory/'timings.csv').open() as f:
                        rows = list(csv.DictReader(f))
                    if len(rows) != 1 or ','.join(rows[0][k] for k in kit.KEYS) != case:
                        raise RuntimeError('Missing or wrong focused case')
                    value = float(rows[0]['ns_per_query'])
                    if not math.isfinite(value) or value <= 0:
                        raise RuntimeError('Invalid timing')
                    entry.update(ns=value, scans=rows[0]['distances'])
                    text = (directory/'query_results.txt').read_text()
                    if not text:
                        raise RuntimeError('Missing Top-K records')
                    slots.append((entry, text))
                    meta['runs'].append(entry)
                    kit.dump(out/'manifest.json', meta)
                if len({s[0]['scans'] for s in slots}) != 1 or len({s[1] for s in slots}) != 1:
                    raise RuntimeError('Focused Top-K/score/scan mismatch')
                aa = [s[0]['ns'] for s in slots[:4] if s[0]['side'] == 'A']
                bb = [s[0]['ns'] for s in slots[:4] if s[0]['side'] == 'B']
                blocks[case].append(dict(block=block+1, before_ns=statistics.mean(aa),
                                         after_ns=statistics.mean(bb), speedup=sum(aa)/sum(bb),
                                         aa_ratio=slots[4][0]['ns']/slots[5][0]['ns']))
        for case, data in blocks.items():
            ab = interval([x['speedup'] for x in data])
            aa = interval([x['aa_ratio'] for x in data])
            verdict = ('PASS' if ab['lo'] >= 1/1.05 else
                       'REGRESSION_CONFIRMED' if ab['hi'] < 1/1.05 else 'REVIEW_REQUIRED')
            summary['cases'][case] = dict(blocks=data, ab=ab, aa=aa, verdict=verdict)
        summary['status'] = ('PASS_FOCUSED' if all(x['verdict'] == 'PASS' for x in summary['cases'].values())
                             else 'PERFORMANCE_REVIEW_REQUIRED')
        return 0 if summary['status'] == 'PASS_FOCUSED' else 1
    except Exception as exc:
        summary.update(status='FAIL', error=str(exc))
        raise
    finally:
        kit.dump(out/'SUMMARY.json', summary)
        print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    sys.exit(main())

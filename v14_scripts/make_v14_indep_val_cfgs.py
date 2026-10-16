#!/usr/bin/env python3
"""Generate the four independent-set validation configs (task 4).

Each config is a byte-faithful copy of its frozen tuning-set production
yaml (int / rrf / native / natref winners, t1 tier) with exactly three
fields swapped to the independent artifacts:
  QueryFile        -> indep/queries_indep_q10m_1000.txt
  GroundTruthFile  -> indep/neighbors_indep_q10m_1000.txt
  RecallLogDir     -> unique per-scheme dir (never clobbers tuning logs)
Every other field (index path, query params, refiner config, thread
counts, timings) is identical to the production control run, so any
recall/QPS difference vs the tuning set is attributable to the queries.

Usage: make_v14_indep_val_cfgs.py --natref-np 170 --natref-topn 55 [--outdir DIR]
"""
import argparse
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
V12 = Path('/home/xingrui/bench/integrate_rabitq_in_hnsw/v12_runs/prod')
V13 = Path('/home/xingrui/bench/integrate_rabitq_in_hnsw/v13_runs')

SOURCES = {
    'int': lambda np_, n: V12 / 'v12_prod50_np293_t1.yaml',
    'rrf': lambda np_, n: V13 / 'v13_rrf55_np180_t1.yaml',
    'native': lambda np_, n: V13 / 'v13_native_np249_t1.yaml',
    'natref': lambda np_, n: HERE / f'v14_natref_np{np_}_top{n}_t1.yaml',
}

QUERY_RE = re.compile(r'^(\s*QueryFile:\s*).+$')
GT_RE = re.compile(r'^(\s*GroundTruthFile:\s*).+$')
LOG_RE = re.compile(r'^(\s*RecallLogDir:\s*).+$')


def transform(text, qfile, gtfile, logdir):
    out = []
    for line in text.splitlines():
        if QUERY_RE.match(line):
            out.append(QUERY_RE.match(line).group(1) + str(qfile))
        elif GT_RE.match(line):
            out.append(GT_RE.match(line).group(1) + str(gtfile))
        elif LOG_RE.match(line):
            out.append(LOG_RE.match(line).group(1) + str(logdir))
        else:
            out.append(line)
    return '\n'.join(out) + '\n'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--natref-np', type=int, required=True)
    ap.add_argument('--natref-topn', type=int, required=True)
    ap.add_argument('--outdir', default=str(HERE))
    args = ap.parse_args()
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    qfile = (out / 'indep' / 'queries_indep_q10m_900.txt').resolve()
    gtfile = (out / 'indep' / 'neighbors_indep_q10m_900.txt').resolve()
    for p in (qfile, gtfile):
        if not p.exists():
            raise SystemExit(f'missing {p} (run run_v14_gtgen.sh first)')

    for scheme, src_fn in SOURCES.items():
        src = src_fn(args.natref_np, args.natref_topn)
        if not src.exists():
            raise SystemExit(f'missing source yaml {src}')
        name = src.stem + '_indep'
        dst = out / f'{name}.yaml'
        dst.write_text(
            transform(src.read_text(encoding='utf-8'), qfile, gtfile,
                      f'./{name}.recall'),
            encoding='utf-8')
        print('wrote', dst)


if __name__ == '__main__':
    main()

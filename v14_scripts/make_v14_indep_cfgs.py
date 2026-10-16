#!/usr/bin/env python3
"""Generate the v14 independent-query-set artifacts (task 4).

Query source: the cohere 10m test split (cohere_test_vector_10m.1000.new.txt,
1000 queries). filter_v14_indep_queries.py removes the 100 queries that are
exact members of the 1M base (md5-verified; a self-indexed query is its own
trivial top-1) and renumbers the rest 0..899 -> queries_indep_q10m_900.txt.
The kept 900 are disjoint from the base AND from the 1000 tuning queries,
so they never took part in any v12/v13/v14 tuning.

Ground truth: exact top-100 per query over the 1M base, produced by searching
the kFlat reference index (FP32 brute force, the same index the refiner
fetches from) with the recall tool; the .knn logs are parsed into the
neighbors.txt text format. The recall run has no external GT, so the tool
also builds its internal GT by is_linear exact scan (must report Recall@100
= 100 as a self-consistency check). A numpy brute-force spot check on a
query sample cross-validates the GT independently (see run_v14_gtgen.sh).
"""
import argparse
from pathlib import Path

FLAT_REF = "/home/xingrui/bench/integrate_rabitq_in_hnsw/v12_runs/flat_ref/cohere_1m_flat_ref.index"
INDEP_QUERY_SRC = "/home/xingrui/bench/dataset/cohere/10m/cohere_test_vector_10m.1000.new.txt"

GTGEN = """IndexCommon:
  IndexConfig: "{{\\"index_type\\":\\"kFlat\\",\\"metric_type\\":\\"kCosine\\",\\"dimension\\":768,\\"data_type\\":\\"DT_FP32\\"}}"
  IndexPath: {flat_ref}
  TopK: 100
  QueryFile: {query_file}
  QueryType: float
  QueryFirstSep: ;
  QuerySecondSep: " "
  RecallLogDir: ./v14_gtgen_flat100_900.recall
  RecallThreadCount: 16
  RecallGTCount: 100
  RecallScorePrecision: 1e-4
  BenchThreadCount: 1
  BenchSecs: 30
  BenchIterCount: 10000000
  CompareById: true
  LogLevel: Info
QueryConfig:
  QueryParam: "{{\\"topk\\":100,\\"radius\\":0,\\"is_linear\\":false,\\"index_type\\":\\"kFlat\\",\\"fetch_vector\\":false}}"
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--outdir', default=str(Path(__file__).parent))
    args = ap.parse_args()
    out = Path(args.outdir)
    indep = out / 'indep'
    indep.mkdir(parents=True, exist_ok=True)
    qfile = indep / 'queries_indep_q10m_900.txt'
    if not qfile.exists():
        raise SystemExit(f'missing {qfile} (run filter_v14_indep_queries.py first)')
    gtgen = indep / 'v14_gtgen_flat100.yaml'
    gtgen.write_text(GTGEN.format(flat_ref=FLAT_REF, query_file=qfile), encoding='utf-8')
    print('wrote', gtgen)


if __name__ == '__main__':
    main()

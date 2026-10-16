#!/usr/bin/env python3
"""Generate v14 native+refine (dedicated kIVFRabitq + kFlat rerank) yamls.

Sweep grid for the v14 natref slot: coarse heap N in {52,55,60} selected
via ScaleFactor -- the C++ side computes coarse_topk = floor(50 * (float)sf)
(src/core/interface/index.cc _get_coarse_search_topk), so every sf is
searched and verified in float32 before use (naive 1.04 -> 51.99f -> 51).
TopK stays 50; N is the coarse candidate count the refiner reranks.
The tuning set is the same full 1000-query set used by v12/v13 tuning;
the 10m test queries are reserved for the post-freeze independent
validation and must NOT enter this sweep.
"""
import argparse
import math
import struct
from pathlib import Path

DED_INDEX = "/home/xingrui/bench/rabitq_refactor/ivf_dedicated/cohere_1m_ivfrabitq_dedicated.index"
FLAT_REF = "/home/xingrui/bench/integrate_rabitq_in_hnsw/v12_runs/flat_ref/cohere_1m_flat_ref.index"
QUERY_FULL = "/home/xingrui/bench/dataset/cohere/1m/cohere_test_vector_1000.new.txt"
GT_FULL = "/home/xingrui/bench/dataset/cohere/1m/neighbors.txt"

INDEX_JSON = ("{\\\"index_type\\\":\\\"kIVFRabitq\\\",\\\"metric_type\\\":\\\"kCosine\\\","
              "\\\"dimension\\\":768,\\\"data_type\\\":\\\"DT_FP32\\\",\\\"nlist\\\":1024,"
              "\\\"total_bits\\\":7,\\\"sample_count\\\":0}")


def f32(x):
    return struct.unpack('<f', struct.pack('<f', x))[0]


def coarse_topk(sf):
    return int(math.floor(f32(f32(50) * f32(sf))))


def find_scale_factor(n):
    sf = n / 50.0
    for _ in range(200):
        if coarse_topk(sf) == n:
            return round(sf, 6)
        sf += 0.001
    raise SystemExit('no scale factor found for N=%d' % n)


def query_json(nprobe):
    return ("{\\\"topk\\\":50,\\\"nprobe\\\":" + str(nprobe) +
            ",\\\"radius\\\":0,\\\"is_linear\\\":false,"
            "\\\"index_type\\\":\\\"kIVFRabitq\\\",\\\"fetch_vector\\\":false,"
            "\\\"rabitq_pruning\\\":true,\\\"rabitq_collect_stats\\\":false,"
            "\\\"rabitq_segment_timing\\\":false,\\\"rabitq_reuse_coarse_dot\\\":false,"
            "\\\"rabitq_export_scan_lists\\\":false,\\\"rabitq_export_scan_samples\\\":false}")


def render(name, nprobe, sf, threads):
    return """IndexCommon:
  IndexConfig: "{index_json}"
  IndexPath: {ded_index}
  TopK: 50
  QueryFile: {query_file}
  QueryType: float
  QueryFirstSep: ;
  QuerySecondSep: " "
  GroundTruthFile: {gt_file}
  RecallLogDir: ./{name}.recall
  RecallThreadCount: 16
  RecallScorePrecision: 1e-4
  BenchThreadCount: {threads}
  BenchSecs: 30
  BenchIterCount: 10000000
  CompareById: true
  LogLevel: Info
QueryConfig:
  QueryParam: "{query_json}"
  RefinerConfig:
    ScaleFactor: {sf}
    ReferenceIndex:
      Config: '{{"index_type":"kFlat","metric_type":"kCosine","dimension":768,"data_type":"DT_FP32"}}'
      Path: {flat_ref}
""".format(index_json=INDEX_JSON, ded_index=DED_INDEX, name=name,
           query_file=QUERY_FULL, gt_file=GT_FULL, threads=threads,
           query_json=query_json(nprobe), sf=sf, flat_ref=FLAT_REF)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--nps', required=True, help='comma list of nprobes, e.g. 150,160,170')
    ap.add_argument('--topns', required=True, help='comma list of coarse heaps N, e.g. 60')
    ap.add_argument('--threads', type=int, default=1)
    ap.add_argument('--outdir', default=str(Path(__file__).parent))
    args = ap.parse_args()
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    nps = [int(x) for x in args.nps.split(',')]
    topns = [int(x) for x in args.topns.split(',')]
    written = []
    for n in topns:
        sf = find_scale_factor(n)
        assert coarse_topk(sf) == n
        print('N=%d sf=%r -> coarse_topk=%d (f32-verified)' % (n, sf, coarse_topk(sf)))
        for np_ in nps:
            name = 'v14_natref_np%d_top%d_t%d' % (np_, n, args.threads)
            (out / (name + '.yaml')).write_text(render(name, np_, sf, args.threads))
            written.append(name)
    print('wrote %d yamls to %s' % (len(written), out))
    for w in written:
        print('  ' + w)


if __name__ == '__main__':
    main()

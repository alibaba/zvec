#!/usr/bin/env python3
"""Generate v13 native (dedicated kIVFRabitq) yaml configs:
- baseline (no refiner) at np249 (first nprobe with recall >= 99%)
- native+refine sweep yamls (RefinerConfig SF=1.2 -> coarse heap 60) at
  candidate nprobes below 249 where the refine may lift recall over 99%.
Each config gets a recall/timing yaml plus a holdout-half variant whose
QueryFile/GroundTruthFile point at the qid 500-999 split (see
make_v13_holdout.py). Bench thread count selects t1/t4 via --threads.
"""
import argparse
import os

DED_INDEX = "/home/xingrui/bench/rabitq_refactor/ivf_dedicated/cohere_1m_ivfrabitq_dedicated.index"
FLAT_REF = "/home/xingrui/bench/integrate_rabitq_in_hnsw/v12_runs/flat_ref/cohere_1m_flat_ref.index"
QUERY_FULL = "/home/xingrui/bench/dataset/cohere/1m/cohere_test_vector_1000.new.txt"
GT_FULL = "/home/xingrui/bench/dataset/cohere/1m/neighbors.txt"
QUERY_TUNE = "/home/xingrui/bench/integrate_rabitq_in_hnsw/v13_runs/holdout/queries_tune_q0_499.txt"
QUERY_HOLD = "/home/xingrui/bench/integrate_rabitq_in_hnsw/v13_runs/holdout/queries_holdout_q500_999.txt"
GT_TUNE = "/home/xingrui/bench/integrate_rabitq_in_hnsw/v13_runs/holdout/neighbors_tune_q0_499.txt"
GT_HOLD = "/home/xingrui/bench/integrate_rabitq_in_hnsw/v13_runs/holdout/neighbors_holdout_q500_999.txt"

BASE = """IndexCommon:
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
  QueryParam: "{query_json}"{refiner}
"""

INDEX_JSON = ("{\\\"index_type\\\":\\\"kIVFRabitq\\\",\\\"metric_type\\\":\\\"kCosine\\\","
              "\\\"dimension\\\":768,\\\"data_type\\\":\\\"DT_FP32\\\",\\\"nlist\\\":1024,"
              "\\\"total_bits\\\":7,\\\"sample_count\\\":0}")


def query_json(nprobe):
    return ("{\\\"topk\\\":50,\\\"nprobe\\\":" + str(nprobe) +
            ",\\\"radius\\\":0,\\\"is_linear\\\":false,"
            "\\\"index_type\\\":\\\"kIVFRabitq\\\",\\\"fetch_vector\\\":false,"
            "\\\"rabitq_pruning\\\":true,\\\"rabitq_collect_stats\\\":false,"
            "\\\"rabitq_segment_timing\\\":false,\\\"rabitq_reuse_coarse_dot\\\":false,"
            "\\\"rabitq_export_scan_lists\\\":false,\\\"rabitq_export_scan_samples\\\":false}")


REFINER = """
  RefinerConfig:
    ScaleFactor: 1.2
    ReferenceIndex:
      Config: '{ref_json}'
      Path: {flat_ref}"""

REF_JSON = ('{"index_type":"kFlat","metric_type":"kCosine",'
            '"dimension":768,"data_type":"DT_FP32"}')


def render(name, nprobe, refine, threads, query_file, gt_file):
    return BASE.format(
        index_json=INDEX_JSON, ded_index=DED_INDEX, name=name,
        query_file=query_file, gt_file=gt_file, threads=threads,
        query_json=query_json(nprobe),
        refiner=(REFINER.format(ref_json=REF_JSON, flat_ref=FLAT_REF) if refine else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=os.path.dirname(os.path.abspath(__file__)))
    args = ap.parse_args()
    out = args.outdir
    os.makedirs(out, exist_ok=True)

    made = []
    # native baseline: np249 (first >= 99% from the v8 sweep)
    for threads in (1, 4):
        name = "v13_native_np249_t%d" % threads
        with open(os.path.join(out, name + ".yaml"), "w") as f:
            f.write(render(name, 249, False, threads, QUERY_FULL, GT_FULL))
        made.append(name)
        hname = name + "_holdout"
        with open(os.path.join(out, hname + ".yaml"), "w") as f:
            f.write(render(hname, 249, False, threads, QUERY_HOLD, GT_HOLD))
        made.append(hname)

    # native+refine sweep: refine lifts recall, so lower nprobe may clear 99%
    for np in (200, 220, 232, 240, 244, 249):
        name = "v13_natref_np%d" % np
        with open(os.path.join(out, name + ".yaml"), "w") as f:
            f.write(render(name, np, True, 1, QUERY_FULL, GT_FULL))
        made.append(name)
        hname = name + "_holdout"
        with open(os.path.join(out, hname + ".yaml"), "w") as f:
            f.write(render(hname, np, True, 1, QUERY_HOLD, GT_HOLD))
        made.append(hname)
    for threads in (1, 4):
        for np in (200, 220, 232, 240, 244, 249):
            name = "v13_natref_np%d_t%d" % (np, threads)
            with open(os.path.join(out, name + ".yaml"), "w") as f:
                f.write(render(name, np, True, threads, QUERY_FULL, GT_FULL))
            made.append(name)
    print("wrote %d yamls to %s:" % (len(made), out))
    for m in made:
        print("  " + m)


if __name__ == "__main__":
    main()

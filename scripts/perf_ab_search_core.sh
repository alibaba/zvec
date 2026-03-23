#!/usr/bin/env bash
set -euo pipefail

DATASET="${1:-1m}"
CPU_CORE="${CPU_CORE:-0}"
REPEAT="${PERF_REPEAT:-5}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ZVEC_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CONDA_SH="${CONDA_SH:-/root/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-bench}"
PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ -f "${CONDA_SH}" ]]; then
  # shellcheck disable=SC1090
  source "${CONDA_SH}"
  conda activate "${CONDA_ENV}"
fi

PERF_EVENTS="cycles,instructions,branches,branch-misses,cache-references,cache-misses,L1-dcache-loads,L1-dcache-load-misses,LLC-loads,LLC-load-misses,dTLB-loads,dTLB-load-misses"

case "${DATASET}" in
  1m)
    CASE_TYPE="Performance768D1M"
    HNSW_PATH="${ZVEC_ROOT}/benchmark_results/cohere_1m_hnsw"
    OMEGA_PATH="${ZVEC_ROOT}/benchmark_results/cohere_1m_omega"
    HNSW_LABEL="16c64g-v0.1"
    OMEGA_LABEL="omega-m15-ef180-int8"
    HNSW_ARGS=(
      zvec
      --path "${HNSW_PATH}"
      --db-label "${HNSW_LABEL}"
      --case-type "${CASE_TYPE}"
      --m 15
      --ef-search 180
      --quantize-type int8
      --num-concurrency 16
      --concurrency-duration 30
      --k 100
      --skip-drop-old
      --skip-load
      --skip-search-concurrent
    )
    OMEGA_ARGS=(
      zvecomega
      --path "${OMEGA_PATH}"
      --db-label "${OMEGA_LABEL}"
      --case-type "${CASE_TYPE}"
      --m 15
      --ef-search 180
      --quantize-type int8
      --min-vector-threshold 100000
      --num-training-queries 4000
      --ef-training 500
      --window-size 100
      --ef-groundtruth 1000
      --target-recall 0.90
      --num-concurrency 16
      --concurrency-duration 30
      --k 100
      --skip-drop-old
      --skip-load
      --skip-search-concurrent
    )
    ;;
  10m)
    CASE_TYPE="Performance768D10M"
    HNSW_PATH="${ZVEC_ROOT}/benchmark_results/cohere_10m_hnsw"
    OMEGA_PATH="${ZVEC_ROOT}/benchmark_results/cohere_10m_omega"
    HNSW_LABEL="16c64g-v0.1"
    OMEGA_LABEL="omega-m50-ef118-int8-refiner"
    HNSW_ARGS=(
      zvec
      --path "${HNSW_PATH}"
      --db-label "${HNSW_LABEL}"
      --case-type "${CASE_TYPE}"
      --m 50
      --ef-search 118
      --quantize-type int8
      --is-using-refiner
      --num-concurrency 12,14,16,18,20
      --concurrency-duration 30
      --k 100
      --skip-drop-old
      --skip-load
      --skip-search-concurrent
    )
    OMEGA_ARGS=(
      zvecomega
      --path "${OMEGA_PATH}"
      --db-label "${OMEGA_LABEL}"
      --case-type "${CASE_TYPE}"
      --m 50
      --ef-search 118
      --quantize-type int8
      --is-using-refiner
      --min-vector-threshold 100000
      --num-training-queries 4000
      --ef-training 500
      --window-size 100
      --ef-groundtruth 1000
      --target-recall 0.90
      --num-concurrency 12,14,16,18,20
      --concurrency-duration 30
      --k 100
      --skip-drop-old
      --skip-load
      --skip-search-concurrent
    )
    ;;
  *)
    echo "Unsupported dataset: ${DATASET}" >&2
    echo "Usage: $0 [1m|10m]" >&2
    exit 1
    ;;
esac

run_perf() {
  local title="$1"
  shift

  echo
  echo "============================================================"
  echo "${title}"
  echo "============================================================"

  taskset -c "${CPU_CORE}" numactl --cpunodebind=0 --membind=0 \
    perf stat -r "${REPEAT}" -e "${PERF_EVENTS}" \
    "$@"
}

cd "${ZVEC_ROOT}"

run_perf \
  "HNSW core search perf (${DATASET})" \
  "${PYTHON_BIN}" -m vectordb_bench.cli.vectordbbench "${HNSW_ARGS[@]}"

run_perf \
  "OMEGA hooks-only core search perf (${DATASET})" \
  env ZVEC_OMEGA_DISABLE_MODEL_PREDICTION=1 \
  "${PYTHON_BIN}" -m vectordb_bench.cli.vectordbbench "${OMEGA_ARGS[@]}"

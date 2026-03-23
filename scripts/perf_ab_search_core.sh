#!/usr/bin/env bash
set -euo pipefail

DATASET="${1:-1m}"
CPU_CORE="${CPU_CORE:-0}"
REPEAT="${PERF_REPEAT:-5}"
MODE="${PERF_MODE:-all}"
RECORD_FREQ="${PERF_RECORD_FREQ:-999}"
TOPN="${PERF_TOPN:-60}"
CALL_GRAPH_MODE="${PERF_CALL_GRAPH_MODE:-fp}"
PERF_USER_ONLY="${PERF_USER_ONLY:-1}"

OPENBLAS_THREADS="${OPENBLAS_NUM_THREADS:-1}"
OMP_THREADS="${OMP_NUM_THREADS:-1}"
MKL_THREADS="${MKL_NUM_THREADS:-1}"
NUMEXPR_THREADS="${NUMEXPR_NUM_THREADS:-1}"
GOTO_THREADS="${GOTO_NUM_THREADS:-1}"
VECLIB_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ZVEC_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUT_DIR="${PERF_OUT_DIR:-${ZVEC_ROOT}/perf_results/${DATASET}}"

CONDA_SH="${CONDA_SH:-/root/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-bench}"
PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ -f "${CONDA_SH}" ]]; then
  # shellcheck disable=SC1090
  source "${CONDA_SH}"
  conda activate "${CONDA_ENV}"
fi

PERF_EVENTS="cycles,instructions,branches,branch-misses,cache-references,cache-misses,L1-dcache-loads,L1-dcache-load-misses,LLC-loads,LLC-load-misses,dTLB-loads,dTLB-load-misses"

COMMON_ENV=(
  env
  OPENBLAS_NUM_THREADS="${OPENBLAS_THREADS}"
  OMP_NUM_THREADS="${OMP_THREADS}"
  MKL_NUM_THREADS="${MKL_THREADS}"
  NUMEXPR_NUM_THREADS="${NUMEXPR_THREADS}"
  GOTO_NUM_THREADS="${GOTO_THREADS}"
  VECLIB_MAXIMUM_THREADS="${VECLIB_THREADS}"
)

PERF_RECORD_ARGS=(-F "${RECORD_FREQ}" -g --call-graph "${CALL_GRAPH_MODE}")
if [[ "${PERF_USER_ONLY}" == "1" ]]; then
  PERF_RECORD_ARGS+=(--all-user)
fi

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

run_record() {
  local title="$1"
  local output_prefix="$2"
  shift 2

  local data_file="${OUT_DIR}/${output_prefix}.data"
  local report_file="${OUT_DIR}/${output_prefix}.report.txt"
  local zvec_report_file="${OUT_DIR}/${output_prefix}.zvec_only.report.txt"

  echo
  echo "============================================================"
  echo "${title}"
  echo "============================================================"
  echo "perf.data: ${data_file}"
  echo "report:    ${report_file}"
  echo "zvec-only: ${zvec_report_file}"

  taskset -c "${CPU_CORE}" numactl --cpunodebind=0 --membind=0 \
    perf record "${PERF_RECORD_ARGS[@]}" -o "${data_file}" -- \
    "$@"

  perf report --stdio --no-children -i "${data_file}" --percent-limit 0.5 \
    > "${report_file}"
  sed -n "1,${TOPN}p" "${report_file}"

  perf report --stdio --no-children -i "${data_file}" \
    --sort dso,symbol --percent-limit 0.1 \
    --dsos _zvec.cpython-311-x86_64-linux-gnu.so \
    > "${zvec_report_file}"
  sed -n "1,${TOPN}p" "${zvec_report_file}"
}

cd "${ZVEC_ROOT}"
mkdir -p "${OUT_DIR}"

HNSW_CMD=(
  "${COMMON_ENV[@]}"
  "${PYTHON_BIN}" -m vectordb_bench.cli.vectordbbench "${HNSW_ARGS[@]}"
)

HNSW_EMPTY_HOOKS_CMD=(
  "${COMMON_ENV[@]}"
  ZVEC_HNSW_ENABLE_EMPTY_HOOKS=1
  "${PYTHON_BIN}" -m vectordb_bench.cli.vectordbbench "${HNSW_ARGS[@]}"
)

OMEGA_HOOKS_CMD=(
  "${COMMON_ENV[@]}"
  ZVEC_OMEGA_DISABLE_MODEL_PREDICTION=1
  "${PYTHON_BIN}" -m vectordb_bench.cli.vectordbbench "${OMEGA_ARGS[@]}"
)

echo "Using thread env:"
echo "  OPENBLAS_NUM_THREADS=${OPENBLAS_THREADS}"
echo "  OMP_NUM_THREADS=${OMP_THREADS}"
echo "  MKL_NUM_THREADS=${MKL_THREADS}"
echo "  NUMEXPR_NUM_THREADS=${NUMEXPR_THREADS}"
echo "  GOTO_NUM_THREADS=${GOTO_THREADS}"
echo "  VECLIB_MAXIMUM_THREADS=${VECLIB_THREADS}"
echo "  PERF_CALL_GRAPH_MODE=${CALL_GRAPH_MODE}"
echo "  PERF_USER_ONLY=${PERF_USER_ONLY}"

case "${MODE}" in
  stat)
    run_perf \
      "HNSW core search perf (${DATASET})" \
      "${HNSW_CMD[@]}"

    run_perf \
      "HNSW empty-hooks core search perf (${DATASET})" \
      "${HNSW_EMPTY_HOOKS_CMD[@]}"

    run_perf \
      "OMEGA hooks-only core search perf (${DATASET})" \
      "${OMEGA_HOOKS_CMD[@]}"
    ;;
  record)
    run_record \
      "HNSW core search hotspots (${DATASET})" \
      "hnsw_core" \
      "${HNSW_CMD[@]}"

    run_record \
      "HNSW empty-hooks core search hotspots (${DATASET})" \
      "hnsw_empty_hooks" \
      "${HNSW_EMPTY_HOOKS_CMD[@]}"

    run_record \
      "OMEGA hooks-only core search hotspots (${DATASET})" \
      "omega_hooks_only" \
      "${OMEGA_HOOKS_CMD[@]}"
    ;;
  all)
    run_perf \
      "HNSW core search perf (${DATASET})" \
      "${HNSW_CMD[@]}"

    run_perf \
      "HNSW empty-hooks core search perf (${DATASET})" \
      "${HNSW_EMPTY_HOOKS_CMD[@]}"

    run_perf \
      "OMEGA hooks-only core search perf (${DATASET})" \
      "${OMEGA_HOOKS_CMD[@]}"

    run_record \
      "HNSW core search hotspots (${DATASET})" \
      "hnsw_core" \
      "${HNSW_CMD[@]}"

    run_record \
      "HNSW empty-hooks core search hotspots (${DATASET})" \
      "hnsw_empty_hooks" \
      "${HNSW_EMPTY_HOOKS_CMD[@]}"

    run_record \
      "OMEGA hooks-only core search hotspots (${DATASET})" \
      "omega_hooks_only" \
      "${OMEGA_HOOKS_CMD[@]}"
    ;;
  *)
    echo "Unsupported PERF_MODE: ${MODE}" >&2
    echo "Use PERF_MODE=stat|record|all" >&2
    exit 1
    ;;
esac

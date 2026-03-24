#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ZVEC_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

BIN="${BIN:-${ZVEC_ROOT}/build/bin/hnsw_hooks_microbench}"
INDEX_PATH="${INDEX_PATH:-${ZVEC_ROOT}/benchmark_results/cohere_1m_hnsw/0/dense.qindex.5.proxima}"
OUT_DIR="${OUT_DIR:-${ZVEC_ROOT}/perf_results/hnsw_hooks_microbench}"

CPU_CORE="${CPU_CORE:-0}"
REPEAT="${REPEAT:-5}"
EVENTS="${EVENTS:-cycles,instructions,branches,branch-misses,cache-references,cache-misses}"
RECORD_FREQ="${RECORD_FREQ:-999}"
CALL_GRAPH_MODE="${CALL_GRAPH_MODE:-fp}"
TOPN="${TOPN:-80}"
MODE_FILTER="${MODE_FILTER:-all}"

QUERY_COUNT="${QUERY_COUNT:-1000}"
WARMUP="${WARMUP:-200}"
ITERATIONS="${ITERATIONS:-2000}"
EF_SEARCH="${EF_SEARCH:-180}"
TOPK="${TOPK:-100}"
WINDOW_SIZE="${WINDOW_SIZE:-100}"
TARGET_RECALL="${TARGET_RECALL:-0.91}"
SEED="${SEED:-12345}"

if ! command -v perf >/dev/null 2>&1; then
  echo "perf not found in PATH" >&2
  exit 1
fi

if [[ ! -x "${BIN}" ]]; then
  echo "microbench binary not found: ${BIN}" >&2
  exit 1
fi

if [[ ! -f "${INDEX_PATH}" ]]; then
  echo "index file not found: ${INDEX_PATH}" >&2
  exit 1
fi

mkdir -p "${OUT_DIR}"

COMMON_ARGS=(
  "${BIN}"
  --index-path "${INDEX_PATH}"
  --ef-search "${EF_SEARCH}"
  --topk "${TOPK}"
  --query-count "${QUERY_COUNT}"
  --warmup "${WARMUP}"
  --iterations "${ITERATIONS}"
  --window-size "${WINDOW_SIZE}"
  --target-recall "${TARGET_RECALL}"
  --seed "${SEED}"
)

run_stat() {
  local mode="$1"
  echo
  echo "============================================================"
  echo "perf stat: ${mode}"
  echo "============================================================"
  taskset -c "${CPU_CORE}" perf stat -r "${REPEAT}" -e "${EVENTS}" \
    "${COMMON_ARGS[@]}" --mode "${mode}"
}

run_record() {
  local mode="$1"
  local data_file="${OUT_DIR}/${mode}.data"
  local report_file="${OUT_DIR}/${mode}.report.txt"
  local zvec_report_file="${OUT_DIR}/${mode}.zvec_only.report.txt"

  echo
  echo "============================================================"
  echo "perf record: ${mode}"
  echo "============================================================"
  echo "perf.data: ${data_file}"
  echo "report:    ${report_file}"
  echo "zvec-only: ${zvec_report_file}"

  taskset -c "${CPU_CORE}" perf record -F "${RECORD_FREQ}" -g \
    --call-graph "${CALL_GRAPH_MODE}" -o "${data_file}" -- \
    "${COMMON_ARGS[@]}" --mode "${mode}"

  perf report --stdio --no-children -i "${data_file}" --percent-limit 0.3 \
    > "${report_file}"
  sed -n "1,${TOPN}p" "${report_file}"

  perf report --stdio --no-children -i "${data_file}" \
    --sort dso,symbol --percent-limit 0.05 > "${zvec_report_file}"
  sed -n "1,${TOPN}p" "${zvec_report_file}"
}

run_mode() {
  local mode="$1"
  run_stat "${mode}"
  run_record "${mode}"
}

case "${MODE_FILTER}" in
  all)
    run_mode fast
    run_mode empty
    run_mode omega
    ;;
  fast|empty|omega)
    run_mode "${MODE_FILTER}"
    ;;
  *)
    echo "Unsupported MODE_FILTER: ${MODE_FILTER}" >&2
    exit 1
    ;;
esac

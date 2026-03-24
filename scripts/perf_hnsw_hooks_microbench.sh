#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ZVEC_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

BIN="${BIN:-${ZVEC_ROOT}/build/bin/hnsw_hooks_microbench}"
DEFAULT_INDEX_DIR="${ZVEC_ROOT}/benchmark_results/cohere_1m_hnsw"
INDEX_PATH="${INDEX_PATH:-}"
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

mkdir -p "${OUT_DIR}"

build_common_args() {
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
}

preflight_index() {
  local candidate="$1"
  [[ -d "${candidate}" ]] || return 1

  local output
  if ! output="$("${BIN}" \
      --index-path "${candidate}" \
      --ef-search "${EF_SEARCH}" \
      --topk "${TOPK}" \
      --query-count 8 \
      --warmup 2 \
      --iterations 4 \
      --window-size "${WINDOW_SIZE}" \
      --target-recall "${TARGET_RECALL}" \
      --seed "${SEED}" \
      --mode fast 2>&1)"; then
    return 1
  fi

  if [[ "${output}" != *"doc_cnt="* ]] || [[ "${output}" == *"doc_cnt=0"* ]]; then
    return 1
  fi

  echo "${output}"
  return 0
}

detect_index_path() {
  if [[ -n "${INDEX_PATH}" ]]; then
    if [[ ! -d "${INDEX_PATH}" ]]; then
      echo "index dir not found: ${INDEX_PATH}" >&2
      exit 1
    fi
    local output
    if ! output="$(preflight_index "${INDEX_PATH}")"; then
      echo "preflight failed for INDEX_PATH=${INDEX_PATH}" >&2
      exit 1
    fi
    echo "Using user-provided INDEX_PATH=${INDEX_PATH}"
    echo "${output}"
    return
  fi

  local candidates=(
    "${DEFAULT_INDEX_DIR}"
  )
  local candidate
  for candidate in "${candidates[@]}"; do
    local output
    if output="$(preflight_index "${candidate}")"; then
      INDEX_PATH="${candidate}"
      echo "Auto-detected INDEX_PATH=${INDEX_PATH}"
      echo "${output}"
      return
    fi
  done

  echo "Failed to auto-detect a valid index file under ${DEFAULT_INDEX_DIR}" >&2
  exit 1
}

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

detect_index_path
build_common_args

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

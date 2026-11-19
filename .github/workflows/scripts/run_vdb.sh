#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
WORKSPACE=${GITHUB_WORKSPACE:-$(cd "$SCRIPT_DIR/../../.." && pwd)}
VDBBENCH_ROOT=${VDBBENCH_ROOT:-/opt/VectorDBBench}
export DATASET_LOCAL_DIR=${DATASET_LOCAL_DIR:-/tmp/vectordb_bench/dataset}
DRY_RUN=false
if [[ ${1:-} == --dry-run && $# == 1 ]]; then
    DRY_RUN=true
elif [[ $# != 0 ]]; then
    echo "Usage: bash $0 [--dry-run]" >&2
    exit 2
fi

QUANTIZE_TYPE_LIST="int8 int4 fp16 fp32"
# The original three cases retain their parameters and monitoring group keys.
CASE_TYPE_LIST="Performance768D1M Performance768D10M Performance1536D500K GIST1ML2 Text2Image1MIP"
DATE=$(date +%Y-%m-%d_%H-%M-%S)
NPROC=$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 2)
REVISION=${GITHUB_SHA:-$(git -C "$WORKSPACE" rev-parse HEAD)}
BRANCH=${GITHUB_REF_NAME:-local}
COMMIT_ID="${BRANCH//\//_}-$DATE-${REVISION:0:8}"
DB_LABEL_PREFIX="Zvec16c64g-$COMMIT_ID"
RUN_ROOT=${BENCH_RUN_ROOT:-$WORKSPACE/bench-runs/$COMMIT_ID}
PUSHGATEWAY_URL=${PUSHGATEWAY_URL:-http://47.93.34.27:9091}

if ! $DRY_RUN; then
    cd "$WORKSPACE"
    git submodule update --init
    python3 -m venv .venv
    source .venv/bin/activate
    pip install cmake ninja psycopg2-binary loguru fire
    pip install -e "$VDBBENCH_ROOT" -r "$SCRIPT_DIR/requirements-datasets.txt"
    CMAKE_GENERATOR="Unix Makefiles" CMAKE_BUILD_PARALLEL_LEVEL="$NPROC" \
        pip install -v "$WORKSPACE"
    # Runs before any benchmark timing; completed datasets are verified and reused.
    python "$SCRIPT_DIR/prepare_datasets.py" --root "$DATASET_LOCAL_DIR"
fi

for BENCH_CASE in $CASE_TYPE_LIST; do
    CASE_TYPE=$BENCH_CASE
    DATASET_ARGS=()
    case "$BENCH_CASE" in
        Performance768D1M)
            DATASET_DESC="Performance768D1M - Cohere Cosine"
            INDEX_ARGS=(--m 15 --ef-search 180)
            ;;
        Performance768D10M)
            DATASET_DESC="Performance768D10M - Cohere Cosine"
            INDEX_ARGS=(--m 50 --ef-search 118 --is-using-refiner)
            ;;
        Performance1536D500K)
            DATASET_DESC="Performance1536D500K - OpenAI IP"
            INDEX_ARGS=(--m 50 --ef-search 100 --is-using-refiner)
            ;;
        GIST1ML2|Text2Image1MIP)
            CASE_TYPE=PerformanceCustomDataset
            if [[ $BENCH_CASE == GIST1ML2 ]]; then
                DATASET_NAME=gist
                DIM=960
                METRIC=L2
            else
                DATASET_NAME=text2image
                DIM=200
                METRIC=IP
            fi
            DATASET_DESC="$BENCH_CASE - $METRIC"
            DATASET_ARGS=(
                --custom-case-name "$BENCH_CASE"
                --custom-dataset-name "$DATASET_NAME"
                --custom-dataset-dir "${DATASET_NAME}_1m_v1"
                --custom-dataset-size 1000000
                --custom-dataset-dim "$DIM"
                --custom-dataset-metric-type "$METRIC"
                --custom-dataset-file-count 1
                --skip-custom-dataset-use-shuffled
                --custom-dataset-with-gt
                --k 100
            )
            # Initial fixed baseline; tune against recall before interpreting regressions.
            INDEX_ARGS=(--m 32 --ef-search 200)
            ;;
    esac
    DATASET_ARGS+=(--case-type "$CASE_TYPE")

    for QUANTIZE_TYPE in $QUANTIZE_TYPE_LIST; do
        DB_LABEL="$DB_LABEL_PREFIX-$BENCH_CASE-$QUANTIZE_TYPE"
        CASE_RUN_DIR="$RUN_ROOT/$BENCH_CASE/$QUANTIZE_TYPE"
        VDB_PARAMS=(
            --path "$CASE_RUN_DIR/index" --db-label "$DB_LABEL"
            --num-concurrency 12,14,16,18,20
            "${DATASET_ARGS[@]}" "${INDEX_ARGS[@]}"
        )
        if [[ $QUANTIZE_TYPE != fp32 ]]; then
            VDB_PARAMS+=(--quantize-type "$QUANTIZE_TYPE")
        fi
        if $DRY_RUN; then
            printf '%q ' vectordbbench zvec "${VDB_PARAMS[@]}"
            printf '\n'
            continue
        fi
        mkdir -p "$CASE_RUN_DIR"
        export RESULTS_LOCAL_DIR="$CASE_RUN_DIR/results"
        vectordbbench zvec "${VDB_PARAMS[@]}" 2>&1 | tee "$CASE_RUN_DIR/bench.log"
        # Scope discovery to this invocation rather than scraping an installation path.
        RESULT_JSON_PATH=$(python - "$RESULTS_LOCAL_DIR" <<'PY'
import sys
from pathlib import Path

files = list(Path(sys.argv[1]).rglob("result_*.json"))
if len(files) != 1:
    raise SystemExit(f"Expected one result JSON, found {len(files)}")
print(files[0])
PY
        )
        QPS=$(jq -er '.results[0].metrics.qps | numbers' "$RESULT_JSON_PATH")
        RECALL=$(jq -er '.results[0].metrics.recall | numbers' "$RESULT_JSON_PATH")
        LATENCY_P99=$(jq -er '.results[0].metrics.serial_latency_p99 | numbers' "$RESULT_JSON_PATH")
        LOAD_DURATION=$(jq -er '.results[0].metrics.load_duration | numbers' "$RESULT_JSON_PATH")

        # BENCH_CASE distinguishes the two custom datasets in labels AND grouping keys.
        label_list="case_type=\"${BENCH_CASE}\",dataset_desc=\"${DATASET_DESC}\",db_label=\"${DB_LABEL}\",commit=\"${COMMIT_ID}\",date=\"${DATE}\",quantize_type=\"${QUANTIZE_TYPE}\""
        cat > "$CASE_RUN_DIR/prom_metrics.txt" <<EOF
# TYPE vdb_bench_qps gauge
vdb_bench_qps{$label_list} $QPS
# TYPE vdb_bench_recall gauge
vdb_bench_recall{$label_list} $RECALL
# TYPE vdb_bench_latency_p99 gauge
vdb_bench_latency_p99{$label_list} $LATENCY_P99
# TYPE vdb_bench_load_duration gauge
vdb_bench_load_duration{$label_list} $LOAD_DURATION
EOF
        curl --fail --show-error --data-binary "@$CASE_RUN_DIR/prom_metrics.txt" \
            "$PUSHGATEWAY_URL/metrics/job/benchmarks-${BENCH_CASE}/case_type/${BENCH_CASE}/quantize_type/${QUANTIZE_TYPE}"
    done
done

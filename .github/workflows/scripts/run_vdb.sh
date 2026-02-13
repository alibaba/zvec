set -e

QUANTIZE_TYPE="int8 int4 fp16 fp32"
CASE_TYPE_LIST="Performance768D1M Performance1536D500K" # respectively test cosine, ip # Performance960D1M l2 metrics
LOG_FILE="bench.log"
DATE=$(date +%Y-%m-%d)
NPROC=$(nproc 2>/dev/null || getconf _NPROCESSORS_ONLN 2>/dev/null || echo 2)

# COMMIT_ID = branch-date-sha
COMMIT_ID=${GITHUB_REF_NAME}-"$DATE"-$(echo ${GITHUB_WORKFLOW_SHA} | cut -c1-8)
echo "COMMIT_ID: $COMMIT_ID"
echo "GITHUB_WORKFLOW_SHA: $GITHUB_WORKFLOW_SHA"
echo "workspace: $GITHUB_WORKSPACE"
DB_LABEL_PREFIX="Zvec16c64g-$COMMIT_ID"

# install zvec
git submodule update --init

python3 -m venv .venv
source .venv/bin/activate
pip install cmake ninja psycopg2-binary loguru fire
pip install -e /opt/VectorDBBench

CMAKE_GENERATOR="Unix Makefiles" \
CMAKE_BUILD_PARALLEL_LEVEL="$NPROC" \
pip install -v .

for CASE_TYPE in $CASE_TYPE_LIST; do
    DATASET_DESC=""
    if [ "$CASE_TYPE" == "Performance960D1M" ]; then
        DATASET_DESC="Performance768D1M - Cohere1M768 Cosine"
    else
        DATASET_DESC="Performance1536D500K - OpenAI500K1536 IP"
    fi
    for QUANTIZE_TYPE in $QUANTIZE_TYPE_LIST; do
        DB_LABEL="$DB_LABEL_PREFIX-$CASE_TYPE-$QUANTIZE_TYPE"
        vectordbbench zvec --path "${DB_LABEL}" \
        --db-label "${DB_LABEL}" \
        --case-type "${CASE_TYPE}" \
        --num-concurrency 12,16,20,30 \
        --quantize-type "${QUANTIZE_TYPE}" --m 50 --ef-search 118 \
        --is-using-refiner 2>&1 | tee $LOG_FILE

        RESULT_JSON_PATH=$(grep -o "/opt/VectorDBBench/.*\.json" $LOG_FILE)
        QPS=$(jq -r '.results[0].metrics.qps' "$RESULT_JSON_PATH")
        RECALL=$(jq -r '.results[0].metrics.recall' "$RESULT_JSON_PATH")
        LATENCY_P99=$(jq -r '.results[0].metrics.serial_latency_p99' "$RESULT_JSON_PATH")

        label_list="case_type=${CASE_TYPE} dataset_desc=${DATASET_DESC} db_label=${DB_LABEL} commit=${COMMIT_ID} date=${DATE} quantize_type=${QUANTIZE_TYPE}"
        cat <<EOF > prom_metrics.txt
        # TYPE vdb_bench_qps gauge
        vdb_bench_qps{$label_list} $QPS
        # TYPE vdb_bench_recall gauge
        vdb_bench_recall{$label_list} $RECALL
        # TYPE vdb_bench_latency_p99 gauge
        vdb_bench_latency_p99{$label_list} $LATENCY_P99
EOF

        curl --data-binary @prom_metrics.txt "http://47.93.34.27:9091/metrics/job/benchmarks/date/${DATE}/commit/${COMMIT_ID}/case_type/${CASE_TYPE}/quantize_type/${QUANTIZE_TYPE}/" -v
    done
done
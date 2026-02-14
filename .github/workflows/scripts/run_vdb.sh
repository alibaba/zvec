set -e

QUANTIZE_TYPE_LIST="int8 int4 fp16 fp32"
CASE_TYPE_LIST="Performance768D1M Performance1536D500K" # respectively test cosine, ip # Performance960D1M l2 metrics
LOG_FILE="bench.log"
DATE=$(date +%Y-%m-%d_%H-%M-%S)
NPROC=$(nproc 2>/dev/null || getconf _NPROCESSORS_ONLN 2>/dev/null || echo 2)

# COMMIT_ID = branch-date-sha
COMMIT_ID=${GITHUB_REF_NAME}-"$DATE"-$(echo ${GITHUB_WORKFLOW_SHA} | cut -c1-8)
COMMIT_ID=$(echo "$COMMIT_ID" | sed 's/\//_/g')
echo "COMMIT_ID: $COMMIT_ID"
echo "GITHUB_WORKFLOW_SHA: $GITHUB_WORKFLOW_SHA"
echo "workspace: $GITHUB_WORKSPACE"
DB_LABEL_PREFIX="Zvec16c64g-$COMMIT_ID"

# install zvec
git submodule update --init

#cd .. # for debug
pwd

python3 -m venv .venv
source .venv/bin/activate
pip install cmake ninja psycopg2-binary loguru fire
pip install -e /opt/VectorDBBench

CMAKE_GENERATOR="Unix Makefiles" \
CMAKE_BUILD_PARALLEL_LEVEL="$NPROC" \
SKBUILD_BUILD_DIR="$GITHUB_WORKSPACE/../build" \
pip install -v "$GITHUB_WORKSPACE"

for CASE_TYPE in $CASE_TYPE_LIST; do
    DATASET_DESC=""
    if [ "$CASE_TYPE" == "Performance768D1M" ]; then
        DATASET_DESC="Performance768D1M - Cohere Cosine"
    else
        DATASET_DESC="Performance1536D500K - OpenAI IP"
    fi
    for QUANTIZE_TYPE in $QUANTIZE_TYPE_LIST; do
        DB_LABEL="$DB_LABEL_PREFIX-$CASE_TYPE-$QUANTIZE_TYPE"
        echo "Running VectorDBBench for $DB_LABEL"
        # if quantize = fp32, do not pass --is-using-refiner
        if [ "$QUANTIZE_TYPE" == "fp32" ]; then
            vectordbbench zvec --path "${DB_LABEL}" \
              --db-label "${DB_LABEL}" \
              --case-type "${CASE_TYPE}" \
              --num-concurrency 12,16,20 \
              --m 50 --ef-search 118 2>&1 | tee $LOG_FILE
        else
            vectordbbench zvec --path "${DB_LABEL}" \
              --db-label "${DB_LABEL}" \
              --case-type "${CASE_TYPE}" \
              --num-concurrency 12,16,20,30 \
              --quantize-type "${QUANTIZE_TYPE}" --m 50 --ef-search 118 \
              --is-using-refiner 2>&1 | tee $LOG_FILE
        fi

        RESULT_JSON_PATH=$(grep -o "/opt/VectorDBBench/.*\.json" $LOG_FILE)
        QPS=$(jq -r '.results[0].metrics.qps' "$RESULT_JSON_PATH")
        RECALL=$(jq -r '.results[0].metrics.recall' "$RESULT_JSON_PATH")
        LATENCY_P99=$(jq -r '.results[0].metrics.serial_latency_p99' "$RESULT_JSON_PATH")
        LOAD_DURATION=$(jq -r '.results[0].metrics.load_duration' "$RESULT_JSON_PATH")

        #quote the var to avoid space in the label
        label_list="case_type=\"${CASE_TYPE}\",dataset_desc=\"${DATASET_DESC}\",db_label=\"${DB_LABEL}\",commit=\"${COMMIT_ID}\",date=\"${DATE}\",quantize_type=\"${QUANTIZE_TYPE}\""
        # replace `/` with `_` in label_list
        label_list=$(echo "$label_list" | sed 's/\//_/g')
        cat <<EOF > prom_metrics.txt
        # TYPE vdb_bench_qps gauge
        vdb_bench_qps{$label_list} $QPS
        # TYPE vdb_bench_recall gauge
        vdb_bench_recall{$label_list} $RECALL
        # TYPE vdb_bench_latency_p99 gauge
        vdb_bench_latency_p99{$label_list} $LATENCY_P99
        # TYPE vdb_bench_load_duration gauge
        vdb_bench_load_duration{$label_list} $LOAD_DURATION
EOF

        curl --data-binary @prom_metrics.txt "http://47.93.34.27:9091/metrics/job/benchmarks-${CASE_TYPE}/case_type/${CASE_TYPE}/quantize_type/${QUANTIZE_TYPE}" -v
    done
done
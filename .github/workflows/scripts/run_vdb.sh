set -e

COMMIT_ID=$(echo ${GITHUB_WORKFLOW_SHA:-$(git rev-parse HEAD)} | cut -c1-8)
echo "COMMIT_ID: $COMMIT_ID"
echo "GITHUB_WORKFLOW_SHA: $GITHUB_WORKFLOW_SHA"
echo "workspace: $GITHUB_WORKSPACE"


DATASET_LIST="Performance768D1M Performance1536D500K Performance960D1M" # respectively test cosine, ip, l2 metrics
DATE=$(date +%Y-%m-%d)
DB_LABEL_PREFIX="Zvec16c64g-$COMMIT_ID"
LOG_FILE="bench.log"
NPROC=$(nproc 2>/dev/null || getconf _NPROCESSORS_ONLN 2>/dev/null || echo 2)

# install zvec
git submodule update --init

python3 -m venv .venv
source .venv/bin/activate
pip install cmake ninja psycopg2-binary

python .github/workflows/scripts/upload_bench_results.py

# CMAKE_GENERATOR="Unix Makefiles" \
# CMAKE_BUILD_PARALLEL_LEVEL="$NPROC" \
# pip install -v .

# # 1. Execute bench and capture the result path from logs
# # We use 'tee' to see output and 'grep' to find the result file path
# pip install -e /opt/VectorDBBench

# for DATASET in $DATASET_LIST; do
#     DB_LABEL="$DB_LABEL_PREFIX-$DATASET"
#     vectordbbench zvec --path $DATASET \
#     --db-label $DB_LABEL \
#     --case-type $DATASET \
#     --num-concurrency 12,14,16,18,20 \
#     --quantize-type int8 --m 50 --ef-search 118 \
#     --is-using-refiner 2>&1 | tee $LOG_FILE

#     # 2. Extract the JSON result path from the log
#     RESULT_JSON=$(grep -o "/opt/VectorDBBench/.*\.json" $LOG_FILE)

#     # 3. Parse metrics using jq
#     # Note: VectorDBBench JSON structure usually contains a list of results
#     QPS=$(jq -r '.results[0].metrics.qps' $RESULT_JSON)
#     RECALL=$(jq -r '.results[0].metrics.recall' $RESULT_JSON)
#     LATENCY_P99=$(jq -r '.results[0].metrics.serial_latency_p99' $RESULT_JSON)

#     # 4. Push to Prometheus Pushgateway
#     cat <<EOF > prom_metrics.txt
#     # TYPE vdb_bench_qps gauge
#     vdb_bench_qps{dataset="$DATASET",db_label="$DB_LABEL",commit="$COMMIT_ID",date="$DATE"} $QPS
#     # TYPE vdb_bench_recall gauge
#     vdb_bench_recall{dataset="$DATASET",db_label="$DB_LABEL",commit="$COMMIT_ID",date="$DATE"} $RECALL
#     # TYPE vdb_bench_latency_p99 gauge
#     vdb_bench_latency_p99{dataset="$DATASET",db_label="$DB_LABEL",commit="$COMMIT_ID",date="$DATE"} $LATENCY_P99
#     EOF

#     curl --data-binary @prom_metrics.txt "http://47.93.34.27:9091/metrics/job/benchmarks/dataset/$DATASET/commit/$COMMIT_ID/date/$DATE" -v
#     python .github/workflows/scripts/upload_bench_results.py
# done
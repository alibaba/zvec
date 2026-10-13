#!/bin/bash
set -u
REPO=/home/xingrui/workspace/zvec/rabitq_refactor/zvec
BIN=$REPO/build_bench/bin
P0RUN=$REPO/../local_validation/v10/p0_20261013-192634
INT_INDEX=/home/xingrui/bench/rabitq_refactor/ivf/cohere_1m_ivf_rabitq_norm1.index
DED_INDEX=/home/xingrui/bench/rabitq_refactor/ivf_dedicated/cohere_1m_ivfrabitq_dedicated.index
QUERIES=/home/xingrui/bench/dataset/cohere/1m/cohere_test_vector_1000.new.txt
RUN=$REPO/../local_validation/v10/p4_$(date +%Y%m%d-%H%M%S)
mkdir -p "$RUN/tmp"
export TMPDIR="$RUN/tmp"
cd "$RUN"
echo "RUN=$RUN" > RUN_DIR.txt

echo "=== V10 P4 BUILD $(date -Is) commit=$(git -C $REPO rev-parse HEAD) ==="
ninja -C "$REPO/build_bench" ivf_rabitq_scan_replay ivf_rabitq_kernel_bench
echo "build rc=$?"

echo "=== V10 P4 KERNEL BENCH $(date -Is) ==="
# Default flags = v9 reference point (dim=768 bits=7 clusters=16 Cosine
# rows=65536 reps=300 warmup=5 seed=123); the crosscheck now compares all 3
# lanes x 32 rows per checked block and exits nonzero on failure.
"$BIN/ivf_rabitq_kernel_bench" > kernel_bench.log 2>&1
echo "kernel rc=$?"

replay() {
  name=$1; mode_int=$2; lists_int=$3; lists_ded=$4
  echo "--- replay $name $(date -Is) ---"
  "$BIN/ivf_rabitq_scan_replay" \
    --index-int="$INT_INDEX" --index-ded="$DED_INDEX" \
    --queries="$QUERIES" \
    --lists-int="$lists_int" --mode-int="$mode_int" \
    --lists-ded="$lists_ded" \
    --rounds=5 --warmup=1 --csv="./$name" > "$name.log" 2>&1
  echo "replay $name rc=$?"
}

echo "=== V10 P4 REPLAY $(date -Is) ==="
# Production pair: integrated baseline (pruning on) vs dedicated default.
replay v10p4_prod baseline "$P0RUN/p0_int_np293_export.lists.csv" \
  "$P0RUN/p0_ded_np249_export.lists.csv"
# Pruning-free pair: both sides scan every probed list fully.
replay v10p4_unpruned unpruned "$P0RUN/p0_int_np293_export.lists.csv" \
  "$P0RUN/p0_ded_np249_unpruned.lists.csv"
echo "=== V10 P4 DONE $(date -Is) ==="

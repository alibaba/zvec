#!/bin/bash
# v12 timing: refine-keyed rerank (rrfN) + lower_bound control (rlbN) +
# prod50, nprobe sweep, 5 rounds, row order rotated per round.
# Reuses the v11 recorded lists (production probe selection per nprobe).
# SMOKE=1 runs np293 only with 20 queries / 2 rounds.
set -u
REPO=/home/xingrui/workspace/zvec/rabitq_refactor/zvec
BIN=$REPO/build_bench/bin
QUERIES=/home/xingrui/bench/dataset/cohere/1m/cohere_test_vector_1000.new.txt
INT_INDEX=/home/xingrui/bench/rabitq_refactor/ivf/cohere_1m_ivf_rabitq_norm1.index
VECS=/home/xingrui/bench/dataset/cohere/1m/cohere_train_vector_1m.new.centaur.vecs
GT=/home/xingrui/bench/dataset/cohere/1m/neighbors.txt
V11=/home/xingrui/bench/integrate_rabitq_in_hnsw/v11_runs
RUN=$(cd "$(dirname "$0")" && pwd)
export TMPDIR="$RUN/tmp"
mkdir -p "$TMPDIR"
cd "$RUN"

if [ "${SMOKE:-0}" = "1" ]; then
  NPROBES="293"
  ROUNDS=2
  QCOUNT=20
  PREFIX=smoke/v12t
else
  NPROBES="150 200 249 272 293 320"
  ROUNDS=5
  QCOUNT=1000
  PREFIX=v12
fi

echo "=== v12 timing $(date -Is) commit=$(git -C $REPO rev-parse HEAD) nprobes=[$NPROBES] rounds=$ROUNDS qcount=$QCOUNT ===" | tee v12_timing_summary.txt

for np in $NPROBES; do
  lists="$V11/iv4b_np${np}.lists.csv"
  if [ ! -f "$lists" ]; then
    echo "missing $lists" | tee -a v12_timing_summary.txt
    continue
  fi
  name="${PREFIX}_np${np}"
  echo "--- replay np=$np $(date -Is) ---" | tee -a v12_timing_summary.txt
  "$BIN/ivf_rabitq_scan_replay" \
    --index-int="$INT_INDEX" --queries="$QUERIES" \
    --lists-int="$lists" --mode-int=baseline \
    --rerank-topn=60,80,100 --vecs="$VECS" --gt="$GT" \
    --queries-count="$QCOUNT" \
    --rounds="$ROUNDS" --warmup=1 --csv="./$name" > "$name.log" 2>&1
  rc=$?
  echo "replay np=$np rc=$rc $(date -Is)" | tee -a v12_timing_summary.txt
  grep -E '^  rerank |^loadavg|verify:' "$name.log" | tee -a v12_timing_summary.txt || true
done

# Normal-sample regression control (v12-3): unmodified samples must pass.
if [ "${SMOKE:-0}" != "1" ]; then
  echo "--- samples normal control $(date -Is) ---" | tee -a v12_timing_summary.txt
  "$BIN/ivf_rabitq_scan_replay" \
    --index-int="$INT_INDEX" --queries="$QUERIES" \
    --lists-int="$V11/int_orig.lists.csv" --mode-int=baseline \
    --rerank-topn=60 --vecs="$VECS" --gt="$GT" \
    --samples-int="$V11/int_orig.samples.csv" \
    --queries-count=1000 --rounds=1 --warmup=0 --csv=./v12_samples_normal \
    > v12_samples_normal.log 2>&1
  rc=$?
  echo "samples normal control rc=$rc (expect 0) $(date -Is)" | tee -a v12_timing_summary.txt
  grep -E 'samples.*checked|verify:' v12_samples_normal.log | tail -3 | tee -a v12_timing_summary.txt || true
fi

echo "=== v12 timing DONE $(date -Is) ===" | tee -a v12_timing_summary.txt

#!/bin/bash
# v13 timing: re-measured with REAL per-round task rotation (the v12
# std::reverse only reordered CSV rows after the measurements had already
# run), plus the requested fill-in: nprobe 160/170/180/190 and N=52/55/60.
# Frontier region np150-200 carries the N sweep (np200 also keeps 80/100
# for continuity with v12); anchors np249-320 use N=60.
# v11/v12 recorded lists are reused where they exist; np160-190 lists are
# exported first by the diagnostic tool (production probe selection).
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

SUM=v13_timing_summary.txt
echo "=== v13 timing $(date -Is) commit=$(git -C $REPO rev-parse HEAD) ===" | tee $SUM

fail() {
  echo "FAILED: $1 -- aborting" | tee -a $SUM
  exit 1
}

for np in 160 170 180 190; do
  if [ -f "iv4b_np${np}.lists.csv" ]; then continue; fi
  echo "--- diag np=$np $(date -Is) ---" | tee -a $SUM
  "$BIN/ivf_rabitq_scan_diagnostic" "int_np${np}.yaml" "./iv4b_np${np}" \
    --export-lists > "iv4b_np${np}.diag.log" 2>&1
  rc=$?
  echo "diag np=$np rc=$rc $(date -Is)" | tee -a $SUM
  grep -E '^int: ' "iv4b_np${np}.diag.log" | tee -a $SUM || true
  [ $rc -ne 0 ] && fail "diag np=$np"
done

run_replay() {
  np=$1
  topns=$2
  if [ -f "iv4b_np${np}.lists.csv" ]; then
    lists="./iv4b_np${np}.lists.csv"
  else
    lists="$V11/iv4b_np${np}.lists.csv"
  fi
  [ -f "$lists" ] || fail "missing lists for np=$np"
  echo "--- replay np=$np N=[$topns] $(date -Is) ---" | tee -a $SUM
  "$BIN/ivf_rabitq_scan_replay" \
    --index-int="$INT_INDEX" --queries="$QUERIES" \
    --lists-int="$lists" --mode-int=baseline \
    --rerank-topn="$topns" --vecs="$VECS" --gt="$GT" \
    --queries-count=1000 --rounds=5 --warmup=1 --csv="./v13_np${np}" \
    > "v13_np${np}.log" 2>&1
  rc=$?
  echo "replay np=$np rc=$rc $(date -Is)" | tee -a $SUM
  grep -E '^  rerank |^loadavg|verify:' "v13_np${np}.log" | tee -a $SUM || true
  [ $rc -ne 0 ] && fail "replay np=$np"
}

for np in 150 160 170 180 190; do
  run_replay $np "52,55,60"
done
run_replay 200 "52,55,60,80,100"
for np in 249 272 293 320; do
  run_replay $np "60"
done

# Normal-sample regression control: unmodified samples must still pass
# with the shared SampleErrsOk predicate (finite + within tol).
echo "--- samples normal control $(date -Is) ---" | tee -a $SUM
"$BIN/ivf_rabitq_scan_replay" \
  --index-int="$INT_INDEX" --queries="$QUERIES" \
  --lists-int="$V11/int_orig.lists.csv" --mode-int=baseline \
  --rerank-topn=60 --vecs="$VECS" --gt="$GT" \
  --samples-int="$V11/int_orig.samples.csv" \
  --queries-count=1000 --rounds=1 --warmup=0 --csv=./v13_samples_normal \
  > v13_samples_normal.log 2>&1
rc=$?
echo "samples normal control rc=$rc (expect 0) $(date -Is)" | tee -a $SUM
grep -E 'samples.*checked|verify:' v13_samples_normal.log | tail -3 | tee -a $SUM || true
[ $rc -ne 0 ] && fail "samples normal control"

echo "=== v13 timing DONE $(date -Is) ===" | tee -a $SUM

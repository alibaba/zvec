#!/bin/bash
# v12 validation: rrfN candidates vs production topk=N, per query, np293.
# Per N: diagnostic topk=N (exports production baseline results), then a
# replay whose rrfN rows verify their candidate sets against those results.
# Exit nonzero if any id or score mismatch.
set -u
REPO=/home/xingrui/workspace/zvec/rabitq_refactor/zvec
BIN=$REPO/build_bench/bin
QUERIES=/home/xingrui/bench/dataset/cohere/1m/cohere_test_vector_1000.new.txt
INT_INDEX=/home/xingrui/bench/rabitq_refactor/ivf/cohere_1m_ivf_rabitq_norm1.index
VECS=/home/xingrui/bench/dataset/cohere/1m/cohere_train_vector_1m.new.centaur.vecs
GT=/home/xingrui/bench/dataset/cohere/1m/neighbors.txt
RUN=$(cd "$(dirname "$0")" && pwd)
export TMPDIR="$RUN/tmp"
mkdir -p "$TMPDIR"
cd "$RUN"

echo "=== v12 validate $(date -Is) commit=$(git -C $REPO rev-parse HEAD) ===" | tee v12_validate_summary.txt
fail=0
for N in 60 80 100; do
  cfg="np293_topk${N}.yaml"
  if [ ! -f "$cfg" ]; then
    echo "missing $cfg (run make_v12_cfgs.py)" | tee -a v12_validate_summary.txt
    fail=1
    continue
  fi
  name="v12_topk${N}"
  echo "--- diag topk=$N $(date -Is) ---" | tee -a v12_validate_summary.txt
  "$BIN/ivf_rabitq_scan_diagnostic" "$cfg" "./$name" --export-lists \
    > "$name.diag.log" 2>&1
  rc=$?
  echo "diag topk=$N rc=$rc $(date -Is)" | tee -a v12_validate_summary.txt
  grep -E '^baseline ' "$name.diag.log" | tee -a v12_validate_summary.txt || true
  if [ $rc -ne 0 ]; then fail=1; continue; fi

  echo "--- replay topk=$N $(date -Is) ---" | tee -a v12_validate_summary.txt
  "$BIN/ivf_rabitq_scan_replay" \
    --index-int="$INT_INDEX" --queries="$QUERIES" \
    --lists-int="$name.lists.csv" --mode-int=baseline \
    --rerank-topn="$N" --vecs="$VECS" --gt="$GT" \
    --results-int="$name.results.csv" \
    --queries-count=1000 --rounds=2 --warmup=1 --csv="./$name" \
    > "$name.log" 2>&1
  rc=$?
  echo "replay topk=$N rc=$rc $(date -Is)" | tee -a v12_validate_summary.txt
  grep -E '^  rerank |^results-int|verify:' "$name.log" | tee -a v12_validate_summary.txt || true
  if [ $rc -ne 0 ]; then fail=1; fi
done
echo "=== v12 validate DONE fail=$fail $(date -Is) ===" | tee -a v12_validate_summary.txt
exit $fail

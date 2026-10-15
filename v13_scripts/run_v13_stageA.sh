#!/bin/bash
# v13 stage A (run AFTER the timing sweep; shares the machine):
# 1. Re-validate the two v12 prod recalls with the REBUILT binaries
#    (stale-binary ODR risk: pre-rebuild bench/recall embedded an older
#    IvfRabitqContext layout; the integrated kIVF path was layout-stable and
#    replay-verified, this re-check closes the loop).
# 2. Native+refine nprobe sweep: find the minimal nprobe whose recall >= 99%
#    (baseline native np249=99.002%; refine at np249 measured 99.518%).
# Prints one line per config; gates every recall at RECALL_MIN.
set -u
REPO=/home/xingrui/workspace/zvec/rabitq_refactor/zvec
BIN=$REPO/build_bench/bin
V12=/home/xingrui/bench/integrate_rabitq_in_hnsw/v12_runs/prod
RUN=$(cd "$(dirname "$0")" && pwd)
export TMPDIR="$RUN/tmp"
mkdir -p "$TMPDIR"
cd "$RUN"
RECALL_MIN=99

fail() { echo "$1" | tee -a v13_stageA_summary.txt; exit 1; }

run_recall() {  # name yaml
  local name=$1 yaml=$2
  echo "--- recall $name $(date -Is) ---" | tee -a v13_stageA_summary.txt
  "$BIN/recall" "$yaml" > "recall_$name.log" 2>&1
  local rc=$?
  local v
  v=$(grep -o 'Recall@50: [0-9.]*' "recall_$name.log" | tail -1 | grep -o '[0-9.]*$')
  echo "recall $name rc=$rc Recall@50: ${v:-MISSING}" | tee -a v13_stageA_summary.txt
  [ $rc -ne 0 ] && fail "recall FAILED for $name"
  [ -z "$v" ] && fail "no Recall@50 line for $name"
  awk -v v="$v" -v m="$RECALL_MIN" 'BEGIN { exit (v < m) ? 1 : 0 }' \
    || fail "recall $name below gate: $v < $RECALL_MIN"
}

echo "=== v13 stageA $(date -Is) commit=$(git -C $REPO rev-parse HEAD) ===" | tee v13_stageA_summary.txt
for c in v12_prod50_np293_t1 v12_rrf60_np200_t1; do
  run_recall "$c" "$V12/$c.yaml"
done
for np in 200 220 232 240 244 249; do
  run_recall "natref_np$np" "v13_natref_np$np.yaml"
done
echo "=== v13 stageA DONE $(date -Is) ===" | tee -a v13_stageA_summary.txt

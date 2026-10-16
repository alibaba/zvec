#!/bin/bash
# v14 task 5 addendum: diagnostic on the INDEPENDENT query set (900 queries,
# cohere 10m test split, §10d) to explain the 2.3-3.3x QPS drop and recall
# degradation seen there. Same no-refiner twins as run_v14_profiling.sh
# (tool refuses refiner configs), with QueryFile swapped to the indep set:
#   int    np293 top50      --segment-timing
#   rrf    np180 top55      --segment-timing
#   native np249 top50      (dedicated: counts only)
#   natref np170 top55      (dedicated: counts only)
# Comparison baseline: the tuning-query diag lines in v14_prof_summary.txt.
# Must run AFTER run_v14_profiling.sh (twins exist) and NOT concurrently
# with any bench (diag timers are load-sensitive, stats ON => relative only).
# Usage: run_v14_indep_diag.sh <natref_np> <natref_topn>
set -u
REPO=/home/xingrui/workspace/zvec/rabitq_refactor/zvec
BIN=$REPO/build_bench/bin
RUN=$(cd "$(dirname "$0")" && pwd)
export TMPDIR="$RUN/tmp"
mkdir -p "$TMPDIR"
cd "$RUN"

NP=${1:?np for native+refine}
TOPN=${2:?N for native+refine}
INDEP_Q=$RUN/indep/queries_indep_q10m_900.txt
[ -f "$INDEP_Q" ] || { echo "missing $INDEP_Q (run filter first)" >&2; exit 1; }

TWINS="v14_prof_int_np293 v14_prof_int_np180_top${TOPN} v14_prof_ded_np249 v14_prof_ded_np${NP}_top${TOPN}"
for t in $TWINS; do
  [ -f "$t.yaml" ] || { echo "missing $t.yaml (run run_v14_profiling.sh first)" >&2; exit 1; }
  sed -e "s|^  QueryFile: .*$|  QueryFile: $INDEP_Q|" \
      -e "s|^  RecallLogDir: .*$|  RecallLogDir: ./${t}_indep.recall|" \
      "$t.yaml" > "${t}_indep.yaml"
  grep -q "$INDEP_Q" "${t}_indep.yaml" || { echo "$t indep swap failed" >&2; exit 1; }
done

SUM=v14_indep_diag_summary.txt
echo "=== v14 indep-query diag np=$NP N=$TOPN $(date -Is) commit=$(git -C $REPO rev-parse HEAD) ===" | tee "$SUM"
echo "queries: indep 900 (10m test split, 100 base members removed); baseline = v14_prof_summary.txt (1000 tuning queries)" | tee -a "$SUM"
fail() { echo "FAILED: $1 -- aborting" | tee -a "$SUM"; exit 1; }

diag() {  # tag cfg extra-flags...
  local tag=$1 cfg=$2; shift 2
  echo "--- diag $tag $(date -Is) loadavg=$(cut -d' ' -f1 /proc/loadavg) ---" | tee -a "$SUM"
  "$BIN/ivf_rabitq_scan_diagnostic" "$cfg" "./$tag" "$@" > "$tag.diag.log" 2>&1
  local rc=$?
  grep -E '^(baseline|unpruned|reuse|reuse_unpruned|dedicated) queries=' "$tag.diag.log" | tee -a "$SUM"
  echo "diag $tag rc=$rc $(date -Is)" | tee -a "$SUM"
  [ $rc -ne 0 ] && fail "diag $tag"
}
diag v14_prof_int_np293_indep v14_prof_int_np293_indep.yaml --segment-timing
diag v14_prof_int_np180_top${TOPN}_indep v14_prof_int_np180_top${TOPN}_indep.yaml --segment-timing
diag v14_prof_ded_np249_indep v14_prof_ded_np249_indep.yaml
diag v14_prof_ded_np${NP}_top${TOPN}_indep v14_prof_ded_np${NP}_top${TOPN}_indep.yaml

echo "=== v14 indep-query diag DONE $(date -Is) ===" | tee -a "$SUM"

#!/bin/bash
# v14 review task 4: recall-loss decomposition on the INDEPENDENT query set.
# Freezes the production shapes (int np293 top50, rrf np180 top55,
# native np249 top50, natref np170 top55) and completes a 2x2 grid
# (nprobe {prod, 1024} x pruning {on, off}) at TopK=100, where coverage of
# GT@50 inside the returned top-100 is (to within quantized-ranking residue)
# the raw probed-bucket coverage: unpruned scans every row of every probed
# list, and a true top-50 member essentially never ranks below rank 100
# among probed candidates. nprobe=1024 = nlist probes every bucket, so
# coverage(unpruned, np1024) is the ranking ceiling and its gap to 100 is
# the pure quantized-ranking loss. Pruning loss is measured both at TopK=100
# (clean, no top-N truncation at 50) and at the production TopK (integrated:
# unpruned mode already in v14_prof_*_indep.results.csv; dedicated: extra
# --disable-pruning runs). Production final Recall@50 and the production
# TopK/TopN coverage come from the already-recorded v14_prof_*_indep twins
# and v14_indep_summary.txt -- nothing here is re-tuned.
# The diagnostic tool overrides QueryParam topk from IndexCommon TopK, so
# only TopK and nprobe are rewritten (nprobe lives inside the escaped JSON
# string as \"nprobe\":<np>).
# Usage: run_v14_recall_split.sh
set -u
REPO=/home/xingrui/workspace/zvec/rabitq_refactor/zvec
BIN=$REPO/build_bench/bin
RUN=$(cd "$(dirname "$0")" && pwd)
export TMPDIR="$RUN/tmp"
mkdir -p "$TMPDIR"
cd "$RUN"

SUM=v14_recall_split_summary.txt
fail() { echo "FAILED: $1 -- aborting" | tee -a "$SUM"; exit 1; }

for t in v14_prof_int_np293_indep v14_prof_int_np180_top55_indep \
         v14_prof_ded_np249_indep v14_prof_ded_np170_top55_indep; do
  [ -f "$t.yaml" ] || fail "missing $t.yaml (run run_v14_indep_diag.sh first)"
done
[ -x "$BIN/ivf_rabitq_scan_diagnostic" ] || fail "missing diagnostic binary"

mk100() {  # src dst np: TopK->100, nprobe->np, fresh RecallLogDir
  local src=$1 dst=$2 np=$3
  sed -e 's|^  TopK: [0-9]*$|  TopK: 100|' \
      -e "s|\\\\\"nprobe\\\\\":[0-9]*|\\\\\"nprobe\\\\\":$np|" \
      -e "s|^  RecallLogDir: .*$|  RecallLogDir: ./$dst.recall|" \
      "$src" > "$dst.yaml"
  grep -q '^  TopK: 100$' "$dst.yaml" || fail "$dst.yaml TopK sed failed"
  grep -qF '\"nprobe\":'"$np" "$dst.yaml" || fail "$dst.yaml nprobe sed failed"
}

echo "=== v14 recall-loss split (indep 900 queries) $(date -Is) commit=$(git -C $REPO rev-parse HEAD) ===" | tee "$SUM"
echo "grid: {int np293, rrf np180, native np249, natref np170} x {np_prod, np1024} x {pruned, unpruned}, TopK=100; + dedicated prod-shape noprune" | tee -a "$SUM"

diag() {  # tag cfg extra-flags...
  local tag=$1 cfg=$2; shift 2
  echo "--- split diag $tag $(date -Is) loadavg=$(cut -d' ' -f1 /proc/loadavg) ---" | tee -a "$SUM"
  "$BIN/ivf_rabitq_scan_diagnostic" "$cfg" "./$tag" "$@" > "$tag.diag.log" 2>&1
  local rc=$?
  grep -E '^(baseline|unpruned|reuse|reuse_unpruned|dedicated) queries=' "$tag.diag.log" | tee -a "$SUM"
  echo "split diag $tag rc=$rc $(date -Is)" | tee -a "$SUM"
  [ $rc -ne 0 ] && fail "diag $tag"
  [ -s "$tag.results.csv" ] || fail "diag $tag produced no results.csv"
}

# integrated twins: one invocation covers baseline/unpruned/reuse/reuse_unpruned
mk100 v14_prof_int_np293_indep.yaml v14_split_int_np293_top100 293
mk100 v14_prof_int_np293_indep.yaml v14_split_int_np1024_top100 1024
mk100 v14_prof_int_np180_top55_indep.yaml v14_split_rrf_np180_top100 180
mk100 v14_prof_int_np180_top55_indep.yaml v14_split_rrf_np1024_top100 1024
diag v14_split_int_np293_top100 v14_split_int_np293_top100.yaml
diag v14_split_int_np1024_top100 v14_split_int_np1024_top100.yaml
diag v14_split_rrf_np180_top100 v14_split_rrf_np180_top100.yaml
diag v14_split_rrf_np1024_top100 v14_split_rrf_np1024_top100.yaml

# dedicated twins: single mode per invocation; unpruned via --disable-pruning
mk100 v14_prof_ded_np249_indep.yaml v14_split_native_np249_top100 249
mk100 v14_prof_ded_np249_indep.yaml v14_split_native_np1024_top100 1024
mk100 v14_prof_ded_np170_top55_indep.yaml v14_split_natref_np170_top100 170
mk100 v14_prof_ded_np170_top55_indep.yaml v14_split_natref_np1024_top100 1024
diag v14_split_native_np249_top100 v14_split_native_np249_top100.yaml
diag v14_split_native_np249_top100_noprune v14_split_native_np249_top100.yaml --disable-pruning
diag v14_split_native_np1024_top100 v14_split_native_np1024_top100.yaml
diag v14_split_native_np1024_top100_noprune v14_split_native_np1024_top100.yaml --disable-pruning
diag v14_split_natref_np170_top100 v14_split_natref_np170_top100.yaml
diag v14_split_natref_np170_top100_noprune v14_split_natref_np170_top100.yaml --disable-pruning
diag v14_split_natref_np1024_top100 v14_split_natref_np1024_top100.yaml
diag v14_split_natref_np1024_top100_noprune v14_split_natref_np1024_top100.yaml --disable-pruning

# dedicated prod-shape unpruned (pruning loss at TopK 50/55): flag only
diag v14_split_native_np249_top50_noprune v14_prof_ded_np249_indep.yaml --disable-pruning
diag v14_split_natref_np170_top55_noprune v14_prof_ded_np170_top55_indep.yaml --disable-pruning

echo "=== v14 recall-loss split DONE $(date -Is) ===" | tee -a "$SUM"

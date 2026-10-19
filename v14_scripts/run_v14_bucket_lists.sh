#!/bin/bash
# v15 review task 1: REAL probed-bucket coverage for the four frozen
# configs on the 900-query independent set.
#
# A. Re-run ivf_rabitq_scan_diagnostic on the four production-shape indep
#    yamls with --export-lists (new output prefixes v14_lists_*; the
#    original v14_prof_* artifacts are left untouched). The exported
#    lists.csv carries the ACTUAL per-query probed bucket set per mode.
# B. Run ivf_rabitq_gt_coverage for the same (index, nprobe) pairs: it
#    rebuilds key->list membership from raw segments and REPLICATES the
#    FP32 centroid selection offline, emitting per-query selected lists,
#    per-GT-id list membership and per-query GT@50 covered counts.
# C. The analyzer cross-checks A's exported sets against B's replicated
#    selection per query (must match; any mismatch is reported, and the
#    DIAG-exported sets remain the source of truth for coverage) and
#    computes the real bucket coverage by intersecting the key->list map
#    of the GT@50 ids with the diag-exported probed sets.
#
# Explicit-nprobe configs set max_scan to the full vector count, so the
# scan-list export is not truncated; the 'limited' column is still
# verified per row.
# Usage: run_v14_bucket_lists.sh
set -u
REPO=/home/xingrui/workspace/zvec/rabitq_refactor/zvec
BIN=$REPO/build_bench/bin
RUN=$(cd "$(dirname "$0")" && pwd)
export TMPDIR="$RUN/tmp"
mkdir -p "$TMPDIR"
cd "$RUN"

INDEP_Q=$RUN/indep/queries_indep_q10m_900.txt
INDEP_GT=$RUN/indep/neighbors_indep_q10m_900.txt
INT_IDX=/home/xingrui/bench/rabitq_refactor/ivf/cohere_1m_ivf_rabitq_norm1.index
DED_IDX=/home/xingrui/bench/rabitq_refactor/ivf_dedicated/cohere_1m_ivfrabitq_dedicated.index
for f in "$INDEP_Q" "$INDEP_GT" "$INT_IDX" "$DED_IDX" \
         v14_prof_int_np293_indep.yaml v14_prof_int_np180_top55_indep.yaml \
         v14_prof_ded_np249_indep.yaml v14_prof_ded_np170_top55_indep.yaml; do
  [ -f "$f" ] || { echo "FAILED: missing $f" | tee -a "$SUM"; exit 1; }
done
for b in ivf_rabitq_scan_diagnostic ivf_rabitq_gt_coverage; do
  [ -x "$BIN/$b" ] || { echo "FAILED: missing $BIN/$b"; exit 1; }
done

SUM=v14_bucket_lists_summary.txt
fail() { echo "FAILED: $1 -- aborting" | tee -a "$SUM"; exit 1; }
echo "=== v14/v15 real bucket-coverage inputs $(date -Is) commit=$(git -C $REPO rev-parse HEAD) ===" | tee "$SUM"
echo "diag --export-lists (prod shape, indep 900) + gt_coverage key->list replication; new prefixes, old logs untouched" | tee -a "$SUM"

diag() {  # tag cfg
  local tag=$1 cfg=$2
  echo "--- lists diag $tag $(date -Is) ---" | tee -a "$SUM"
  "$BIN/ivf_rabitq_scan_diagnostic" "$cfg" "./$tag" --export-lists \
      > "$tag.diag.log" 2>&1
  local rc=$?
  grep -E '^(baseline|unpruned|reuse|reuse_unpruned|dedicated) queries=' "$tag.diag.log" | tee -a "$SUM"
  echo "lists diag $tag rc=$rc $(date -Is)" | tee -a "$SUM"
  [ $rc -ne 0 ] && fail "diag $tag rc=$rc"
  [ -f "$tag.lists.csv" ] || fail "no $tag.lists.csv"
}

gtcov() {  # tag index np
  local tag=$1 idx=$2 np=$3
  echo "--- gtcov $tag $(date -Is) ---" | tee -a "$SUM"
  "$BIN/ivf_rabitq_gt_coverage" --index="$idx" --queries="$INDEP_Q" \
      --gt="$INDEP_GT" --nprobe="$np" --gt_top=50 --csv="./$tag" \
      > "$tag.gt.log" 2>&1
  local rc=$?
  grep -E '^index=|^gt_coverage_mean=' "$tag.gt.log" | tee -a "$SUM"
  echo "gtcov $tag rc=$rc $(date -Is)" | tee -a "$SUM"
  [ $rc -ne 0 ] && fail "gtcov $tag rc=$rc"
  for f in coverage selected gtlists; do
    [ -s "$tag.$f.csv" ] || fail "no/empty $tag.$f.csv"
  done
}

diag v14_lists_int_np293 v14_prof_int_np293_indep.yaml
diag v14_lists_rrf_np180 v14_prof_int_np180_top55_indep.yaml
diag v14_lists_native_np249 v14_prof_ded_np249_indep.yaml
diag v14_lists_natref_np170 v14_prof_ded_np170_top55_indep.yaml

gtcov v14_gtcov_int_np293 "$INT_IDX" 293
gtcov v14_gtcov_rrf_np180 "$INT_IDX" 180
gtcov v14_gtcov_native_np249 "$DED_IDX" 249
gtcov v14_gtcov_natref_np170 "$DED_IDX" 170

echo "=== v14/v15 real bucket-coverage inputs DONE $(date -Is) ===" | tee -a "$SUM"

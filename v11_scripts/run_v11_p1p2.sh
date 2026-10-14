#!/bin/bash
# v11 P1+P2: query-mapping regression + replay equivalence verification.
# Repo copy: outputs default to ../local_validation/v11 (override: V11_RUN_DIR).
# The executed record of the 2026-10-14 run lives in
# ~/bench/integrate_rabitq_in_hnsw/v11_runs (identical logic, RUN=script dir).
set -u
REPO=/home/xingrui/workspace/zvec/rabitq_refactor/zvec
BIN=$REPO/build_bench/bin
V8=/home/xingrui/bench/rabitq_refactor/v8
QUERIES=/home/xingrui/bench/dataset/cohere/1m/cohere_test_vector_1000.new.txt
INT_INDEX=/home/xingrui/bench/rabitq_refactor/ivf/cohere_1m_ivf_rabitq_norm1.index
DED_INDEX=/home/xingrui/bench/rabitq_refactor/ivf_dedicated/cohere_1m_ivfrabitq_dedicated.index
RUN="${V11_RUN_DIR:-$(dirname "$REPO")/local_validation/v11}"
mkdir -p "$RUN/tmp"
export TMPDIR="$RUN/tmp"
cd "$RUN"
STRIDE=100
echo "=== V11 P1+P2 $(date -Is) commit=$(git -C $REPO rev-parse HEAD) stride=$STRIDE ===" | tee p1p2_summary.txt

echo "--- build $(date -Is) ---" | tee -a p1p2_summary.txt
ninja -C "$REPO/build_bench" ivf_rabitq_scan_diagnostic ivf_rabitq_scan_replay > build.log 2>&1
echo "build rc=$?" | tee -a p1p2_summary.txt

echo "--- variants $(date -Is) ---" | tee -a p1p2_summary.txt
python3 "$REPO/v11_scripts/make_v11_variants.py" "$RUN" | tee -a p1p2_summary.txt

run_diag() {
  name=$1; cfg=$2; shift 2
  "$BIN/ivf_rabitq_scan_diagnostic" "$cfg" "./$name" "$@" > "$name.log" 2>&1
  echo "diag $name rc=$? $(date -Is)" | tee -a p1p2_summary.txt
}

echo "--- diagnostics $(date -Is) ---" | tee -a p1p2_summary.txt
SAMPLE_FLAGS="--export-lists --export-scan-samples --sample-query-stride=$STRIDE"
# original order (production pair + unpruned dedicated pair)
run_diag int_orig   "$V8/v8_base_np293.yaml"   $SAMPLE_FLAGS
run_diag ded_orig   "$V8/v8_ded_diag_np249.yaml" $SAMPLE_FLAGS
run_diag ded_orig_unpruned "$V8/v8_ded_diag_np249.yaml" $SAMPLE_FLAGS --disable-pruning
# (a) qid shuffle and (b) row shuffle
run_diag int_qid    "$RUN/int_qid.yaml" $SAMPLE_FLAGS
run_diag ded_qid    "$RUN/ded_qid.yaml" $SAMPLE_FLAGS
run_diag int_row    "$RUN/int_row.yaml" $SAMPLE_FLAGS
run_diag ded_row    "$RUN/ded_row.yaml" $SAMPLE_FLAGS

echo "--- qid-invariance of diagnostic CSVs $(date -Is) ---" | tee -a p1p2_summary.txt
for f in lists.csv samples.csv results.csv; do
  if cmp -s "int_orig.$f" "int_qid.$f" && cmp -s "ded_orig.$f" "ded_qid.$f"; then
    echo "qid-invariance $f: IDENTICAL" | tee -a p1p2_summary.txt
  else
    echo "qid-invariance $f: DIFFER (unexpected)" | tee -a p1p2_summary.txt
  fi
done

replay() {
  name=$1; mode_int=$2; queries=$3; lists_int=$4; lists_ded=$5; samples_int=$6; samples_ded=$7; rounds=$8
  echo "--- replay $name $(date -Is) ---" | tee -a p1p2_summary.txt
  "$BIN/ivf_rabitq_scan_replay" \
    --index-int="$INT_INDEX" --index-ded="$DED_INDEX" \
    --queries="$queries" \
    --lists-int="$lists_int" --mode-int="$mode_int" \
    --lists-ded="$lists_ded" \
    --samples-int="$samples_int" --samples-ded="$samples_ded" \
    --rounds="$rounds" --warmup=1 --csv="./$name" > "$name.log" 2>&1
  echo "replay $name rc=$? $(date -Is)" | tee -a p1p2_summary.txt
}

echo "--- replays $(date -Is) ---" | tee -a p1p2_summary.txt
# R1: production pair, original order (main equivalence verification).
replay v11p2_prod baseline "$QUERIES" \
  int_orig.lists.csv ded_orig.lists.csv int_orig.samples.csv ded_orig.samples.csv 5
# R2: pruning-free pair.
replay v11p2_unpruned unpruned "$QUERIES" \
  int_orig.lists.csv ded_orig_unpruned.lists.csv int_orig.samples.csv ded_orig_unpruned.samples.csv 5
# R3: scrambled qids must not change association (old qid_to_row bug).
replay v11p1_qid_shuffle baseline "$RUN/queries_qid_shuffle.txt" \
  int_qid.lists.csv ded_qid.lists.csv int_qid.samples.csv ded_qid.samples.csv 1
# R4: shuffled rows (non-contiguous, out-of-order qids) map by load order.
replay v11p1_row_shuffle baseline "$RUN/queries_row_shuffle.txt" \
  int_row.lists.csv ded_row.lists.csv int_row.samples.csv ded_row.samples.csv 1
# R5 negative control: wrong query association must FAIL.
replay v11p1_negative baseline "$RUN/queries_row_shuffle.txt" \
  int_orig.lists.csv ded_orig.lists.csv int_orig.samples.csv ded_orig.samples.csv 1

python3 "$REPO/v11_scripts/analyze_v11_replays.py" "$RUN" | tee v11_p1p2_analysis.out

echo "=== V11 P1+P2 DONE $(date -Is) ===" | tee -a p1p2_summary.txt

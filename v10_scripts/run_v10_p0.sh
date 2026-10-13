#!/bin/bash
set -u
REPO=/home/xingrui/workspace/zvec/rabitq_refactor/zvec
RABITQ_BIN=$REPO/build_bench/bin
V8DIR=/home/xingrui/bench/rabitq_refactor/v8
V9RUN=$REPO/v9_out/export_20261013-174804
INT_CFG=$V8DIR/v8_base_np293.yaml
DED_CFG=$V8DIR/v8_ded_diag_np249.yaml
INT256_CFG=$V9RUN/int_np256.yaml
DED256_CFG=$V9RUN/ded_np256.yaml
RUN=$REPO/../local_validation/v10/p0_20261013-192634
export TMPDIR="$RUN/tmp"
cd "$RUN"
echo "=== V10 P0 RERUN $(date -Is) commit=$(git -C $REPO rev-parse HEAD) ==="

run_diag() {
  name=$1; cfg=$2; shift 2
  "$RABITQ_BIN/ivf_rabitq_scan_diagnostic" "$cfg" "./$name" "$@" > "$name.log" 2>&1
  echo "diag $name rc=$? $(date -Is)"
}

# determinism pairs (plain, diagnostics all off)
run_diag p0_int_np293_a "$INT_CFG"
run_diag p0_int_np293_b "$INT_CFG"
run_diag p0_int_np256_a "$INT256_CFG"
run_diag p0_int_np256_b "$INT256_CFG"
run_diag p0_ded_np249_a "$DED_CFG"
run_diag p0_ded_np249_b "$DED_CFG"
run_diag p0_ded_np256_a "$DED256_CFG"
run_diag p0_ded_np256_b "$DED256_CFG"
# diagnostics on (export / segment-timing / disable-pruning)
run_diag p0_int_np293_export "$INT_CFG" --export-lists
run_diag p0_int_np293_timing "$INT_CFG" --segment-timing
run_diag p0_int_np256_export "$INT256_CFG" --export-lists
run_diag p0_int_np256_timing "$INT256_CFG" --segment-timing
run_diag p0_ded_np249_export "$DED_CFG" --export-lists
run_diag p0_ded_np249_unpruned "$DED_CFG" --export-lists --disable-pruning
run_diag p0_ded_np256_export "$DED256_CFG" --export-lists
run_diag p0_ded_np256_unpruned "$DED256_CFG" --export-lists --disable-pruning
echo "=== V10 P0 DONE $(date -Is) ==="

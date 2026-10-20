#!/bin/bash
# v17-E1/E2: prep sub-stage decomposition (stride 1) + sampled-rate
# validation (stride 8). Diagnostic tool only; no QPS measurement here.
# Reuses the frozen v16 configs: integrated np180/Top55/reuse, dedicated
# np170/Top55. All raw logs stay in this directory; TMPDIR is local.
set -u
ROOT=/home/xingrui/bench/integrate_rabitq_in_hnsw/v17_runs
V16=$ROOT/../v16_runs
BIN=/home/xingrui/workspace/zvec/rabitq_refactor/zvec/build_bench/bin
export TMPDIR=$ROOT/tmp
export LD_LIBRARY_PATH=/home/xingrui/workspace/zvec/rabitq_refactor/zvec/build_bench/lib
mkdir -p "$TMPDIR"
cd "$ROOT"

CONFIGS=(
  "int_tuning:$V16/v16_prof_int_tuning.reuse.yaml"
  "int_indep:$V16/v16_prof_int_indep.reuse.yaml"
  "ded_tuning:$V16/v16_prof_ded_tuning.dedicated.yaml"
  "ded_indep:$V16/v16_prof_ded_indep.dedicated.yaml"
)

ROUNDS=${1:-3}
for r in $(seq 1 "$ROUNDS"); do
  for entry in "${CONFIGS[@]}"; do
    name=${entry%%:*}
    cfg=${entry#*:}
    for stride in 1 8; do
      out="v17_diag_s${stride}_r${r}_${name}"
      {
        echo "### $(date +%F_%T) round=$r cfg=$name stride=$stride"
        uptime
      } > "${out}.load.log"
      "$BIN/ivf_rabitq_scan_diagnostic" "$cfg" "$out" \
        --segment-timing --timing-stride="$stride" \
        > "${out}.log" 2>&1
      rc=$?
      uptime >> "${out}.load.log"
      echo "round=$r $name stride=$stride rc=$rc"
      if [ "$rc" -ne 0 ]; then
        echo "FAILED: $out rc=$rc"
        exit "$rc"
      fi
    done
  done
done
echo "v17-E1/E2 diagnostic rounds complete"

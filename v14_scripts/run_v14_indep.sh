#!/bin/bash
# v14 task 4: independent-set validation on the frozen parameters.
# Runs AFTER run_v14_gtgen.sh (queries + GT must exist) and after the
# parameters were frozen in task 2/3: no tuning happens here, results are
# reported whatever they are. For each of the four schemes: recall on the
# 1000 independent queries (t1 config, byte-identical params to the prod
# control, only QueryFile/GroundTruthFile/RecallLogDir swapped), then a
# warmup + 3 rotated bench rounds (QPS confirmation, drift-resistant
# ordering). No recall gate: the independent set is a measurement, not a
# selection criterion -- a miss here would be reported as a finding.
# Appends 'indep recall'/'indep bench' lines to v14_indep_summary.txt
# (consumed by analyze_v14_prod.py).
# Usage: run_v14_indep.sh <natref_np> <natref_topn>
set -u
REPO=/home/xingrui/workspace/zvec/rabitq_refactor/zvec
BIN=$REPO/build_bench/bin
RUN=$(cd "$(dirname "$0")" && pwd)
export TMPDIR="$RUN/tmp"
mkdir -p "$TMPDIR"
cd "$RUN"

NP=${1:?np for native+refine}
TOPN=${2:?N for native+refine}
ROUNDS=3

SUM=v14_indep_summary.txt
[ -f "$SUM" ] || { echo "FAILED: $SUM missing (run run_v14_gtgen.sh first)" >&2; exit 1; }
fail() { echo "FAILED: $1 -- aborting" | tee -a "$SUM"; exit 1; }

CFGS=(
  v12_prod50_np293_t1_indep
  v13_rrf55_np180_t1_indep
  v13_native_np249_t1_indep
  v14_natref_np${NP}_top${TOPN}_t1_indep
)
for c in "${CFGS[@]}"; do
  [ -f "$c.yaml" ] || fail "missing $c.yaml (run make_v14_indep_val_cfgs.py)"
done
n=${#CFGS[@]}

echo "=== v14 independent-set validation np=$NP N=$TOPN $(date -Is) commit=$(git -C $REPO rev-parse HEAD) ===" | tee -a "$SUM"

for c in "${CFGS[@]}"; do
  echo "--- indep recall $c $(date -Is) ---" | tee -a "$SUM"
  "$BIN/recall" "$c.yaml" > "recall_$c.log" 2>&1
  rc=$?
  v=$(grep -o 'Recall@50: [0-9.]*' "recall_$c.log" | tail -1 | grep -o '[0-9.]*$')
  echo "indep recall $c rc=$rc Recall@50: ${v:-MISSING}" | tee -a "$SUM"
  [ $rc -ne 0 ] && fail "indep recall FAILED for $c"
  [ -z "$v" ] && fail "no Recall@50 line for $c"
done

run_bench() {  # tag cfg
  local tag=$1 cfg=$2
  /usr/bin/time -v "$BIN/bench" "$cfg" > "bench_${tag}.log" 2> "bench_${tag}.time.log"
  local rc=$?
  local qps rss
  qps=$(grep -o 'Avg latency: [0-9.]*ms qps: [0-9.]*' "bench_${tag}.log" | grep -o '[0-9.]*$')
  rss=$(grep -o 'Maximum resident set size (kbytes): [0-9]*' "bench_${tag}.time.log" | grep -o '[0-9]*$')
  echo "indep bench $tag rc=$rc qps=${qps:-MISSING} rss_kb=${rss:-MISSING} loadavg=$(cut -d' ' -f1 /proc/loadavg)" | tee -a "$SUM"
  [ $rc -ne 0 ] && fail "indep bench $tag rc=$rc"
  for f in "qps:$qps" "rss:$rss"; do
    [ -z "${f#*:}" ] && fail "indep bench $tag incomplete output: ${f%%:*} MISSING (see bench_${tag}.log)"
  done
}

for c in "${CFGS[@]}"; do
  echo "--- indep warmup $c $(date -Is) ---" | tee -a "$SUM"
  run_bench "warmup_$c" "$c.yaml"
done
for r in $(seq 1 $ROUNDS); do
  off=$(( (r - 1) % n ))
  echo "=== indep ROUND r$r $(date -Is) loadavg=$(cut -d' ' -f1-3 /proc/loadavg) ===" | tee -a "$SUM"
  for i in $(seq 0 $((n - 1))); do
    c=${CFGS[$(( (i + off) % n ))]}
    echo "--- indep r$r $c $(date -Is) ---" | tee -a "$SUM"
    run_bench "${c}_r${r}" "$c.yaml"
  done
done

echo "=== v14 independent-set validation DONE $(date -Is) ===" | tee -a "$SUM"

#!/bin/bash
# v13 4-way same-round production experiment, all on the REBUILT binaries:
#   int    integrated        prod50@np293        (baseline, 99.002%)
#   rrf    integrated+refine rrfN@np             (from the v13 sweep)
#   native native            np249               (baseline, 99.002%)
#   natref native+refine     natref@np           (from the stage-A sweep)
# Each scheme uses its own parameters meeting the >=99% recall gate (gates run
# in run_v13_stageA.sh / run_v13_holdout.sh; this script is timing-only).
# Per run: QPS, avg latency, P50/P95/P99 (BenchResult::print) and peak RSS
# (/usr/bin/time -v). Round r rotates the 8-config order by (r-1)%8; 5 rounds.
# Usage: run_v13_prod.sh <np> <topn> <natref_np>
set -u
REPO=/home/xingrui/workspace/zvec/rabitq_refactor/zvec
BIN=$REPO/build_bench/bin
V12=/home/xingrui/bench/integrate_rabitq_in_hnsw/v12_runs/prod
RUN=$(cd "$(dirname "$0")" && pwd)
export TMPDIR="$RUN/tmp"
mkdir -p "$TMPDIR"
cd "$RUN"

NP=${1:?np for integrated+refine}
TOPN=${2:?N for integrated+refine}
NATREF_NP=${3:?np for native+refine}
ROUNDS=5

INT_T1=$V12/v12_prod50_np293_t1.yaml
INT_T4=$V12/v12_prod50_np293_t4.yaml
RRF_T1=v13_rrf${TOPN}_np${NP}_t1.yaml
RRF_T4=v13_rrf${TOPN}_np${NP}_t4.yaml
NAT_T1=v13_native_np249_t1.yaml
NAT_T4=v13_native_np249_t4.yaml
NRF_T1=v13_natref_np${NATREF_NP}_t1.yaml
NRF_T4=v13_natref_np${NATREF_NP}_t4.yaml
CFGS=($INT_T1 $RRF_T1 $NAT_T1 $NRF_T1 $INT_T4 $RRF_T4 $NAT_T4 $NRF_T4)
n=${#CFGS[@]}

for c in "${CFGS[@]}"; do
  [ -f "$c" ] || { echo "missing $c (run make_v13_prod_cfgs.py / make_v13_native_cfgs.py)" >&2; exit 1; }
done
for b in bench; do
  [ -x "$BIN/$b" ] || { echo "missing $BIN/$b" >&2; exit 1; }
done

SUM=v13_prod_summary.txt
echo "=== v13 prod 4way np=$NP N=$TOPN natref_np=$NATREF_NP $(date -Is) commit=$(git -C $REPO rev-parse HEAD) ===" | tee $SUM
fail() { echo "FAILED: $1 -- aborting" | tee -a $SUM; exit 1; }

RECALL_MIN=99
if [ "${SKIP_RECALL:-0}" != "1" ]; then
  for c in v12_prod50_np293_t1:$INT_T1 v13_rrf${TOPN}_np${NP}_t1:$RRF_T1 v13_native_np249_t1:$NAT_T1 v13_natref_np${NATREF_NP}_t1:$NRF_T1; do
    name=${c%%:*}; yaml=${c##*:}
    echo "--- recall $name $(date -Is) ---" | tee -a $SUM
    "$BIN/recall" "$yaml" > "recall_$name.log" 2>&1
    rc=$?
    v=$(grep -o 'Recall@50: [0-9.]*' "recall_$name.log" | tail -1 | grep -o '[0-9.]*$')
    echo "recall $name rc=$rc Recall@50: ${v:-MISSING}" | tee -a $SUM
    [ $rc -ne 0 ] && fail "recall FAILED for $name"
    [ -z "$v" ] && fail "no Recall@50 line for $name"
    awk -v v="$v" -v m="$RECALL_MIN" 'BEGIN { exit (v < m) ? 1 : 0 }' \
      || fail "recall $name below gate: $v < $RECALL_MIN"
  done
fi

run_bench() {  # tag cfg
  local tag=$1 cfg=$2
  /usr/bin/time -v "$BIN/bench" "$cfg" > "bench_${tag}.log" 2> "bench_${tag}.time.log"
  local rc=$?
  local qps rss p95 p99 avg
  qps=$(grep -o 'Avg latency: [0-9.]*ms qps: [0-9.]*' "bench_${tag}.log" | grep -o '[0-9.]*$')
  avg=$(grep -o 'Avg latency: [0-9.]*ms' "bench_${tag}.log" | grep -o '[0-9.]*' | head -1)
  p95=$(grep '95 Percentile' "bench_${tag}.log" | awk '{print $(NF-1)}')
  p99=$(grep '99 Percentile' "bench_${tag}.log" | awk '{print $(NF-1)}')
  rss=$(grep -o 'Maximum resident set size (kbytes): [0-9]*' "bench_${tag}.time.log" | grep -o '[0-9]*$')
  echo "bench $tag rc=$rc qps=${qps:-MISSING} avg_ms=${avg:-MISSING} p95_ms=${p95:-MISSING} p99_ms=${p99:-MISSING} rss_kb=${rss:-MISSING} loadavg=$(cut -d' ' -f1 /proc/loadavg)" | tee -a $SUM
  [ $rc -ne 0 ] && fail "bench $tag rc=$rc"
  for f in "qps:$qps" "avg:$avg" "p95:$p95" "p99:$p99" "rss:$rss"; do
    [ -z "${f#*:}" ] && fail "bench $tag incomplete output: ${f%%:*} MISSING (see bench_${tag}.log)"
  done
}

for c in "${CFGS[@]}"; do
  tag=$(basename "$c" .yaml)
  echo "--- warmup $tag $(date -Is) ---" | tee -a $SUM
  run_bench "warmup_$tag" "$c"
done

for r in $(seq 1 $ROUNDS); do
  off=$(( (r - 1) % n ))
  echo "=== ROUND r$r $(date -Is) loadavg=$(cut -d' ' -f1-3 /proc/loadavg) ===" | tee -a $SUM
  for i in $(seq 0 $((n - 1))); do
    c=${CFGS[$(( (i + off) % n ))]}
    tag="$(basename "$c" .yaml)_r${r}"
    echo "--- r$r $tag $(date -Is) ---" | tee -a $SUM
    run_bench "$tag" "$c"
  done
done

echo "=== v13 prod 4way DONE $(date -Is) ===" | tee -a $SUM

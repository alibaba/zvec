#!/bin/bash
# v14 4-way same-round production control, all on the REBUILT binaries:
#   int    integrated        prod50@np293        (v12 yaml, unchanged)
#   rrf    integrated+refine rrf55@np180         (v13 yaml, unchanged)
#   native native            np249               (v13 yaml, unchanged)
#   natref native+refine     v14 natref@np topN  (from the v14 sweep)
# Each scheme uses its own parameters meeting the >=99% recall gate on the
# TUNING set (gates re-run here for the same-build guarantee). Per run: QPS,
# avg latency, P50/P95/P99 (BenchResult::print) and peak RSS (/usr/bin/time -v).
# Round r rotates the 8-config order by (r-1)%8; 5 rounds. Failed rounds and
# MISSING fields abort immediately (no silent skips); the analyzer gate then
# verifies 4 schemes x 2 tiers x 5 rounds = 40 unique rc=0 records.
# Usage: run_v14_prod.sh <natref_np> <natref_topn>
set -u
REPO=/home/xingrui/workspace/zvec/rabitq_refactor/zvec
BIN=$REPO/build_bench/bin
V12=/home/xingrui/bench/integrate_rabitq_in_hnsw/v12_runs/prod
V13=/home/xingrui/bench/integrate_rabitq_in_hnsw/v13_runs
RUN=$(cd "$(dirname "$0")" && pwd)
export TMPDIR="$RUN/tmp"
mkdir -p "$TMPDIR"
cd "$RUN"

NATREF_NP=${1:?np for native+refine}
NATREF_TOPN=${2:?N (coarse heap) for native+refine}
ROUNDS=5

INT_T1=$V12/v12_prod50_np293_t1.yaml
INT_T4=$V12/v12_prod50_np293_t4.yaml
RRF_T1=$V13/v13_rrf55_np180_t1.yaml
RRF_T4=$V13/v13_rrf55_np180_t4.yaml
NAT_T1=$V13/v13_native_np249_t1.yaml
NAT_T4=$V13/v13_native_np249_t4.yaml
NRF_T1=v14_natref_np${NATREF_NP}_top${NATREF_TOPN}_t1.yaml
NRF_T4=v14_natref_np${NATREF_NP}_top${NATREF_TOPN}_t4.yaml
CFGS=($INT_T1 $RRF_T1 $NAT_T1 $NRF_T1 $INT_T4 $RRF_T4 $NAT_T4 $NRF_T4)
n=${#CFGS[@]}

for c in "${CFGS[@]}"; do
  [ -f "$c" ] || { echo "missing $c (run make_v14_natref_cfgs.py --nps $NATREF_NP --topns $NATREF_TOPN --threads 1/4)" >&2; exit 1; }
done
for b in bench recall; do
  [ -x "$BIN/$b" ] || { echo "missing $BIN/$b" >&2; exit 1; }
done

SUM=v14_prod_summary.txt
echo "=== v14 prod 4way natref_np=$NATREF_NP N=$NATREF_TOPN $(date -Is) commit=$(git -C $REPO rev-parse HEAD) ===" | tee $SUM
fail() { echo "FAILED: $1 -- aborting" | tee -a $SUM; exit 1; }

RECALL_MIN=99
if [ "${SKIP_RECALL:-0}" != "1" ]; then
  for c in v12_prod50_np293_t1:$INT_T1 v13_rrf55_np180_t1:$RRF_T1 v13_native_np249_t1:$NAT_T1 v14_natref_np${NATREF_NP}_top${NATREF_TOPN}_t1:$NRF_T1; do
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

echo "=== v14 prod 4way DONE $(date -Is) ===" | tee -a $SUM

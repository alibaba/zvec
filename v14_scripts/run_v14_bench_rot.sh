#!/bin/bash
# v14 native+refine ROTATED bench screening on the TUNING set (t1).
# Shared-host drift makes back-to-back configs incomparable (observed
# +-13% between warmup and rounds), so all candidate configs run once
# per round in a rotated order (offset (r-1)%n), with np200_top60 as
# the fixed anchor: selection uses per-round paired QPS ratios vs the
# anchor, not absolute QPS. 1 warmup round per config, then 3 rounds.
# Usage: run_v14_bench_rot.sh   (config list below, yamls next to script)
set -u
REPO=/home/xingrui/workspace/zvec/rabitq_refactor/zvec
BIN=$REPO/build_bench/bin
RUN=$(cd "$(dirname "$0")" && pwd)
export TMPDIR="$RUN/tmp"
mkdir -p "$TMPDIR"
cd "$RUN"

CFGS=(
  v14_natref_np200_top60_t1
  v14_natref_np168_top60_t1
  v14_natref_np170_top60_t1
  v14_natref_np170_top55_t1
  v14_natref_np175_top55_t1
  v14_natref_np180_top55_t1
  v14_natref_np175_top52_t1
  v14_natref_np180_top52_t1
  v14_natref_np180_top60_t1
)
n=${#CFGS[@]}
ROUNDS=3

SUM=v14_natref_bench_rot_summary.txt
if [ ! -f "$SUM" ]; then
  echo "=== v14 natref rotated bench screening $(date -Is) commit=$(git -C $REPO rev-parse HEAD) ===" | tee "$SUM"
fi
fail() { echo "FAILED: $1 -- aborting" | tee -a "$SUM"; exit 1; }
[ -x "$BIN/bench" ] || fail "missing $BIN/bench"
for c in "${CFGS[@]}"; do
  [ -f "$c.yaml" ] || fail "missing $c.yaml"
done

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
  echo "bench $tag rc=$rc qps=${qps:-MISSING} avg_ms=${avg:-MISSING} p95_ms=${p95:-MISSING} p99_ms=${p99:-MISSING} rss_kb=${rss:-MISSING} loadavg=$(cut -d' ' -f1 /proc/loadavg)" | tee -a "$SUM"
  [ $rc -ne 0 ] && fail "bench $tag rc=$rc"
  for f in "qps:$qps" "avg:$avg" "p95:$p95" "p99:$p99" "rss:$rss"; do
    [ -z "${f#*:}" ] && fail "bench $tag incomplete output: ${f%%:*} MISSING (see bench_${tag}.log)"
  done
}

for c in "${CFGS[@]}"; do
  echo "--- warmup $c $(date -Is) ---" | tee -a "$SUM"
  run_bench "warmup_$c" "$c.yaml"
done

for r in $(seq 1 $ROUNDS); do
  off=$(( (r - 1) % n ))
  echo "=== ROUND r$r $(date -Is) loadavg=$(cut -d' ' -f1-3 /proc/loadavg) ===" | tee -a "$SUM"
  for i in $(seq 0 $((n - 1))); do
    c=${CFGS[$(( (i + off) % n ))]}
    echo "--- r$r ${c}_r${r} $(date -Is) ---" | tee -a "$SUM"
    run_bench "${c}_r${r}" "$c.yaml"
  done
done
echo "=== v14 rotated bench screening DONE $(date -Is) ===" | tee -a "$SUM"

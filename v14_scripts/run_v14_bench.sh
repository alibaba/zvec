#!/bin/bash
# v14 native+refine bench screening on the TUNING set (t1, per config:
# 1 warmup + 2 measured rounds). This is a SELECTION measurement, not the
# formal control; the formal 4-way run happens in run_v14_prod.sh with
# 5 rotated rounds and the integrity-gated analyzer.
# Usage: run_v14_bench.sh <cfg-basename>...   (yamls live next to this script)
set -u
REPO=/home/xingrui/workspace/zvec/rabitq_refactor/zvec
BIN=$REPO/build_bench/bin
RUN=$(cd "$(dirname "$0")" && pwd)
export TMPDIR="$RUN/tmp"
mkdir -p "$TMPDIR"
cd "$RUN"

SUM=v14_natref_bench_summary.txt
if [ ! -f "$SUM" ]; then
  echo "=== v14 natref bench screening $(date -Is) commit=$(git -C $REPO rev-parse HEAD) ===" | tee "$SUM"
fi
fail() { echo "FAILED: $1 -- aborting" | tee -a "$SUM"; exit 1; }
[ -x "$BIN/bench" ] || fail "missing $BIN/bench"

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

for name in "$@"; do
  yaml="$name.yaml"
  [ -f "$yaml" ] || fail "missing $yaml"
  echo "--- warmup $name $(date -Is) ---" | tee -a "$SUM"
  run_bench "warmup_$name" "$yaml"
  for r in 1 2; do
    echo "--- r$r $name $(date -Is) ---" | tee -a "$SUM"
    run_bench "${name}_r${r}" "$yaml"
  done
done
echo "=== v14 bench screening DONE $(date -Is) ===" | tee -a "$SUM"

#!/bin/bash
# v17-E5: SIGPROF leaf-PC sampling profile with fine segment timing OFF.
# perf (perf_event_paranoid=4) and gdb attach (ptrace_scope=1) are both
# blocked on this server, so profiling uses an LD_PRELOAD ITIMER_PROF
# sampler. Runs are profiling-only: the ~2kHz signal adds ~0.2% overhead,
# so QPS from these runs must not be reported as speed measurements.
# Tuning query set; frozen configs (int np180/Top55/reuse, ded np170/Top55).
set -u
ROOT=/home/xingrui/bench/integrate_rabitq_in_hnsw/v17_runs
V16=$ROOT/../v16_runs
BIN=/home/xingrui/workspace/zvec/rabitq_refactor/zvec/build_bench/bin
export TMPDIR=$ROOT/tmp
export LD_LIBRARY_PATH=/home/xingrui/workspace/zvec/rabitq_refactor/zvec/build_bench/lib
mkdir -p "$TMPDIR" "$ROOT/cfg"
cd "$ROOT"

# Profiling configs: fine timing OFF, longer bench window (60s).
for side in int ded; do
  sed -e "s|/home/xingrui/bench/integrate_rabitq_in_hnsw/v16_runs/v16_ovh_${side}_off.recall|$ROOT/v17_prof_${side}.recall|" \
      -e 's|BenchSecs: 12|BenchSecs: 60|' \
      "$V16/v16_ovh_${side}_off.yaml" > "cfg/v17_prof_${side}.yaml"
done

REPS=${1:-2}
for rep in $(seq 1 "$REPS"); do
  for side in int ded; do
    tag="v17_prof_sig_${side}_r${rep}"
    {
      echo "### $(date +%F_%T) $tag"
      uptime
    } > "${tag}.load.log"
    LD_PRELOAD="$ROOT/sigprof_sampler.so" \
    SIGPROF_OUT="$ROOT/${tag}.bin" \
    SIGPROF_HZ=2000 \
    SIGPROF_SECONDS=100 \
      "$BIN/bench" "cfg/v17_prof_${side}.yaml" > "${tag}.log" 2>&1
    rc=$?
    uptime >> "${tag}.load.log"
    echo "$tag rc=$rc"
    if [ "$rc" -ne 0 ]; then
      echo "FAILED: $tag"
      exit "$rc"
    fi
    python3 "$ROOT/analyze_sigprof.py" "${tag}.bin" --tail-secs 55 \
      > "${tag}.symbols.txt" 2>&1
  done
done
echo "v17-E5 sampling profiles complete"

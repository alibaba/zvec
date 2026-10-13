#!/bin/bash
set -u
REPO=/home/xingrui/workspace/zvec/rabitq_refactor/zvec
BIN=$REPO/build_bench/bin
V9RUN=$REPO/v9_out/bench_20261013-1708
RUN=$REPO/../local_validation/v10/p1_$(date +%Y%m%d-%H%M%S)
mkdir -p "$RUN/tmp"
export TMPDIR="$RUN/tmp"
cd "$RUN"
echo "RUN=$RUN" > "$RUN/RUN_DIR.txt"

# v10 yamls copied from v9 (diagnostics all off: pruning on, stats/timing off;
# reuse config keeps rabitq_reuse_coarse_dot=true as the mode under test)
for c in base_np293_t1 reuse_np293_t1 ded_np249_t1 base_np293_t4 reuse_np293_t4 ded_np249_t4; do
  sed "s#v9_${c}.recall#v10_${c}.recall#" "$V9RUN/v9_${c}.yaml" > "v10_${c}.yaml"
done

CFGS=(v10_base_np293_t1 v10_reuse_np293_t1 v10_ded_np249_t1 v10_base_np293_t4 v10_reuse_np293_t4 v10_ded_np249_t4)
n=${#CFGS[@]}

env_snap() {
  echo "--- env $1 $(date -Is) loadavg=$(cut -d' ' -f1-3 /proc/loadavg) "
       "cpu_MHz=$(grep -m1 'cpu MHz' /proc/cpuinfo | grep -o '[0-9.]*$') ---"
}

echo "=== V10 P1 BENCH $(date -Is) commit=$(git -C $REPO rev-parse HEAD) ==="
echo "pinning: none (no taskset/numactl); build_bench Release"
env_snap start

echo "=== V10 P1 WARMUP $(date -Is) ==="
for c in "${CFGS[@]}"; do
  "$BIN/bench" "$c.yaml" > "warmup_${c}.log" 2>&1
  echo "warmup $c rc=$?"
done

for r in 1 2 3 4 5; do
  off=$(( (r - 1) % n ))
  echo "=== V10 P1 ROUND r$r $(date -Is) ==="
  env_snap "round r$r"
  for i in $(seq 0 $((n - 1))); do
    c=${CFGS[$(( (i + off) % n ))]}
    echo "--- r$r $c $(date -Is) ---"
    "$BIN/bench" "$c.yaml" > "${c}_r${r}.log" 2>&1
    echo "rc=$?"
  done
done

echo "=== V10 P1 RECALL $(date -Is) ==="
for c in "${CFGS[@]}"; do
  "$BIN/recall" "$c.yaml" > "recall_${c}.log" 2>&1
  echo "$c rc=$? recall=$(grep -o 'Recall@50: [0-9.]*' "recall_${c}.log" | tail -1)"
done

echo "=== V10 P1 DONE $(date -Is) ==="

#!/bin/bash
# v12 production experiment: end-to-end t1/t4 QPS of the gated
# refine->Top-60->exact-rerank scheme (rrf60@np200) vs production
# (prod50@np293), via the bench tool and the yaml RefinerConfig gate.
# Recall is validated FIRST; timing runs only after both recalls land
# at RECALL_MIN (99.002% expected for prod50, ~99.26% for rrf60@np200
# per the replay). A missing Recall@50 line or a value below the
# threshold aborts before timing; missing binaries, failed warmups and
# failed timed rounds all fail the run.
# SKIP_RECALL=1 starts directly at the timing stage.
set -u
REPO=/home/xingrui/workspace/zvec/rabitq_refactor/zvec
BIN=$REPO/build_bench/bin
RUN=$(cd "$(dirname "$0")" && pwd)/prod
RECALL_MIN=99
mkdir -p "$RUN/tmp"
export TMPDIR="$RUN/tmp"
cd "$RUN"

CFGS=(v12_prod50_np293_t1 v12_rrf60_np200_t1 v12_prod50_np293_t4 v12_rrf60_np200_t4)
n=${#CFGS[@]}
ROUNDS=5

env_snap() {
  echo "--- env $1 $(date -Is) loadavg=$(cut -d' ' -f1-3 /proc/loadavg) ---"
}

fail() {
  echo "FAILED: $1 -- aborting" | tee -a v12_prod_summary.txt
  exit 1
}

for b in recall bench; do
  [ -x "$BIN/$b" ] || { echo "missing $BIN/$b (build failed?)" >&2; exit 1; }
done

if [ "${SKIP_RECALL:-0}" != "1" ]; then
  echo "=== v12 prod VALIDATE $(date -Is) commit=$(git -C $REPO rev-parse HEAD) ===" | tee v12_prod_summary.txt
  for c in v12_prod50_np293_t1 v12_rrf60_np200_t1; do
    echo "--- recall $c $(date -Is) ---" | tee -a v12_prod_summary.txt
    "$BIN/recall" "$c.yaml" > "recall_$c.log" 2>&1
    rc=$?
    v=$(grep -o 'Recall@50: [0-9.]*' "recall_$c.log" | tail -1 | grep -o '[0-9.]*$')
    echo "recall $c rc=$rc Recall@50=${v:-MISSING} (min $RECALL_MIN)" | tee -a v12_prod_summary.txt
    [ $rc -ne 0 ] && fail "recall run for $c"
    [ -z "$v" ] && fail "no Recall@50 line in recall_$c.log"
    awk -v v="$v" -v m="$RECALL_MIN" 'BEGIN { exit (v < m) ? 1 : 0 }' ||
      fail "recall $c below threshold: $v < $RECALL_MIN"
  done
else
  echo "=== v12 prod (recall skipped) $(date -Is) commit=$(git -C $REPO rev-parse HEAD) ===" | tee v12_prod_summary.txt
fi

echo "=== v12 prod WARMUP $(date -Is) ===" | tee -a v12_prod_summary.txt
for c in "${CFGS[@]}"; do
  "$BIN/bench" "$c.yaml" > "warmup_$c.log" 2>&1
  rc=$?
  echo "warmup $c rc=$rc" | tee -a v12_prod_summary.txt
  [ $rc -ne 0 ] && fail "warmup for $c"
done

for r in $(seq 1 $ROUNDS); do
  off=$(( (r - 1) % n ))
  echo "=== v12 prod ROUND r$r $(date -Is) ===" | tee -a v12_prod_summary.txt
  env_snap "round r$r" | tee -a v12_prod_summary.txt
  for i in $(seq 0 $((n - 1))); do
    c=${CFGS[$(( (i + off) % n ))]}
    echo "--- r$r $c $(date -Is) ---" | tee -a v12_prod_summary.txt
    "$BIN/bench" "$c.yaml" > "${c}_r${r}.log" 2>&1
    rc=$?
    grep -m1 'Avg latency' "${c}_r${r}.log" | tee -a v12_prod_summary.txt
    echo "bench r$r $c rc=$rc" | tee -a v12_prod_summary.txt
    [ $rc -ne 0 ] && fail "bench r$r $c"
  done
done

echo "=== v12 prod DONE $(date -Is) ===" | tee -a v12_prod_summary.txt

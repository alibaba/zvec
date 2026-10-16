#!/bin/bash
# v13 stage C: grouped check of the tuning data (q500-999) + scan-volume stats.
# Holdout = the 500 test queries q500-999. NOT an independent confirmation
# set: the v12/v13 (np, N) choices were made on the full 1000-query set that
# includes these queries, so this is a grouped check of the tuning data (a
# truly independent query set is v14 work); the knobs are two global
# integers, so per-query overfitting is not possible -- this half mainly
# demonstrates that recall and QPS are stable across query subsets rather
# than artifacts of the aggregate.
# 1. Recall on the holdout half for all four schemes. The PROPOSED configs
#    (rrf, natref) hard-gate at >=99% (guards against params fitted to the
#    full 1000); the baseline anchors (int, native) are report-only -- they
#    were selected as "first np >= 99% on the FULL set" with razor-thin
#    margin, and a 500-query half can sit slightly below 99 purely by split
#    variance (int@np293: full 99.002 -> holdout 98.952 measured from the
#    per-query logs). The full-set value governs anchor qualification;
#    aborting on the half-split would only block data collection.
# 2. One t1 bench round per scheme on the holdout half (QPS grouped check).
# 3. Diagnostic scan-volume runs: int np293 topk50, int np{np} topk{N},
#    ded np249 topk50, ded np{natref_np} topk60.
# Usage: run_v13_holdout.sh <np> <topn> <natref_np>
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
RECALL_MIN=99

SUM=v13_holdout_summary.txt
echo "=== v13 holdout np=$NP N=$TOPN natref_np=$NATREF_NP $(date -Is) commit=$(git -C $REPO rev-parse HEAD) ===" | tee $SUM
fail() { echo "FAILED: $1 -- aborting" | tee -a $SUM; exit 1; }

HOLDOFF=(
  v13_int50_np293_t1_holdout:v13_int50_np293_t1_holdout.yaml
  v13_rrf${TOPN}_np${NP}_t1_holdout:v13_rrf${TOPN}_np${NP}_t1_holdout.yaml
  v13_native_np249_t1_holdout:v13_native_np249_t1_holdout.yaml
  v13_natref_np${NATREF_NP}_holdout:v13_natref_np${NATREF_NP}_holdout.yaml
)
for c in "${HOLDOFF[@]}"; do
  yaml=${c##*:}
  [ -f "$yaml" ] || fail "missing $yaml"
done
# int (np, N) diag yaml for scan volume
INT_DIAG=v13_intdiag_np${NP}_topk${TOPN}.yaml
[ -f "$INT_DIAG" ] || fail "missing $INT_DIAG (int diag)"
[ -f v13_deddiag_np249_topk50.yaml ] || fail "missing v13_deddiag_np249_topk50.yaml"
[ -f v13_deddiag_np${NATREF_NP}_topk60.yaml ] || fail "missing v13_deddiag_np${NATREF_NP}_topk60.yaml"

# 1=proposed (hard gate), 0=baseline anchor (report-only, see header)
declare -A HOLDGATE=(
  [v13_int50_np293_t1_holdout]=0
  [v13_rrf${TOPN}_np${NP}_t1_holdout]=1
  [v13_native_np249_t1_holdout]=0
  [v13_natref_np${NATREF_NP}_holdout]=1
)
declare -A HOLDVAL=()
for c in "${HOLDOFF[@]}"; do
  name=${c%%:*}; yaml=${c##*:}
  echo "--- holdout recall $name $(date -Is) ---" | tee -a $SUM
  "$BIN/recall" "$yaml" > "holdout_recall_$name.log" 2>&1
  rc=$?
  v=$(grep -o 'Recall@50: [0-9.]*' "holdout_recall_$name.log" | tail -1 | grep -o '[0-9.]*$')
  HOLDVAL[$name]=${v:-MISSING}
  role=$([ "${HOLDGATE[$name]}" = "1" ] && echo proposed || echo anchor)
  echo "holdout recall $name rc=$rc Recall@50: ${v:-MISSING} ($role)" | tee -a $SUM
  [ $rc -ne 0 ] && fail "holdout recall FAILED for $name"
  [ -z "$v" ] && fail "no Recall@50 line for $name"
done
for c in "${HOLDOFF[@]}"; do
  name=${c%%:*}; v=${HOLDVAL[$name]}
  if awk -v v="$v" -v m="$RECALL_MIN" 'BEGIN { exit (v < m) ? 1 : 0 }'; then
    continue
  fi
  if [ "${HOLDGATE[$name]}" = "1" ]; then
    fail "holdout recall $name below gate: $v < $RECALL_MIN"
  fi
  echo "WARN: anchor $name holdout recall $v < $RECALL_MIN (split variance; full-set value governs qualification)" | tee -a $SUM
done

for c in "${HOLDOFF[@]}"; do
  name=${c%%:*}; yaml=${c##*:}
  echo "--- holdout bench $name $(date -Is) ---" | tee -a $SUM
  /usr/bin/time -v "$BIN/bench" "$yaml" > "holdout_bench_$name.log" 2> "holdout_bench_$name.time.log"
  rc=$?
  qps=$(grep -o 'Avg latency: [0-9.]*ms qps: [0-9.]*' "holdout_bench_$name.log" | grep -o '[0-9.]*$')
  rss=$(grep -o 'Maximum resident set size (kbytes): [0-9]*' "holdout_bench_$name.time.log" | grep -o '[0-9]*$')
  echo "holdout bench $name rc=$rc qps=${qps:-MISSING} rss_kb=${rss:-MISSING} loadavg=$(cut -d' ' -f1 /proc/loadavg)" | tee -a $SUM
  [ $rc -ne 0 ] && fail "holdout bench $name"
  [ -z "$qps" ] && fail "holdout bench $name incomplete output: qps MISSING (see holdout_bench_$name.log)"
  [ -z "$rss" ] && fail "holdout bench $name incomplete output: rss_kb MISSING (see holdout_bench_$name.time.log)"
done

diag_run() {  # tag cfg
  local tag=$1 cfg=$2
  echo "--- diag $tag $(date -Is) ---" | tee -a $SUM
  "$BIN/ivf_rabitq_scan_diagnostic" "$cfg" "./$tag" > "$tag.diag.log" 2>&1
  rc=$?
  grep -E '^(baseline|dedicated) queries=' "$tag.diag.log" | tee -a $SUM || true
  echo "diag $tag rc=$rc $(date -Is)" | tee -a $SUM
  [ $rc -ne 0 ] && fail "diag $tag"
}

diag_run v13_diag_int_np293_topk50 /home/xingrui/bench/rabitq_refactor/v8/v8_base_np293.yaml
diag_run v13_diag_int_np${NP}_topk${TOPN} "$INT_DIAG"
diag_run v13_diag_ded_np249_topk50 v13_deddiag_np249_topk50.yaml
diag_run v13_diag_ded_np${NATREF_NP}_topk60 v13_deddiag_np${NATREF_NP}_topk60.yaml

echo "=== v13 holdout DONE $(date -Is) ===" | tee -a $SUM

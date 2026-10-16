#!/bin/bash
# v14 task 5: bottleneck localization, strictly separate from the formal
# bench (profiling runs, own summary file, never mixed into
# v14_prod_summary.txt).
# A. Scan-volume + segment breakdown via ivf_rabitq_scan_diagnostic on
#    no-refiner twins of the four production configs (the tool refuses
#    refiner configs, so the coarse stage is measured alone):
#      int    np293 top50   (v12 prod yaml as-is)
#      rrf    np180 top55   (v13 rrf yaml minus RefinerConfig, TopK->55)
#      native np249 top50   (v13 native yaml as-is)
#      natref np{W} top{N}  (v14 natref yaml minus RefinerConfig, TopK->N)
#    --segment-timing on the two integrated configs (per-segment ns:
#    prep / cand(block gate+coarse fetch) / coarse(RaBitQ estimate) /
#    post_scan(row refine+heap) / finalize(heap sort+key map) / query).
#    Dedicated kIVFRabitq has no segment timing: counts only. Timers run
#    with stats collection ON, so they are for RELATIVE attribution only,
#    never comparable to bench QPS.
#    Heap ops: each surviving candidate does block_scanner_->refine() then
#    exactly one heap emplace (ivf_entity.cc consume lambda), so
#    refined_count == heap push count; heap capacity = coarse_topk.
# B. Differential bench (refine cost): emitted no-stats mode yamls are
#    benched alternated with the production refine yamls, 3 rotated rounds
#    per pair, so the FP32-refine delta is a same-round paired difference.
# Usage: run_v14_profiling.sh <natref_np> <natref_topn>
set -u
REPO=/home/xingrui/workspace/zvec/rabitq_refactor/zvec
BIN=$REPO/build_bench/bin
V12=/home/xingrui/bench/integrate_rabitq_in_hnsw/v12_runs/prod
V13=/home/xingrui/bench/integrate_rabitq_in_hnsw/v13_runs
RUN=$(cd "$(dirname "$0")" && pwd)
export TMPDIR="$RUN/tmp"
mkdir -p "$TMPDIR"
cd "$RUN"

NP=${1:?np for native+refine}
TOPN=${2:?N for native+refine}
NRF=$RUN/v14_natref_np${NP}_top${TOPN}_t1.yaml
[ -f "$NRF" ] || { echo "missing $NRF" >&2; exit 1; }

SUM=v14_prof_summary.txt
echo "=== v14 profiling np=$NP N=$TOPN $(date -Is) commit=$(git -C $REPO rev-parse HEAD) ===" | tee "$SUM"
fail() { echo "FAILED: $1 -- aborting" | tee -a "$SUM"; exit 1; }

# no-refiner twins (tool refuses refiner/group-by/filter configs)
sed -e '/^  RefinerConfig:$/,$d' -e 's|^  TopK: 50$|  TopK: '"$TOPN"'|' \
  "$V13/v13_rrf55_np180_t1.yaml" |
  sed 's|^  RecallLogDir: .*$|  RecallLogDir: ./v14_prof_int_np180_top'"$TOPN"'.recall|' \
  > v14_prof_int_np180_top${TOPN}.yaml
sed -e '/^  RefinerConfig:$/,$d' -e 's|^  TopK: 50$|  TopK: '"$TOPN"'|' \
  "$NRF" |
  sed 's|^  RecallLogDir: .*$|  RecallLogDir: ./v14_prof_ded_np'"$NP"'_top'"$TOPN"'.recall|' \
  > v14_prof_ded_np${NP}_top${TOPN}.yaml
sed 's|^  RecallLogDir: .*$|  RecallLogDir: ./v14_prof_int_np293.recall|' \
  "$V12/v12_prod50_np293_t1.yaml" > v14_prof_int_np293.yaml
sed 's|^  RecallLogDir: .*$|  RecallLogDir: ./v14_prof_ded_np249.recall|' \
  "$V13/v13_native_np249_t1.yaml" > v14_prof_ded_np249.yaml
for f in v14_prof_int_np180_top${TOPN}.yaml v14_prof_ded_np${NP}_top${TOPN}.yaml v14_prof_int_np293.yaml v14_prof_ded_np249.yaml; do
  [ -f "$f" ] && grep -q RefinerConfig "$f" && fail "$f still has RefinerConfig"
done

diag() {  # tag cfg extra-flags...
  local tag=$1 cfg=$2; shift 2
  echo "--- diag $tag $(date -Is) ---" | tee -a "$SUM"
  "$BIN/ivf_rabitq_scan_diagnostic" "$cfg" "./$tag" "$@" > "$tag.diag.log" 2>&1
  local rc=$?
  grep -E '^(baseline|unpruned|reuse|reuse_unpruned|dedicated) queries=' "$tag.diag.log" | tee -a "$SUM"
  echo "diag $tag rc=$rc $(date -Is)" | tee -a "$SUM"
  [ $rc -ne 0 ] && fail "diag $tag"
}
diag v14_prof_int_np293 v14_prof_int_np293.yaml --segment-timing
diag v14_prof_int_np180_top${TOPN} v14_prof_int_np180_top${TOPN}.yaml --segment-timing
diag v14_prof_ded_np249 v14_prof_ded_np249.yaml
diag v14_prof_ded_np${NP}_top${TOPN} v14_prof_ded_np${NP}_top${TOPN}.yaml

# differential bench: no-stats mode yamls emitted by the diag tool
pbench() {  # tag cfg
  local tag=$1 cfg=$2
  /usr/bin/time -v "$BIN/bench" "$cfg" > "bench_${tag}.log" 2> "bench_${tag}.time.log"
  local rc=$?
  local qps avg p95 p99
  qps=$(grep -o 'Avg latency: [0-9.]*ms qps: [0-9.]*' "bench_${tag}.log" | grep -o '[0-9.]*$')
  avg=$(grep -o 'Avg latency: [0-9.]*ms' "bench_${tag}.log" | grep -o '[0-9.]*' | head -1)
  p95=$(grep '95 Percentile' "bench_${tag}.log" | awk '{print $(NF-1)}')
  p99=$(grep '99 Percentile' "bench_${tag}.log" | awk '{print $(NF-1)}')
  echo "prof bench $tag rc=$rc qps=${qps:-MISSING} avg_ms=${avg:-MISSING} p95_ms=${p95:-MISSING} p99_ms=${p99:-MISSING} loadavg=$(cut -d' ' -f1 /proc/loadavg)" | tee -a "$SUM"
  [ $rc -ne 0 ] && fail "prof bench $tag"
  [ -z "$qps" ] && fail "prof bench $tag incomplete output"
}

# pairs: (plain coarse twin, production refine config), rotated 3 rounds
PAIRS=(
  "v14_prof_int_np180_top${TOPN}.baseline.yaml:$V13/v13_rrf55_np180_t1.yaml:rrf"
  "v14_prof_ded_np${NP}_top${TOPN}.dedicated.yaml:$NRF:natref"
  "v14_prof_int_np293.baseline.yaml:$V12/v12_prod50_np293_t1.yaml:int_same"
  "v14_prof_ded_np249.dedicated.yaml:$V13/v13_native_np249_t1.yaml:native_same"
)
for p in "${PAIRS[@]}"; do
  plain=${p%%:*}; rest=${p#*:}; ref=${rest%%:*}; label=${rest##*:}
  [ -f "$plain" ] || fail "missing $plain (diag must emit it)"
  [ -f "$ref" ] || fail "missing $ref"
done
for p in "${PAIRS[@]}"; do
  plain=${p%%:*}; rest=${p#*:}; ref=${rest%%:*}; label=${rest##*:}
  pbench "warmup_plain_$label" "$plain"
  pbench "warmup_refine_$label" "$ref"
done
for r in 1 2 3; do
  echo "=== prof ROUND r$r $(date -Is) loadavg=$(cut -d' ' -f1-3 /proc/loadavg) ===" | tee -a "$SUM"
  i=0
  for p in "${PAIRS[@]}"; do
    plain=${p%%:*}; rest=${p#*:}; ref=${rest%%:*}; label=${rest##*:}
    if [ $(( (i + r - 1) % 2 )) -eq 0 ]; then
      pbench "plain_${label}_r${r}" "$plain"; pbench "refine_${label}_r${r}" "$ref"
    else
      pbench "refine_${label}_r${r}" "$ref"; pbench "plain_${label}_r${r}" "$plain"
    fi
    i=$((i + 1))
  done
done

echo "=== v14 profiling DONE $(date -Is) ===" | tee -a "$SUM"

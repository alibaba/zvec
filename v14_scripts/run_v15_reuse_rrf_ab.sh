#!/bin/bash
# v15 task 4: production OFF/ON A/B for rabitq_reuse_coarse_dot on the rrf
# scheme (integrated IVF RaBitQ np180 + kFlat refiner, SF=1.1 -> coarse heap
# 55, final topk 50). Feasible since the reuse+refiner guard split (see the
# v15 report section 12c); attempt 1 in v14_runs was engine-rejected under
# the old bundled guard and its logs are kept there untouched.
# Same index, same frozen parameters, same query set per pair: the only
# difference between each off/on config pair is the reuse flag inside the
# QueryParam JSON. Formal timing keeps rabitq_collect_stats and
# rabitq_segment_timing OFF (both false in the production sources).
# Protocol: per (query set x tier) group of 3 configs {rrf off, rrf on,
# natref np170 top55 anchor}: one warmup each, then 5 rounds with rotated
# order (each config rotates through the start slot). The native+refine
# np170 Top55 anchor runs in the SAME rounds so the rrf-vs-anchor QPS gap
# is a same-round measurement. /usr/bin/time -v for RSS.
# Recall runs on the t1 configs of BOTH query sets (tuning 1000 + indep 900)
# for the two rrf arms AND the two anchor arms: per-query t*.knn logs to
# distinct per-config RecallLogDirs so the analyzer can compare returned ids
# and per-query GT hits, not just the aggregate Recall@50. Nothing is tuned
# here; the indep set is a hard set below the 99% gate, so it only carries
# the frozen-parameter comparison.
# Usage: run_v15_reuse_rrf_ab.sh
set -u
REPO=/home/xingrui/workspace/zvec/rabitq_refactor/zvec
BIN=$REPO/build_bench/bin
V14=/home/xingrui/bench/integrate_rabitq_in_hnsw/v14_runs
RUN=${V15_RUN_DIR:-/home/xingrui/bench/integrate_rabitq_in_hnsw/v15_runs}
export TMPDIR="$RUN/tmp"
mkdir -p "$TMPDIR"
cd "$RUN"

ROUNDS=5
SUM=v15_reuse_rrf_ab_summary.txt
fail() { echo "FAILED: $1 -- aborting" | tee -a "$SUM"; exit 1; }
# Repeated executions use an independent run directory: refuse to touch an
# existing summary or bench/recall logs so nothing from a previous run is
# overwritten or mixed.
if [ -e "$SUM" ]; then
  echo "FAILED: $SUM already exists -- rerun in a fresh run directory (old logs are kept)"
  exit 1
fi

RRF_T1_TUNING=$V14/v14_ab_rrf_np180_t1_tuning_off.yaml
RRF_T4_TUNING=$V14/v14_ab_rrf_np180_t4_tuning_off.yaml
RRF_T1_INDEP=$V14/v14_ab_rrf_np180_t1_indep_off.yaml
RRF_T4_INDEP=$V14/v14_ab_rrf_np180_t4_indep_off.yaml
ANCHOR_T1_TUNING=$V14/v14_natref_np170_top55_t1.yaml
ANCHOR_T4_TUNING=$V14/v14_natref_np170_top55_t4.yaml
ANCHOR_T1_INDEP=$V14/v14_natref_np170_top55_t1_indep.yaml
for f in "$RRF_T1_TUNING" "$RRF_T4_TUNING" "$RRF_T1_INDEP" "$RRF_T4_INDEP" \
         "$ANCHOR_T1_TUNING" "$ANCHOR_T4_TUNING" "$ANCHOR_T1_INDEP"; do
  [ -f "$f" ] || fail "missing source yaml $f"
done
for b in bench recall; do
  [ -x "$BIN/$b" ] || fail "missing $BIN/$b"
done
for f in "$RRF_T1_TUNING" "$RRF_T4_TUNING" "$RRF_T1_INDEP" "$RRF_T4_INDEP" \
         "$ANCHOR_T1_TUNING" "$ANCHOR_T4_TUNING" "$ANCHOR_T1_INDEP"; do
  grep -qF 'rabitq_collect_stats\":false' "$f" \
    || fail "$f has rabitq_collect_stats not false"
  grep -qF 'rabitq_segment_timing\":false' "$f" \
    || fail "$f has rabitq_segment_timing not false"
done

mkcfg_rrf() {  # src dst reuse(0|1)
  local src=$1 dst=$2 reuse=$3 flag=false
  [ "$reuse" = 1 ] && flag=true
  sed -e "s|\\\\\"rabitq_reuse_coarse_dot\\\\\":false|\\\\\"rabitq_reuse_coarse_dot\\\\\":$flag|" \
      -e "s|^  RecallLogDir: .*$|  RecallLogDir: ./$dst.recall|" \
      "$src" > "$dst.yaml"
  grep -qF '\"rabitq_reuse_coarse_dot\":'"$flag" "$dst.yaml" \
    || fail "$dst.yaml reuse flag sed failed"
  grep -q '^  RecallLogDir: \./'"$dst"'.recall$' "$dst.yaml" \
    || fail "$dst.yaml recall dir sed failed"
}

mkcfg_anchor() {  # src dst threads
  local src=$1 dst=$2 threads=$3
  sed -e "s|^  BenchThreadCount: [0-9]*$|  BenchThreadCount: $threads|" \
      -e "s|^  RecallLogDir: .*$|  RecallLogDir: ./$dst.recall|" \
      "$src" > "$dst.yaml"
  grep -q "^  BenchThreadCount: $threads\$" "$dst.yaml" \
    || fail "$dst.yaml thread sed failed"
  grep -q '^  RecallLogDir: \./'"$dst"'.recall$' "$dst.yaml" \
    || fail "$dst.yaml recall dir sed failed"
}

echo "=== v15 reuse OFF/ON rrf A/B + natref anchor $(date -Is) commit=$(git -C $REPO rev-parse HEAD) ===" | tee "$SUM"
echo "pairs: {rrf np180 topk50 SF1.1 coarse55} x {t1,t4} x {tuning 1000, indep 900} x {off,on}; same-round anchor {natref np170 top55} per group; stats/segment-timing off; warmup + 5 rotated rounds" | tee -a "$SUM"

mkcfg_rrf "$RRF_T1_TUNING" v15_ab_rrf_np180_t1_tuning_off 0
mkcfg_rrf "$RRF_T1_TUNING" v15_ab_rrf_np180_t1_tuning_on 1
mkcfg_rrf "$RRF_T4_TUNING" v15_ab_rrf_np180_t4_tuning_off 0
mkcfg_rrf "$RRF_T4_TUNING" v15_ab_rrf_np180_t4_tuning_on 1
mkcfg_rrf "$RRF_T1_INDEP" v15_ab_rrf_np180_t1_indep_off 0
mkcfg_rrf "$RRF_T1_INDEP" v15_ab_rrf_np180_t1_indep_on 1
mkcfg_rrf "$RRF_T4_INDEP" v15_ab_rrf_np180_t4_indep_off 0
mkcfg_rrf "$RRF_T4_INDEP" v15_ab_rrf_np180_t4_indep_on 1
mkcfg_anchor "$ANCHOR_T1_TUNING" v15_anchor_natref_np170_top55_t1_tuning 1
mkcfg_anchor "$ANCHOR_T4_TUNING" v15_anchor_natref_np170_top55_t4_tuning 4
mkcfg_anchor "$ANCHOR_T1_INDEP" v15_anchor_natref_np170_top55_t1_indep 1
mkcfg_anchor "$ANCHOR_T1_INDEP" v15_anchor_natref_np170_top55_t4_indep 4

run_recall() {  # tag cfg
  local tag=$1 cfg=$2
  # Repeated executions must use a fresh run directory so old per-query
  # logs are never overwritten or mixed (the recall tool truncates t*.knn
  # in place but leaves foreign shard files untouched).
  if [ -e "$tag.recall" ]; then
    fail "$tag.recall already exists from a previous execution; rerun in a fresh run directory (old logs are kept)"
  fi
  echo "--- ab recall $tag $(date -Is) ---" | tee -a "$SUM"
  "$BIN/recall" "$cfg" > "recall_$tag.log" 2>&1
  local rc=$?
  local v
  v=$(grep -o 'Recall@50: [0-9.]*' "recall_$tag.log" | tail -1 | grep -o '[0-9.]*$')
  echo "ab recall $tag rc=$rc Recall@50: ${v:-MISSING}" | tee -a "$SUM"
  [ $rc -ne 0 ] && fail "ab recall $tag rc=$rc"
  [ -z "$v" ] && fail "no Recall@50 line for $tag"
  # rc=0 with Recall@50 exactly 0 means every knn_search failed (e.g. the
  # engine's Unsupported path); the recall tool does not propagate that.
  [ "$(awk -v v="$v" 'BEGIN{print (v+0 > 0) ? 1 : 0}')" = 1 ] \
    || fail "ab recall $tag Recall@50=$v (all searches failed?)"
  [ -d "$tag.recall" ] || fail "no RecallLogDir $tag.recall for per-query logs"
}

for tag in v15_ab_rrf_np180_t1_tuning_off v15_ab_rrf_np180_t1_tuning_on \
           v15_ab_rrf_np180_t1_indep_off v15_ab_rrf_np180_t1_indep_on \
           v15_anchor_natref_np170_top55_t1_tuning \
           v15_anchor_natref_np170_top55_t1_indep; do
  run_recall "$tag" "$tag.yaml"
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
  rss=$(grep -o 'Maximum resident set size (kbytes): [0-9]*' "bench_${tag}.time.log" | grep -o '[0-9.]*$')
  echo "bench $tag rc=$rc qps=${qps:-MISSING} avg_ms=${avg:-MISSING} p95_ms=${p95:-MISSING} p99_ms=${p99:-MISSING} rss_kb=${rss:-MISSING} loadavg=$(cut -d' ' -f1 /proc/loadavg)" | tee -a "$SUM"
  [ $rc -ne 0 ] && fail "bench $tag rc=$rc"
  for f in "qps:$qps" "avg:$avg" "p95:$p95" "p99:$p99" "rss:$rss"; do
    [ -z "${f#*:}" ] && fail "bench $tag incomplete output: ${f%%:*} MISSING (see bench_${tag}.log)"
  done
}

group() {  # label cfg1 cfg2 cfg3
  local label=$1; shift
  local cfgs=("$@") n=$#
  local c tag
  for c in "${cfgs[@]}"; do
    tag=$(basename "$c" .yaml)
    echo "--- ab warmup $tag $(date -Is) ---" | tee -a "$SUM"
    run_bench "warmup_$tag" "$c"
  done
  for r in $(seq 1 $ROUNDS); do
    local off=$(( (r - 1) % n ))
    echo "=== ab ROUND $label r$r $(date -Is) loadavg=$(cut -d' ' -f1-3 /proc/loadavg) ===" | tee -a "$SUM"
    for i in $(seq 0 $((n - 1))); do
      c=${cfgs[$(( (i + off) % n ))]}
      tag="$(basename "$c" .yaml)_r${r}"
      echo "--- ab r$r $tag $(date -Is) ---" | tee -a "$SUM"
      run_bench "$tag" "$c"
    done
  done
}

group tuning_t1 v15_ab_rrf_np180_t1_tuning_off.yaml v15_ab_rrf_np180_t1_tuning_on.yaml v15_anchor_natref_np170_top55_t1_tuning.yaml
group tuning_t4 v15_ab_rrf_np180_t4_tuning_off.yaml v15_ab_rrf_np180_t4_tuning_on.yaml v15_anchor_natref_np170_top55_t4_tuning.yaml
group indep_t1 v15_ab_rrf_np180_t1_indep_off.yaml v15_ab_rrf_np180_t1_indep_on.yaml v15_anchor_natref_np170_top55_t1_indep.yaml
group indep_t4 v15_ab_rrf_np180_t4_indep_off.yaml v15_ab_rrf_np180_t4_indep_on.yaml v15_anchor_natref_np170_top55_t4_indep.yaml

echo "=== v15 reuse OFF/ON rrf A/B + natref anchor DONE $(date -Is) ===" | tee -a "$SUM"

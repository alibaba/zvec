#!/bin/bash
# v17-E3: segment-timing ON/OFF alternating bench rounds (tuning set,
# integrated np180/Top55/reuse + dedicated np170/Top55), 5 rounds.
# v17-E4: refine_stage_timing ON/OFF alternating bench rounds (refiner
# configs, tuning + indep sets), 3 rounds.
# Alternating order per round cancels drift; per-run load snapshots kept.
set -u
ROOT=/home/xingrui/bench/integrate_rabitq_in_hnsw/v17_runs
V16=$ROOT/../v16_runs
BIN=/home/xingrui/workspace/zvec/rabitq_refactor/zvec/build_bench/bin
export TMPDIR=$ROOT/tmp
export LD_LIBRARY_PATH=/home/xingrui/workspace/zvec/rabitq_refactor/zvec/build_bench/lib
mkdir -p "$TMPDIR" "$ROOT/cfg"
cd "$ROOT"

# --- generate v17 configs from the frozen v16 sources ---------------------
for side in int ded; do
  for state in on off; do
    sed -e "s|/home/xingrui/bench/integrate_rabitq_in_hnsw/v16_runs/v16_ovh_${side}_${state}.recall|$ROOT/v17_ovh_${side}_${state}.recall|" \
        "$V16/v16_ovh_${side}_${state}.yaml" > "cfg/v17_ovh_${side}_${state}.yaml"
  done
done
for side in int ded; do
  for set in tuning indep; do
    src="$V16/v16r2_ref_${side}_${set}.refined.yaml"
    for state in on off; do
      out="cfg/v17_ref_${side}_${set}_${state}.yaml"
      sed -e "s|RecallLogDir: v16r2_ref_${side}_${set}.refined.recall|RecallLogDir: $ROOT/v17_ref_${side}_${set}_${state}.recall|" "$src" > "$out"
      if [ "$state" = on ]; then
        sed -i 's|\\"is_linear\\":false,|\\"is_linear\\":false,\\"refine_stage_timing\\":true,|' "$out"
      fi
    done
  done
done

run_bench() {  # $1 yaml, $2 tag
  local yaml=$1 tag=$2
  {
    echo "### $(date +%F_%T) $tag"
    uptime
  } > "${tag}.load.log"
  /usr/bin/time -v "$BIN/bench" "$yaml" > "${tag}.log" 2> "${tag}.time.log"
  local rc=$?
  uptime >> "${tag}.load.log"
  echo "$tag rc=$rc"
  return $rc
}

# --- E3: segment timing ON/OFF, tuning set, 5 rounds ----------------------
E3_ROUNDS=${1:-5}
for r in $(seq 1 "$E3_ROUNDS"); do
  if [ $((r % 2)) -eq 1 ]; then
    order="int_off ded_off int_on ded_on"
  else
    order="ded_on int_on ded_off int_off"
  fi
  for entry in $order; do
    side=${entry%_*}
    state=${entry##*_}
    run_bench "cfg/v17_ovh_${side}_${state}.yaml" "v17_ovh_${side}_${state}_r${r}" || exit 1
  done
done

# --- E4: refine stage timing ON/OFF, both query sets, 3 rounds ------------
E4_ROUNDS=${2:-3}
for r in $(seq 1 "$E4_ROUNDS"); do
  if [ $((r % 2)) -eq 1 ]; then
    order="int_tuning_off int_tuning_on ded_tuning_off ded_tuning_on int_indep_off int_indep_on ded_indep_off ded_indep_on"
  else
    order="ded_indep_on ded_indep_off int_indep_on int_indep_off ded_tuning_on ded_tuning_off int_tuning_on int_tuning_off"
  fi
  for entry in $order; do
    name=${entry%_*}
    state=${entry##*_}
    run_bench "cfg/v17_ref_${name}_${state}.yaml" "v17_ref_${name}_${state}_r${r}" || exit 1
  done
done
echo "v17-E3/E4 bench rounds complete"

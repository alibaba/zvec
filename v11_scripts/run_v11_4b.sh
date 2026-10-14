#!/bin/bash
# v11-4b: candidate-rerank experiment in the replay.
# The integrated side runs two row types inside one invocation:
#   prod50   production pruning replica (Top-50 heap of refined scores)
#   rerankN  production pruning path collecting Top-N candidates by
#            lower_bound, then exact rerank with original vectors from
#            the .vecs store into a Top-50 result.
# Both rows reuse the recorded lists of the diagnostic run at the given
# nprobe (production probe selection preserved) and exclude probe
# selection + finalize, so totals pair within one invocation.
# SMOKE=1 runs 20 queries / 1 round / topn=60,80.
set -u
REPO=/home/xingrui/workspace/zvec/rabitq_refactor/zvec
BIN=$REPO/build_bench/bin
QUERIES=/home/xingrui/bench/dataset/cohere/1m/cohere_test_vector_1000.new.txt
INT_INDEX=/home/xingrui/bench/rabitq_refactor/ivf/cohere_1m_ivf_rabitq_norm1.index
VECS=/home/xingrui/bench/dataset/cohere/1m/cohere_train_vector_1m.new.centaur.vecs
GT=/home/xingrui/bench/dataset/cohere/1m/neighbors.txt
V8=/home/xingrui/bench/rabitq_refactor/v8
RUN=$(cd "$(dirname "$0")" && pwd)
export TMPDIR="$RUN/tmp"
mkdir -p "$TMPDIR"
cd "$RUN"

if [ "${SMOKE:-0}" = "1" ]; then
  NPROBES="${1:-293}"
  ROUNDS=1
  QCOUNT=20
  TOPNS="60,80"
  PREFIX=smoke/iv4b
else
  NPROBES="150 200 249 272 293 320 352"
  ROUNDS=5
  QCOUNT=1000
  TOPNS="60,80,100"
  PREFIX=iv4b
fi

echo "=== V11 4b $(date -Is) commit=$(git -C $REPO rev-parse HEAD) nprobes=[$NPROBES] topns=$TOPNS rounds=$ROUNDS qcount=$QCOUNT ===" | tee iv4b_summary.txt

for np in $NPROBES; do
  # Recorded lists at this nprobe via the diagnostic (production probes).
  if [ -f "$RUN/int_np${np}.yaml" ]; then
    cfg="$RUN/int_np${np}.yaml"
  else
    cfg="$V8/v8_base_np${np}.yaml"
  fi
  if [ ! -f "$cfg" ]; then
    echo "missing config for np=$np; run make_v11_nprobe_cfgs.py first" | tee -a iv4b_summary.txt
    continue
  fi
  name="${PREFIX}_np${np}"
  echo "--- diag np=$np $(date -Is) ---" | tee -a iv4b_summary.txt
  "$BIN/ivf_rabitq_scan_diagnostic" "$cfg" "./$name" \
    --export-lists > "$name.diag.log" 2>&1
  rc=$?
  echo "diag np=$np rc=$rc $(date -Is)" | tee -a iv4b_summary.txt
  grep -E "^baseline " "$name.diag.log" | tee -a iv4b_summary.txt || true
  if [ $rc -ne 0 ]; then continue; fi

  echo "--- replay np=$np topns=$TOPNS $(date -Is) ---" | tee -a iv4b_summary.txt
  "$BIN/ivf_rabitq_scan_replay" \
    --index-int="$INT_INDEX" \
    --queries="$QUERIES" \
    --lists-int="$name.lists.csv" --mode-int=baseline \
    --rerank-topn="$TOPNS" \
    --vecs="$VECS" --gt="$GT" \
    --queries-count="$QCOUNT" \
    --rounds="$ROUNDS" --warmup=1 --csv="./$name" > "$name.log" 2>&1
  rc=$?
  echo "replay np=$np rc=$rc $(date -Is)" | tee -a iv4b_summary.txt
  grep -E "^  rerank |verify:" "$name.log" | tee -a iv4b_summary.txt || true
done

echo "=== V11 4b DONE $(date -Is) ===" | tee -a iv4b_summary.txt

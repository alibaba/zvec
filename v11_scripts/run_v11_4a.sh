#!/bin/bash
# v11-4a: single-factor coarse-scan experiments in the replay.
# Each invocation pairs one integrated variant with the production rows of
# both sides inside the same invocation (immune to cross-invocation drift):
#   cluster0  rabitqlib split_batch_estdist over the block BatchDataMap with
#             cluster-0 coefficients -> isolates the per-row epilogue
#             (model-ID prefix load + g_add/g_error gather + AoS stores).
#   dedshape  cluster0 without the key-keep mask walk -> isolates the mask.
#   flat      production scanner over a page-aligned contiguous copy of the
#             scan segment -> isolates mapped-segment locality.
#   flat16    flat copy shifted +16B -> alignment sensitivity.
# Uses the pruning-free pair (mode-int=unpruned, ded_orig_unpruned) so the
# scanned work is the full unpruned sweep, 5 rounds, 1000 queries.
# SMOKE=1 runs one variant with 20 queries / 1 round.
set -u
REPO=/home/xingrui/workspace/zvec/rabitq_refactor/zvec
BIN=$REPO/build_bench/bin
QUERIES=/home/xingrui/bench/dataset/cohere/1m/cohere_test_vector_1000.new.txt
INT_INDEX=/home/xingrui/bench/rabitq_refactor/ivf/cohere_1m_ivf_rabitq_norm1.index
DED_INDEX=/home/xingrui/bench/rabitq_refactor/ivf_dedicated/cohere_1m_ivfrabitq_dedicated.index
RUN=$(cd "$(dirname "$0")" && pwd)
export TMPDIR="$RUN/tmp"
mkdir -p "$TMPDIR"
cd "$RUN"

if [ "${SMOKE:-0}" = "1" ]; then
  VARIANTS="${1:-cluster0}"
  ROUNDS=1
  QCOUNT=20
  PREFIX=smoke/iv4a
else
  VARIANTS="cluster0 dedshape flat flat16"
  ROUNDS=5
  QCOUNT=1000
  PREFIX=iv4a
fi

echo "=== V11 4a $(date -Is) commit=$(git -C $REPO rev-parse HEAD) variants=[$VARIANTS] rounds=$ROUNDS qcount=$QCOUNT ===" | tee iv4a_summary.txt

for v in $VARIANTS; do
  name="${PREFIX}_${v}"
  echo "--- variant $v $(date -Is) ---" | tee -a iv4a_summary.txt
  "$BIN/ivf_rabitq_scan_replay" \
    --index-int="$INT_INDEX" --index-ded="$DED_INDEX" \
    --queries="$QUERIES" \
    --lists-int=int_orig.lists.csv --mode-int=unpruned \
    --lists-ded=ded_orig_unpruned.lists.csv \
    --samples-int=int_orig.samples.csv --samples-ded=ded_orig_unpruned.samples.csv \
    --int-variant="$v" --queries-count="$QCOUNT" \
    --rounds="$ROUNDS" --warmup=1 --csv="./$name" > "$name.log" 2>&1
  rc=$?
  echo "variant $v rc=$rc $(date -Is)" | tee -a iv4a_summary.txt
  grep -E "^  factor |verify:" "$name.log" | tee -a iv4a_summary.txt || true
done

echo "=== V11 4a DONE $(date -Is) ===" | tee -a iv4a_summary.txt

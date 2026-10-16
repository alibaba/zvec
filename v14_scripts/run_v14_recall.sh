#!/bin/bash
# v14 native+refine recall sweep on the TUNING set (full 1000 queries).
# Usage: run_v14_recall.sh <cfg-basename>...   (yamls live next to this script)
# Records one line per config: natref_recall <name> rc=<n> Recall@50: <v>
# Fails (non-zero, FAILED: line) on rc!=0 or missing Recall@50.
set -u
REPO=/home/xingrui/workspace/zvec/rabitq_refactor/zvec
BIN=$REPO/build_bench/bin
RUN=$(cd "$(dirname "$0")" && pwd)
export TMPDIR="$RUN/tmp"
mkdir -p "$TMPDIR"
cd "$RUN"

SUM=v14_natref_recall_summary.txt
if [ ! -f "$SUM" ]; then
  echo "=== v14 natref recall sweep $(date -Is) commit=$(git -C $REPO rev-parse HEAD) ===" | tee "$SUM"
fi
fail() { echo "FAILED: $1 -- aborting" | tee -a "$SUM"; exit 1; }
[ -x "$BIN/recall" ] || fail "missing $BIN/recall"

for name in "$@"; do
  yaml="$name.yaml"
  [ -f "$yaml" ] || fail "missing $yaml"
  echo "--- recall $name $(date -Is) ---" | tee -a "$SUM"
  "$BIN/recall" "$yaml" > "recall_$name.log" 2>&1
  rc=$?
  v=$(grep -o 'Recall@50: [0-9.]*' "recall_$name.log" | tail -1 | grep -o '[0-9.]*$')
  echo "natref_recall $name rc=$rc Recall@50: ${v:-MISSING}" | tee -a "$SUM"
  [ $rc -ne 0 ] && fail "recall FAILED for $name"
  [ -z "$v" ] && fail "no Recall@50 line for $name"
done
echo "=== v14 recall batch DONE $(date -Is) ===" | tee -a "$SUM"

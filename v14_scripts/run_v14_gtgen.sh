#!/bin/bash
# v14 task 4: generate the independent-query-set ground truth (900 queries
# disjoint from the 1M base, see indep/overlap_report.txt).
# 1. Search the kFlat reference index (FP32 brute force over the 1M base)
#    on the 900 independent queries with the recall tool (no external GT:
#    the tool also builds internal GT via is_linear exact scan and must
#    report Recall@100 = 100 as a self-consistency check).
# 2. Parse the .knn logs into neighbors_indep_q10m_900.txt (qid;id1..id100,
#    ids are dataset keys, not row positions).
# 3. numpy brute-force spot check over the raw base .vecs (key-mapped).
set -u
REPO=/home/xingrui/workspace/zvec/rabitq_refactor/zvec
BIN=$REPO/build_bench/bin
RUN=$(cd "$(dirname "$0")" && pwd)
export TMPDIR="$RUN/tmp"
mkdir -p "$TMPDIR"
cd "$RUN"

SUM=v14_indep_summary.txt
fail() { echo "FAILED: $1 -- aborting" | tee -a "$SUM"; exit 1; }
[ -x "$BIN/recall" ] || fail "missing $BIN/recall"

echo "=== v14 independent GT generation $(date -Is) commit=$(git -C $REPO rev-parse HEAD) ===" | tee "$SUM"

echo "--- gtgen flat100 $(date -Is) ---" | tee -a "$SUM"
"$BIN/recall" indep/v14_gtgen_flat100.yaml > recall_v14_gtgen_flat100.log 2>&1
rc=$?
v=$(grep -o 'Recall@100: [0-9.]*' recall_v14_gtgen_flat100.log | tail -1 | grep -o '[0-9.]*$')
echo "gtgen_recall v14_gtgen_flat100 rc=$rc Recall@100: ${v:-MISSING}" | tee -a "$SUM"
[ $rc -ne 0 ] && fail "gtgen recall FAILED"
[ -z "$v" ] && fail "no Recall@100 line for gtgen"
[ "$v" != "100" ] && fail "gtgen self-consistency: Recall@100=$v != 100 (internal GT vs kFlat search disagree)"

python3 "$REPO/v14_scripts/parse_v14_gt_knn.py" \
  --logdir v14_gtgen_flat100_900.recall \
  --out indep/neighbors_indep_q10m_900.txt \
  --queries indep/queries_indep_q10m_900.txt \
  --keys 100 --check-sample 10 2>&1 | tee -a "$SUM"
[ ${PIPESTATUS[0]} -ne 0 ] && fail "GT parse/brute-force check FAILED"

echo "=== v14 independent GT generation DONE $(date -Is) ===" | tee -a "$SUM"

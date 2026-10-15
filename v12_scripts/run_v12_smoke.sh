#!/bin/bash
# v12 smoke: rerank rrf/rlb schemes + prod50 verification + NaN/Inf controls.
set -u
REPO=/home/xingrui/workspace/zvec/rabitq_refactor/zvec
BIN=$REPO/build_bench/bin
QUERIES=/home/xingrui/bench/dataset/cohere/1m/cohere_test_vector_1000.new.txt
INT_INDEX=/home/xingrui/bench/rabitq_refactor/ivf/cohere_1m_ivf_rabitq_norm1.index
VECS=/home/xingrui/bench/dataset/cohere/1m/cohere_train_vector_1m.new.centaur.vecs
GT=/home/xingrui/bench/dataset/cohere/1m/neighbors.txt
V11=/home/xingrui/bench/integrate_rabitq_in_hnsw/v11_runs
RUN=$(cd "$(dirname "$0")" && pwd)
export TMPDIR="$RUN/tmp"
mkdir -p "$TMPDIR" smoke
cd "$RUN"

echo "=== v12 smoke $(date -Is) commit=$(git -C $REPO rev-parse HEAD) ==="

# A: rerank schemes, prod50 verified against topk=50 results (20 queries).
"$BIN/ivf_rabitq_scan_replay" \
  --index-int="$INT_INDEX" --queries="$QUERIES" \
  --lists-int="$V11/iv4b_np293.lists.csv" --mode-int=baseline \
  --rerank-topn=60,80 --vecs="$VECS" --gt="$GT" \
  --results-int="$V11/iv4b_np293.results.csv" \
  --queries-count=20 --rounds=2 --warmup=1 --csv=./smoke/v12a > smoke/v12a.log 2>&1
echo "A rerank-smoke rc=$?"
grep -E '^  rerank |^results-int|verify:' smoke/v12a.log

# B/C: NaN and Inf negative controls on samples verification.
python3 - <<'EOF'
from pathlib import Path
src = Path('/home/xingrui/bench/integrate_rabitq_in_hnsw/v11_runs/int_orig.samples.csv')
lines = src.read_text().splitlines()
head, body = lines[0], lines[1:]
def mutate(value):
    out = body[:]
    f = out[3].split(',')
    f[6] = value  # distance of the 4th sample row
    out[3] = ','.join(f)
    return head + '\n' + '\n'.join(out) + '\n'
Path('smoke/nan.samples.csv').write_text(mutate('nan'))
Path('smoke/inf.samples.csv').write_text(mutate('inf'))
print('negative controls written (mutated sample row 3 distance)')
EOF
for ctl in nan inf; do
  "$BIN/ivf_rabitq_scan_replay" \
    --index-int="$INT_INDEX" --queries="$QUERIES" \
    --lists-int="$V11/int_orig.lists.csv" --mode-int=baseline \
    --rerank-topn=60 --vecs="$VECS" --gt="$GT" \
    --samples-int="smoke/$ctl.samples.csv" \
    --queries-count=1000 --rounds=1 --warmup=0 --csv="./smoke/v12_$ctl" \
    > "smoke/v12_$ctl.log" 2>&1
  echo "$ctl-control rc=$? (expect 1)"
  grep -E 'non-finite|replay failed' "smoke/v12_$ctl.log" | head -2
done

echo "=== v12 smoke DONE $(date -Is) ==="

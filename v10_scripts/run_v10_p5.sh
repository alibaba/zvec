#!/bin/bash
set -u
REPO=/home/xingrui/workspace/zvec/rabitq_refactor/zvec
BIN=$REPO/build_bench/bin
V8DIR=/home/xingrui/bench/rabitq_refactor/v8
RUN=$REPO/../local_validation/v10/p5_$(date +%Y%m%d-%H%M%S)
mkdir -p "$RUN/tmp"
export TMPDIR="$RUN/tmp"
cd "$RUN"
echo "RUN=$RUN" > RUN_DIR.txt

# TopK=100 variants of the v8 configs (both IndexCommon.TopK and the
# QueryParam topk field). Deeper heap makes GT hits that quantized ordering
# pushes to ranks 51-100 visible for the loss analysis; ranks 1-50 of the
# unpruned modes stay directly comparable to the v9 TopK=50 numbers.
for src in v8_base_np293 v8_ded_diag_np249; do
  sed -e 's/TopK: 50/TopK: 100/' \
      -e 's/"topk":50/"topk":100/' \
      -e "s#\./${src}\.recall#./v10s5_${src}.recall#" \
      "$V8DIR/${src}.yaml" > "v10s5_${src}.yaml"
done

echo "=== V10 P5 EXPORTS $(date -Is) commit=$(git -C $REPO rev-parse HEAD) ==="
"$BIN/ivf_rabitq_scan_diagnostic" v10s5_v8_base_np293.yaml ./v10s5_int \
  --export-lists > v10s5_int.log 2>&1
echo "int rc=$?"
"$BIN/ivf_rabitq_scan_diagnostic" v10s5_v8_ded_diag_np249.yaml ./v10s5_ded \
  --export-lists > v10s5_ded.log 2>&1
echo "ded rc=$?"
"$BIN/ivf_rabitq_scan_diagnostic" v10s5_v8_ded_diag_np249.yaml \
  ./v10s5_ded_unpruned --export-lists --disable-pruning \
  > v10s5_ded_unpruned.log 2>&1
echo "ded unpruned rc=$?"
echo "=== V10 P5 DONE $(date -Is) ==="

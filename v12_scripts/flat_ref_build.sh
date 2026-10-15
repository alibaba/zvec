#!/bin/bash
# Build the exact kFlat reference index for the v12 refiner experiments.
#
# The index MUST be created through the core_interface API (Index::open kMMAP
# create_new + add + close), not local_builder/FlatBuilder + FileDumper:
# FileDumper emits the package layout that only MMapFileReadStorage consumers
# (e.g. IVFIndex::open) read back, while the refiner opens the reference via
# the base Index::open -> MMapFileStorage -> IndexMapping path, which rejects
# the package layout's placeholder meta_footer_offset (Unsupported).
set -eu
BIN=/home/xingrui/workspace/zvec/rabitq_refactor/zvec/build_bench/bin
OUT=$(cd "$(dirname "$0")" && pwd)/cohere_1m_flat_ref.index
TMPDIR=$(dirname "$0")/tmp
mkdir -p "$TMPDIR"
"$BIN/flat_ref_builder" \
  --vecs=/home/xingrui/bench/dataset/cohere/1m/cohere_train_vector_1m.new.centaur.vecs \
  --index="$OUT" \
  --metric=cosine

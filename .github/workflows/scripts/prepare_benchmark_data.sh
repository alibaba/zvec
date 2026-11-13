#!/usr/bin/env bash
# One-shot local download/conversion pipeline. Does not run benchmarks or CI.
set -euo pipefail
if [[ $# != 2 ]]; then
    echo "Usage: $0 /absolute/datasets/root /absolute/gist.tar.gz" >&2
    exit 2
fi
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
DATA_ROOT=$1
GIST_ARCHIVE=$2
PYTHON_BIN=${PYTHON_BIN:-python3}
mkdir -p "$DATA_ROOT/_sources"
trap 'echo "Pipeline failed; inspect the log and rerun to resume verified downloads." >&2' ERR
if [[ ! -f "$DATA_ROOT/_sources/gist/gist_base.fvecs" ||
      ! -f "$DATA_ROOT/_sources/gist/gist_query.fvecs" ||
      ! -f "$DATA_ROOT/_sources/gist/gist_groundtruth.ivecs" ]]; then
    tar -xzf "$GIST_ARCHIVE" -C "$DATA_ROOT/_sources" \
      gist/gist_base.fvecs gist/gist_query.fvecs gist/gist_groundtruth.ivecs
fi
"$PYTHON_BIN" "$SCRIPT_DIR/prepare_local_datasets.py" --root "$DATA_ROOT" --dataset gist
"$PYTHON_BIN" "$SCRIPT_DIR/download_text2image.py" \
    --root "$DATA_ROOT/_sources/text2image1B" --workers "${DOWNLOAD_WORKERS:-12}"
"$PYTHON_BIN" "$SCRIPT_DIR/prepare_local_datasets.py" --root "$DATA_ROOT" --dataset text2image
echo "ALL THREE DATASETS PREPARED: $DATA_ROOT"

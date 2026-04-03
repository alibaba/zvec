#!/usr/bin/env bash
# Code coverage script for the zvec Rust SDK.
#
# Prerequisites:
#   1. Install cargo-llvm-cov:
#      cargo install cargo-llvm-cov
#
#   2. Set environment variables:
#      export ZVEC_LIB_DIR=/path/to/zvec/build/src/binding/c
#      export ZVEC_INCLUDE_DIR=/path/to/zvec/src/include
#
# Usage:
#   ./scripts/coverage.sh          # Generate HTML report
#   ./scripts/coverage.sh --lcov   # Generate LCOV report
#   ./scripts/coverage.sh --text   # Print text summary

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUST_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${RUST_DIR}"

# Check prerequisites
if ! command -v cargo-llvm-cov &> /dev/null; then
    echo "Error: cargo-llvm-cov is not installed."
    echo "Install it with: cargo install cargo-llvm-cov"
    exit 1
fi

if [ -z "${ZVEC_LIB_DIR:-}" ]; then
    echo "Warning: ZVEC_LIB_DIR is not set. Integration tests may fail."
fi

OUTPUT_FORMAT="${1:---html}"

case "${OUTPUT_FORMAT}" in
    --lcov)
        echo "Generating LCOV coverage report..."
        cargo llvm-cov --workspace --lcov --output-path lcov.info
        echo "Coverage report written to: ${RUST_DIR}/lcov.info"
        ;;
    --text)
        echo "Generating text coverage summary..."
        cargo llvm-cov --workspace
        ;;
    --html)
        echo "Generating HTML coverage report..."
        cargo llvm-cov --workspace --html --output-dir coverage-html
        echo "Coverage report written to: ${RUST_DIR}/coverage-html/index.html"
        ;;
    --json)
        echo "Generating JSON coverage report..."
        cargo llvm-cov --workspace --json --output-path coverage.json
        echo "Coverage report written to: ${RUST_DIR}/coverage.json"
        ;;
    *)
        echo "Usage: $0 [--html|--lcov|--text|--json]"
        exit 1
        ;;
esac

echo "Done!"

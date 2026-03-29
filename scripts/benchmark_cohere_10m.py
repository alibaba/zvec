#!/usr/bin/env python3
"""Compatibility wrapper for the generic JSON-driven HNSW vs OMEGA runner."""

import sys
from pathlib import Path

import benchmark_hnsw_vs_omega


def main() -> int:
    script_dir = Path(__file__).resolve().parent
    config_path = script_dir / "benchmark_hnsw_vs_omega.json"
    sys.argv = [
        str(script_dir / "benchmark_hnsw_vs_omega.py"),
        "--config",
        str(config_path),
        "--dataset",
        "cohere_10m",
        *sys.argv[1:],
    ]
    return benchmark_hnsw_vs_omega.main()


if __name__ == "__main__":
    sys.exit(main())

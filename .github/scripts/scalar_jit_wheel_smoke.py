"""Exercise a repaired wheel's persisted forward filters and require JIT use.

Run directly under qemu-x86_64 to test the same wheel with restricted CPUID;
this script deliberately does not launch child Python processes.
"""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import faulthandler
import json
import os
from pathlib import Path
import re
import sys
import tempfile

import numpy as np
import zvec


@contextlib.contextmanager
def capture_native_log(path: Path):
    """Capture both Python and native logging without spawning a process."""
    sys.stdout.flush()
    sys.stderr.flush()
    saved = [os.dup(1), os.dup(2)]
    with path.open("w") as output:
        try:
            os.dup2(output.fileno(), 1)
            os.dup2(output.fileno(), 2)
            yield
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            ctypes.CDLL(None).fflush(None)
            for fd, original in zip((1, 2), saved):
                os.dup2(original, fd)
                os.close(original)


def exercise(path: Path) -> int:
    zvec.init(
        log_type=zvec.LogType.CONSOLE,
        log_level=zvec.LogLevel.DEBUG,
        query_threads=1,
        optimize_threads=1,
        memory_limit_mb=512,
        invert_to_forward_scan_ratio=0.5,
    )
    rng = np.random.default_rng(889)
    vectors = rng.normal(size=(512, 8)).astype(np.float32)
    schema = zvec.CollectionSchema(
        name="scalar_jit_packaging",
        # An inverted predicate matching most rows is rewritten into the HNSW
        # forward callback. Without an index the planner can instead prepare
        # an Arrow bitmap, which would not exercise the JIT adapter.
        fields=[
            zvec.FieldSchema("value", zvec.DataType.INT32, nullable=True,
                             index_param=zvec.InvertIndexParam(enable_range_optimization=True)),
            zvec.FieldSchema("label", zvec.DataType.STRING),
        ],
        vectors=[
            zvec.VectorSchema(
                "vector", zvec.DataType.VECTOR_FP32, 8,
                index_param=zvec.HnswIndexParam(metric_type=zvec.MetricType.L2),
            )
        ],
    )
    values = [None if i % 13 == 0 else i for i in range(len(vectors))]
    writer = zvec.create_and_open(str(path), schema)
    try:
        result = writer.insert([
            zvec.Doc(
                id=f"doc_{i}",
                fields={"value": values[i], "label": f"label_{i}"},
                vectors={"vector": vector.tolist()},
            )
            for i, vector in enumerate(vectors)
        ])
        assert len(result) == len(vectors) and all(item.ok() for item in result)
        writer.flush()
        writer.optimize()
    finally:
        writer.close()
    reader = zvec.open(
        str(path), zvec.CollectionOption(read_only=True, enable_mmap=True)
    )
    checked = 0
    try:
        for query_vector in rng.normal(size=(3, 8)).astype(np.float32):
            distances = np.sum((vectors - query_vector) ** 2, axis=1)
            # Even after excluding nulls, every predicate matches >50%.
            for threshold in (16, 32, 48, 64, 96, 128, 160, 192):
                valid = [
                    i for i, value in enumerate(values)
                    if value is not None and value > threshold
                ]
                expected = sorted(valid, key=lambda i: float(distances[i]))[:8]
                result = reader.query(
                    zvec.Query(
                        field_name="vector", vector=query_vector.tolist(),
                        param=zvec.HnswQueryParam(ef=512),
                    ),
                    filter=f"value > {threshold}", topk=8,
                    output_fields=["value", "label"],
                )
                assert [doc.id for doc in result] == [f"doc_{i}" for i in expected]
                for doc, i in zip(result, expected):
                    assert doc.fields == {"value": values[i], "label": f"label_{i}"}
                checked += 1
    finally:
        reader.close()
    # Static LLVM is a packaging requirement, including when the builder has
    # an LLVM installation that could otherwise mask an external dependency.
    maps = Path("/proc/self/maps").read_text()
    assert "libLLVM" not in maps and "libclang" not in maps, maps
    return checked


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path)
    parser.add_argument("--expect-jit", choices=("on", "off"), default="on")
    parser.add_argument("--expect-avx2-only", action="store_true")
    args = parser.parse_args()
    faulthandler.enable(all_threads=True)
    with tempfile.TemporaryDirectory(prefix="zvec-scalar-jit-") as directory:
        root = Path(directory)
        log_path = args.log.resolve() if args.log else root / "native.log"
        try:
            with capture_native_log(log_path):
                queries = exercise(root / "collection")
            # Some planner DEBUG entries contain binary encoded index keys.
            log = log_path.read_text(errors="replace")
            ready = log.count("Scalar JIT forward filter ready")
            if args.expect_jit == "on":
                assert ready >= queries, f"Only {ready} JIT bindings for {queries} queries"
                assert "Detected CPU Name" in log, "Gandiva did not initialize LLVM"
                assert "Scalar JIT unavailable" not in log, "A query fell back to Arrow"
                if args.expect_avx2_only:
                    assert "+avx2" in log and "-avx512f" in log, log
                    assert "+avx512f" not in log, log
            else:
                assert ready == 0, "JIT was unexpectedly enabled"
        except BaseException:
            if log_path.exists():
                print(log_path.read_text(errors="replace"), file=sys.stderr)
            raise
        print(json.dumps({"queries": queries, "jit_bindings": ready,
                          "jit_expected": args.expect_jit,
                          "cpu_names": re.findall(r"Detected CPU Name : (\S+)", log),
                          "external_llvm_loaded": False}))


if __name__ == "__main__":
    main()

"""Exercise a repaired wheel's persisted forward filters and require JIT use.

Run directly under qemu-x86_64 to test the same wheel with restricted CPUID;
this script deliberately does not launch child Python processes.
"""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import faulthandler
import hashlib
import json
import os
import re
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np

os.environ.setdefault("ZVEC_SCALAR_JIT_PROFILE", "1")

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
            for fd, original in zip((1, 2), saved, strict=True):
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
            zvec.FieldSchema(
                "value",
                zvec.DataType.INT32,
                nullable=True,
                index_param=zvec.InvertIndexParam(enable_range_optimization=True),
            ),
            zvec.FieldSchema(
                "label",
                zvec.DataType.STRING,
                index_param=zvec.InvertIndexParam(enable_range_optimization=True),
            ),
            zvec.FieldSchema(
                "rating",
                zvec.DataType.FLOAT,
                nullable=True,
                index_param=zvec.InvertIndexParam(enable_range_optimization=True),
            ),
            zvec.FieldSchema(
                "flag",
                zvec.DataType.BOOL,
                index_param=zvec.InvertIndexParam(enable_range_optimization=True),
            ),
        ],
        vectors=[
            zvec.VectorSchema(
                "vector",
                zvec.DataType.VECTOR_FP32,
                8,
                index_param=zvec.HnswIndexParam(metric_type=zvec.MetricType.L2),
            )
        ],
    )
    values = [None if i % 13 == 0 else i for i in range(len(vectors))]
    ratings = [None if i % 17 == 0 else i / 4.0 for i in range(len(vectors))]
    flags = [i % 7 != 0 for i in range(len(vectors))]
    writer = zvec.create_and_open(str(path), schema)
    try:
        result = writer.insert(
            [
                zvec.Doc(
                    id=f"doc_{i}",
                    fields={
                        "value": values[i],
                        "label": f"label_{i}",
                        "rating": ratings[i],
                        "flag": flags[i],
                    },
                    vectors={"vector": vector.tolist()},
                )
                for i, vector in enumerate(vectors)
            ]
        )
        assert len(result) == len(vectors)
        assert all(item.ok() for item in result)
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
            predicates = [
                (
                    f"value > {threshold}",
                    lambda i, t=threshold: values[i] is not None and values[i] > t,
                )
                for threshold in (16, 32, 48, 64, 96, 128, 160, 192)
            ]
            members = ",".join(str(i) for i in range(17, 512))
            predicates += [
                (
                    "value > 16 AND (rating >= 0.0 OR value > 16)",
                    lambda i: values[i] is not None and values[i] > 16,
                ),
                (
                    "value > 16 AND (value > 32 OR value < 384)",
                    lambda i: values[i] is not None and values[i] > 16,
                ),
                (
                    "value > 16 AND label >= 'label_0'",
                    lambda i: values[i] is not None and values[i] > 16,
                ),
                (
                    "value > 16 AND label LIKE 'label_%'",
                    lambda i: values[i] is not None and values[i] > 16,
                ),
                (
                    f"value > 16 AND value IN ({members})",
                    lambda i: values[i] is not None and values[i] > 16,
                ),
            ]
            for predicate, matches in predicates:
                valid = [i for i in range(len(values)) if matches(i)]
                expected = sorted(valid, key=lambda i: float(distances[i]))[:8]
                result = reader.query(
                    zvec.Query(
                        field_name="vector",
                        vector=query_vector.tolist(),
                        param=zvec.HnswQueryParam(ef=512),
                    ),
                    filter=predicate,
                    topk=8,
                    output_fields=["value", "label"],
                )
                assert [doc.id for doc in result] == [f"doc_{i}" for i in expected]
                for doc, i in zip(result, expected, strict=True):
                    assert doc.fields == {"value": values[i], "label": f"label_{i}"}
                    assert np.isclose(doc.score, distances[i], rtol=2e-5, atol=2e-5), (
                        doc.score,
                        distances[i],
                    )
                checked += 1
    finally:
        reader.close()
    # Static LLVM is a packaging requirement, including when the builder has
    # an LLVM installation that could otherwise mask an external dependency.
    if sys.platform == "linux":
        maps = Path("/proc/self/maps").read_text()
        assert "libLLVM" not in maps, maps
        assert "libclang" not in maps, maps
    return checked


def main() -> None:  # noqa: PLR0915 - sequential acceptance checks and failure report
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path)
    parser.add_argument("--expect-jit", choices=("on", "off"), default="on")
    parser.add_argument("--expect-avx2-only", action="store_true")
    parser.add_argument("--expect-sse42-only", action="store_true")
    parser.add_argument("--artifact", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    faulthandler.enable(all_threads=True)
    with tempfile.TemporaryDirectory(prefix="zvec-scalar-jit-") as directory:
        root = Path(directory)
        log_path = args.log.resolve() if args.log else root / "native.log"
        report = {"passed": False, "jit_expected": args.expect_jit}
        try:
            extension = Path(zvec._zvec.__file__)
            report["extension_sha256"] = hashlib.sha256(
                extension.read_bytes()
            ).hexdigest()
            if args.artifact:
                report["artifact"] = args.artifact.name
                report["artifact_sha256"] = hashlib.sha256(
                    args.artifact.read_bytes()
                ).hexdigest()
                with zipfile.ZipFile(args.artifact) as wheel:
                    compiled = [
                        name
                        for name in wheel.namelist()
                        if Path(name).name == extension.name
                    ]
                    assert len(compiled) == 1
                    assert (
                        hashlib.sha256(wheel.read(compiled[0])).hexdigest()
                        == report["extension_sha256"]
                    )
            with capture_native_log(log_path):
                queries = exercise(root / "collection")
            # Some planner DEBUG entries contain binary encoded index keys.
            log = log_path.read_text(errors="replace")
            ready = log.count("Scalar JIT forward filter ready")
            info = zvec.scalar_jit_info()
            report.update(
                queries=queries,
                jit_bindings=ready,
                diagnostics=info,
                cpu_names=re.findall(r"Detected CPU Name : (\S+)", log),
            )
            cpu = info["cpu_features"]
            if args.expect_avx2_only:
                assert cpu["sse42"], cpu
                assert cpu["avx2"], cpu
                assert not cpu["avx512f"], cpu
            if args.expect_sse42_only:
                assert cpu["sse42"], cpu
                assert not any(cpu[name] for name in ("avx", "avx2", "avx512f")), cpu
            if args.expect_jit == "on":
                assert info["enabled"], info
                assert info["initialized"], info
                assert info["execution_samples"]["jit"]["count"] > 0, info
                # LIKE/IN remain non-callback plans by design. Their native
                # scalar differential tests cover generated helper execution.
                assert ready >= queries - 6, (
                    f"Only {ready} JIT bindings for {queries - 6} callback queries"
                )
                report["non_callback_cases"] = ["LIKE", "IN"]
                assert "Detected CPU Name" in log, "Gandiva did not initialize LLVM"
                assert "Scalar JIT unavailable" not in log, "A query fell back to Arrow"
                if args.expect_avx2_only:
                    assert "+avx2" in log, log
                    assert "-avx512f" in log, log
                    assert "+avx512f" not in log, log
            else:
                assert ready == 0, "JIT was unexpectedly enabled"
                assert not info["enabled"], info
                assert not info["initialized"], info
                assert info["execution_samples"]["jit"]["count"] == 0, info
                assert info["execution_samples"]["arrow"]["count"] > 0, info
            report["passed"] = True
        except BaseException as error:
            report["error"] = f"{type(error).__name__}: {error}"
            report["diagnostics"] = zvec.scalar_jit_info()
            if log_path.exists():
                sys.stderr.write(log_path.read_text(errors="replace") + "\n")
            raise
        finally:
            if args.report:
                args.report.write_text(json.dumps(report, indent=2) + "\n")
        sys.stdout.write(json.dumps(report) + "\n")


if __name__ == "__main__":
    main()

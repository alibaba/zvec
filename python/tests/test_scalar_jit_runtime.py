# Copyright 2025-present the zvec project
# SPDX-License-Identifier: Apache-2.0
"""Process-isolated JIT diagnostics and lifecycle tests (never run under QEMU)."""

from __future__ import annotations

import concurrent.futures
import importlib.util
import json
import multiprocessing
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import threading

import pytest


def load_zvec():
    # Test-only support for ON/OFF CMake trees without installing stale wheels.
    extension = os.environ.get("ZVEC_TEST_EXTENSION")
    if extension and "zvec._zvec" not in sys.modules:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        spec = importlib.util.spec_from_file_location("zvec._zvec", extension)
        native = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(native)
        sys.modules[spec.name] = native
    import zvec

    if extension:
        zvec._zvec = sys.modules["zvec._zvec"]
    return zvec


def query_threads(root):
    zvec = load_zvec()
    before = zvec.scalar_jit_info()
    assert not before["initialized"]
    zvec.init(
        query_threads=1,
        optimize_threads=1,
        memory_limit_mb=512,
        invert_to_forward_scan_ratio=0.5,
    )
    schema = zvec.CollectionSchema(
        name="jit_lifecycle",
        fields=[
            zvec.FieldSchema(
                "value",
                zvec.DataType.INT32,
                index_param=zvec.InvertIndexParam(enable_range_optimization=True),
            )
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
    writer = zvec.create_and_open(str(root), schema)
    try:
        statuses = writer.insert(
            [
                zvec.Doc(
                    id=f"doc_{i}",
                    fields={"value": i},
                    vectors={"vector": [float(i), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]},
                )
                for i in range(512)
            ]
        )
        assert all(status.ok() for status in statuses)
        writer.flush()
        writer.optimize()
    finally:
        writer.close()
    reader = zvec.open(
        str(root), zvec.CollectionOption(read_only=True, enable_mmap=True)
    )
    barrier = threading.Barrier(8)

    def query(index):
        barrier.wait(timeout=30)
        # Concurrent first preparation may bypass the in-flight compilation.
        # Repeats subsequently reuse the shape with independently owned values.
        for repeat in range(8):
            threshold = 16 + index * 8 + repeat
            result = reader.query(
                zvec.Query(
                    field_name="vector",
                    vector=[0.0] * 8,
                    param=zvec.HnswQueryParam(ef=512),
                ),
                filter=f"value > {threshold}",
                topk=4,
                output_fields=["value"],
            )
            assert [doc.id for doc in result] == [
                f"doc_{i}" for i in range(threshold + 1, threshold + 5)
            ]
            assert [doc.fields["value"] for doc in result] == list(
                range(threshold + 1, threshold + 5)
            )

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(query, range(8)))
    finally:
        # The executor joins all query threads before closing the collection.
        reader.close()
    info = zvec.scalar_jit_info()
    assert info["cache"]["active_compilations"] == 0
    assert info["memory"]["shared_snapshot_references"] == 0
    assert info["memory"]["live_bindings"] == 0
    assert info["memory"]["live_program_ir_bytes"] == 0
    if info["enabled"]:
        assert info["initialized"] and info["cache"]["hits"] > 0
        assert info["timings"]["lowering"]["count"] > 0
        assert info["memory"]["live_program_object_bytes"] > 0
    else:
        assert not info["initialized"]
        assert info["selections"]["arrow"] == 64
    samples = info["execution_samples"]
    if info["profile"]:
        assert samples["jit" if info["enabled"] else "arrow"]["count"] > 0
    else:
        assert samples["jit"]["count"] == samples["arrow"]["count"] == 0
    return info


def spawned_worker(connection):
    try:
        with tempfile.TemporaryDirectory() as root:
            connection.send(query_threads(Path(root) / "collection"))
    finally:
        connection.close()


def child_main(mode):
    zvec = load_zvec()
    if mode == "info":
        return zvec.scalar_jit_info()
    if mode == "spawn":
        # Inspecting in the parent must not initialize the child's backend.
        assert not zvec.scalar_jit_info()["initialized"]
        context = multiprocessing.get_context("spawn")
        receive, send = context.Pipe(duplex=False)
        process = context.Process(target=spawned_worker, args=(send,))
        process.start()
        send.close()
        try:
            assert receive.poll(90), "Spawn worker timed out"
            info = receive.recv()
            process.join(15)
            assert process.exitcode == 0
        finally:
            if process.is_alive():
                process.terminate()
                process.join(10)
            receive.close()
        assert not zvec.scalar_jit_info()["initialized"]
        return info
    with tempfile.TemporaryDirectory() as root:
        info = query_threads(Path(root) / "collection")
    if mode == "fork":
        pid = os.fork()
        if pid == 0:
            signal.alarm(5)
            try:
                child = zvec.scalar_jit_info()
                assert child["disable_reason"] == (
                    "forked_process" if child["built"] else "build_disabled"
                )
                assert not child["enabled"] and not child["accounting_available"]
                assert set(child["memory"]) == set(info["memory"])
                os._exit(0)
            except BaseException:
                os._exit(2)
        _, status = os.waitpid(pid, 0)
        assert os.waitstatus_to_exitcode(status) == 0
    return info


def run_child(mode, **overrides):
    env = {k: v for k, v in os.environ.items() if not k.startswith("ZVEC_SCALAR_JIT")}
    env.update(ZVEC_SCALAR_JIT="1", ZVEC_SCALAR_JIT_PROFILE="1")
    env.update(overrides)
    result = subprocess.run(
        [sys.executable, __file__, mode],
        env=env,
        capture_output=True,
        text=True,
        errors="replace",
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_diagnostics_do_not_initialize():
    info = run_child("info")
    assert info["version"] == 1
    assert not info["initialized"]
    assert not info["dump_ir"]
    assert info["enabled"] == info["built"]
    assert info["disable_reason"] == ("" if info["built"] else "build_disabled")
    assert info["limits"]["compilations"] == 2
    assert info["limits"]["live_programs"] == 256
    assert not info["memory"]["borrowed_arrow_buffers_included"]


@pytest.mark.parametrize("value", ["0", "-1", "1x", "", "184467440737095516160"])
@pytest.mark.parametrize(
    "name", ["ZVEC_SCALAR_JIT_MAX_COMPILATIONS", "ZVEC_SCALAR_JIT_MAX_LIVE_PROGRAMS"]
)
def test_invalid_configuration(name, value):
    info = run_child("info", **{name: value})
    assert not info["enabled"] and not info["initialized"]
    assert info["disable_reason"] == (
        "invalid_configuration" if info["built"] else "build_disabled"
    )


def test_effective_configuration():
    info = run_child(
        "info",
        ZVEC_SCALAR_JIT_MAX_COMPILATIONS="1",
        ZVEC_SCALAR_JIT_MAX_LIVE_PROGRAMS="7",
        ZVEC_SCALAR_JIT_DUMP_IR="1",
    )
    assert info["limits"]["compilations"] == 1
    assert info["limits"]["live_programs"] == 7
    assert info["dump_ir"]


@pytest.mark.parametrize("enabled", ["1", "0"])
@pytest.mark.parametrize("mode", ["queries", "spawn", "fork"])
def test_process_lifecycle(mode, enabled):
    if mode == "fork" and not hasattr(os, "fork"):
        pytest.skip("fork is POSIX-only")
    info = run_child(mode, ZVEC_SCALAR_JIT=enabled)
    assert info["enabled"] == (info["built"] and enabled == "1")


def test_profile_off_has_no_execution_samples():
    run_child("queries", ZVEC_SCALAR_JIT_PROFILE="0")


if __name__ == "__main__":
    if sys.argv[1] == "smoke":
        import runpy

        load_zvec()
        sys.argv = [
            str(
                Path(__file__).resolve().parents[2]
                / ".github/scripts/scalar_jit/wheel_smoke.py"
            ),
            *sys.argv[2:],
        ]
        runpy.run_path(sys.argv[0], run_name="__main__")
    else:
        sys.stdout.write(json.dumps(child_main(sys.argv[1])) + "\n")

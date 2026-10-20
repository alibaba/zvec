#!/usr/bin/env python3
"""Unit tests for the v17 calibration analyzers and sampler symbolizer.

Synthetic fixtures only; the real v17_runs logs are referenced by path in the
report, not by these tests.
"""
import importlib.util
import os
import struct
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))


def load(name):
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(HERE, name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_diag_parse_mode_line_with_substages():
    mod = load("analyze_v17_diag")
    line = ("reuse queries=1000 active=1000 coarse=201679250 refined=711349 "
            "refine_ratio=0.00352713 coarse_mean=201679 "
            "segments_us/q prep=320.24 coarse=4135.82 cand=355.464 "
            "post_scan=723.976 finalize=7.7903 query=5753.28 "
            "prep_sub_us/q select=273.292 transform=0.164473 encode=37.942 "
            "lut=0.921597 scanner=6.53272 cluster=0 timed_blocks=6387862\n")
    with tempfile.NamedTemporaryFile(
            "w", suffix=".log", delete=False,
            prefix="v17_diag_s1_r1_int_tuning.") as f:
        f.write(line)
        path = f.name
    try:
        rows = mod.parse_file(path)
    finally:
        os.unlink(path)
    assert len(rows) == 1
    stride, rnd, name, mode, vals = rows[0]
    assert (stride, rnd, name, mode) == (1, 1, "int_tuning", "reuse")
    assert vals["prep"] == 320.24 and vals["coarse"] == 4135.82
    assert vals["select"] == 273.292 and vals["encode"] == 37.942
    assert vals["cluster"] == 0 and vals["timed_blocks"] == 6387862
    assert vals["queries"] == 1000
    assert vals["coarse_rows"] == 201679.25


def test_bench_parse_qps_and_load():
    mod = load("analyze_v17_bench")
    log = ("QueryParam: {\"topk\":55}\n"
           "Process query: 3054, total process time: 12000ms\n"
           "Avg latency: 3.9ms qps: 254.4\n")
    with tempfile.TemporaryDirectory() as d:
        tag = "v17_ovh_int_on_r1"
        log_path = os.path.join(d, tag + ".log")
        with open(log_path, "w") as f:
            f.write(log)
        with open(os.path.join(d, tag + ".load.log"), "w") as f:
            f.write("12:00:00 up 1 day, load average: 2.50, 3.00, 4.00\n")
        row = mod.parse(log_path)
    assert row is not None
    phase, name, state, rnd, qps, lat, load1 = row
    assert (phase, name, state, rnd) == ("ovh", "int", "on", 1)
    assert qps == 254.4 and lat == 3.9 and load1 == "2.50"


def test_bench_parse_skips_logs_without_qps():
    mod = load("analyze_v17_bench")
    with tempfile.TemporaryDirectory() as d:
        log_path = os.path.join(d, "v17_ovh_int_on_r2.load.log")
        with open(log_path, "w") as f:
            f.write("load average: 1.0\n")
        assert mod.parse(log_path) is None


def test_sigprof_load_maps_keeps_only_executable():
    mod = load("analyze_sigprof")
    maps = ("56324c60d000-56324c799000 r-xp 0003b000 103:03 30127056 /fake/bench\n"
            "56324c799000-56324c7e5000 r--p 001c7000 103:03 30127056 /fake/bench\n"
            "7d0000000000-7d0000002000 r-xp 00046000 103:03 999 /fake/libfoo.so\n")
    with tempfile.NamedTemporaryFile("w", suffix=".maps", delete=False) as f:
        f.write(maps)
        path = f.name
    try:
        entries = mod.load_maps(path)
    finally:
        os.unlink(path)
    assert len(entries) == 2  # the r--p mapping is dropped
    assert entries[0] == (0x56324c60d000, 0x56324c799000, 0x3b000, "/fake/bench")
    assert entries[1] == (0x7d0000000000, 0x7d0000002000, 0x46000,
                          "/fake/libfoo.so")


def test_sigprof_object_bases_rebias_pie():
    mod = load("analyze_sigprof")
    entries = [
        (0x56324c60d000, 0x56324c799000, 0x3b000, "/fake/bench"),
        (0x7d0000000000, 0x7d0000002000, 0x46000, "/fake/libfoo.so"),
    ]
    bases = mod.object_bases(entries)
    assert bases["/fake/bench"] == 0x56324c60d000 - 0x3b000
    assert bases["/fake/libfoo.so"] == 0x7d0000000000 - 0x46000


def test_sigprof_resolve_pie_rebases_runtime_rip():
    mod = load("analyze_sigprof")
    base = 0x7d0000000000 - 0x46000
    entries = [(0x7d0000000000, 0x7d0000002000, 0x46000, "/fake/libfoo.so")]
    starts = [e[0] for e in entries]
    bases = {"/fake/libfoo.so": base}
    tables = {"/fake/libfoo.so": [(0x46000, "funcA"), (0x47000, "funcB")]}
    rip = base + 0x47010  # inside funcB at link time, far from raw rip
    assert mod.resolve(rip, entries, starts, bases, tables) == \
        "libfoo.so:funcB+0x10"


def test_sigprof_resolve_nonpie_uses_absolute_rip():
    mod = load("analyze_sigprof")
    entries = [(0x400000, 0x402000, 0x0, "/fake/python3.10")]
    starts = [e[0] for e in entries]
    bases = {"/fake/python3.10": 0x400000}
    tables = {"/fake/python3.10": [(0x401000, "PyMain")]}
    assert mod.resolve(0x401010, entries, starts, bases, tables) == \
        "python3.10:PyMain+0x10"


def test_sigprof_resolve_unmapped_and_symbolless():
    mod = load("analyze_sigprof")
    entries = [(0x400000, 0x402000, 0x0, "/fake/vdso")]
    starts = [e[0] for e in entries]
    assert mod.resolve(0x1000, entries, starts, {}, {}) == "[unmapped]"
    assert mod.resolve(0x401000, entries, starts,
                       {"/fake/vdso": 0x400000}, {}) == "vdso+0x1000"


def test_sigprof_load_samples_roundtrip():
    mod = load("analyze_sigprof")
    blob = struct.pack("<4Q", 1000, 0x401000, 2000, 0x401010)
    with tempfile.NamedTemporaryFile("wb", suffix=".bin", delete=False) as f:
        f.write(blob)
        path = f.name
    try:
        samples = mod.load_samples(path)
    finally:
        os.unlink(path)
    assert samples == [(1000, 0x401000), (2000, 0x401010)]


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as e:
                fails += 1
                print(f"FAIL {name}: {e}")
    print("OK" if fails == 0 else f"{fails} FAILURES")
    sys.exit(1 if fails else 0)

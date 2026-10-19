#!/usr/bin/env python3
"""Unit tests for the v16 profiling analyzer and twin verifier.

Synthetic fixtures only; the real v16_runs logs are referenced by path in the
report, not by these tests.
"""
import importlib.util
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))


def load(name):
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(HERE, name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_parse_diag_segments_and_load():
    mod = load("analyze_v16_gap")
    log = ("20:00:00 up 1 day, load average: 2.50, 3.00, 4.00\n"
           "reuse queries=10 active=10 coarse=1000 refined=100 "
           "refine_ratio=0.1 mid_text=1 "
           "segments_us/q prep=1.5 coarse=2.5 cand=0.5 post_scan=3.5 "
           "finalize=0.25 query=8.5\n")
    with tempfile.NamedTemporaryFile("w", suffix=".log", delete=False) as f:
        f.write(log)
        path = f.name
    try:
        d = mod.parse_diag(path, "reuse")
    finally:
        os.unlink(path)
    assert d["load1"] == 2.5
    assert d["queries"] == 10 and d["coarse"] == 1000 and d["refined"] == 100
    assert d["prep"] == 1.5 and d["coarse_ns"] == 2.5 and d["cand"] == 0.5
    assert d["post"] == 3.5 and d["fin"] == 0.25 and d["query"] == 8.5


def test_parse_diag_refine_stages():
    mod = load("analyze_v16_gap")
    log = "x\nrefined queries=10 stages_us/q base=100.0 refine=10.0 " \
          "wall=110.5 residual=0.5\n"
    with tempfile.NamedTemporaryFile("w", suffix=".log", delete=False) as f:
        f.write(log)
        path = f.name
    try:
        d = mod.parse_diag(path, "refined")
    finally:
        os.unlink(path)
    assert d["base"] == 100.0 and d["refine"] == 10.0
    assert d["wall"] == 110.5 and d["resid"] == 0.5


def test_fit_post_scan_recovers_rates():
    mod = load("analyze_v16_gap")
    # post = 2ns * coarse_rows + 400ns * refined_rows, per query (us)
    tuning = {"queries": 100, "coarse": 20000, "refined": 1000,
              "post": (2 * 200 + 400 * 10) / 1000.0}
    indep = {"queries": 100, "coarse": 18000, "refined": 20000,
             "post": (2 * 180 + 400 * 200) / 1000.0}
    a_us, b_us = mod.fit_post_scan(tuning, indep)
    assert abs(a_us * 1000 - 2.0) < 1e-6      # 2 ns per coarse row
    assert abs(b_us - 0.400) < 1e-6           # 400 ns per refined row


def test_twin_subset_and_counts():
    mod = load("verify_v16_twins")
    twin = {0: [(k, "1.0") for k in range(55)]}
    refined = {0: [(k, "2.0") for k in range(50)]}
    errors = []
    mod.check_counts(twin, 55, "t", errors)
    mod.check_counts(refined, 50, "r", errors)
    assert mod.check_subset(refined, twin, "s", errors) == 0
    refined_bad = {0: [(55, "2.0")]}
    assert mod.check_subset(refined_bad, twin, "s", errors) == 1
    assert errors and "not in twin" in errors[-1]


def test_round_trip_identical_detects_diff():
    mod = load("verify_v16_twins")
    errors = []
    assert mod.check_identical({0: [(1, "1")]}, {0: [(1, "1")]}, "x", errors)
    assert not mod.check_identical({0: [(1, "1")]}, {0: [(2, "1")]}, "x",
                                   errors)


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

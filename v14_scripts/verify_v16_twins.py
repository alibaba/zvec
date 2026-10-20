#!/usr/bin/env python3
"""Verify v16 twin runs against refined runs and cross-round determinism.

Twin = no-refiner TopK=55 run of the identical base computation used by the
production refiner pipeline's base stage (coarse_topk = 50 * 1.1 = 55).
Checks per (set, side, round):
  1. twin has exactly 55 rows per query, refined has exactly 50;
  2. every refined final top-50 key is inside the twin's top-55 key set;
  3. twin results are bit-identical across rounds r1 vs r2 (same mode);
  4. refined results are bit-identical across rounds r1 vs r2.
Integrated twin uses mode "reuse" (reuse ON, matching production); dedicated
uses mode "dedicated".
"""
import csv
import sys
from collections import defaultdict

BASE = "/home/xingrui/bench/integrate_rabitq_in_hnsw/v16_runs"
SETS = ["tuning", "indep"]
ROUNDS = {"r1": "v16_", "r2": "v16r2_"}
SIDES = {"int": "reuse", "ded": "dedicated"}


def load(path, mode):
    per_query = defaultdict(list)
    with open(path) as f:
        for row in csv.DictReader(f):
            if row["mode"] == mode:
                per_query[int(row["query"])].append((int(row["key"]),
                                                      row["score"]))
    return per_query


def check_counts(per_query, expect, label, errors):
    for q, rows in per_query.items():
        if len(rows) != expect:
            errors.append(f"{label}: query {q} has {len(rows)} rows, "
                          f"expected {expect}")
    return len(per_query)


def check_subset(refined, twin, label, errors):
    bad = 0
    for q, rows in refined.items():
        twin_keys = {k for k, _ in twin[q]}
        missing = [k for k, _ in rows if k not in twin_keys]
        if missing:
            bad += 1
            if bad <= 3:
                errors.append(f"{label}: query {q} refined keys not in twin: "
                              f"{missing[:5]}")
    return bad


def check_identical(a, b, label, errors):
    if a == b:
        return True
    diffs = 0
    for q in sorted(set(a) | set(b)):
        if a.get(q) != b.get(q):
            diffs += 1
            if diffs <= 3:
                errors.append(f"{label}: query {q} differs between rounds")
    errors.append(f"{label}: {diffs} queries differ between rounds")
    return False


def main():
    errors = []
    summary = []
    for st in SETS:
        for side, mode in SIDES.items():
            twins = {}
            refineds = {}
            for rnd, prefix in ROUNDS.items():
                twin = load(f"{BASE}/{prefix}prof_{side}_{st}.results.csv", mode)
                ref = load(f"{BASE}/{prefix}ref_{side}_{st}.results.csv",
                           "refined")
                n_t = check_counts(twin, 55, f"{prefix}prof_{side}_{st}",
                                   errors)
                n_r = check_counts(ref, 50, f"{prefix}ref_{side}_{st}", errors)
                if n_t != n_r:
                    errors.append(f"{prefix}{side}_{st}: twin has {n_t} "
                                  f"queries, refined has {n_r}")
                bad = check_subset(ref, twin, f"{prefix}{side}_{st}", errors)
                summary.append(f"{st:6s} {side} r{rnd[-1]}: queries={n_t} "
                               f"subset_violations={bad}")
                twins[rnd] = twin
                refineds[rnd] = ref
            check_identical(twins["r1"], twins["r2"],
                            f"prof_{side}_{st} twin determinism", errors)
            check_identical(refineds["r1"], refineds["r2"],
                            f"ref_{side}_{st} refined determinism", errors)
    for line in summary:
        print(line)
    if errors:
        print("FAILED:")
        for e in errors:
            print("  " + e)
        return 1
    print("ALL TWIN VERIFICATIONS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Aggregate v17-E1/E2 diagnostic logs (prep sub-stages, stride validation).

Usage: analyze_v17_diag.py [GLOB ...]   (default: v17_diag_*.log)

Parses the per-mode summary lines of ivf_rabitq_scan_diagnostic runs,
groups them by (config, stride, round, mode) and prints per-query stage
means plus the derived per-block rates. For stride 8 the coarse/cand/
post_scan segments cover only every 8th block, so per-block rates are
re-derived as segment / (timed_blocks / queries * stride) and compared
against the stride-1 caliber of the same round.
"""
import glob
import re
import sys
from collections import defaultdict

MODE_LINE = re.compile(
    r"^(baseline|unpruned|reuse|reuse_unpruned|dedicated) queries=(\d+) "
    r"active=\d+ coarse=(\d+) refined=(\d+) refine_ratio=\S+.*?"
    r"segments_us/q (.+)$")

FIELDS = [
    "prep", "coarse", "cand", "post_scan", "finalize", "query",
]


def parse_tail(tail):
    out = {}
    for kv in tail.split():
        if "=" in kv:
            k, v = kv.split("=", 1)
            out[k] = float(v)
    return out


def parse_file(path):
    m = re.match(r"v17_diag_s(\d+)_r(\d+)_(\w+)", path.rsplit("/", 1)[-1])
    if not m:
        return []
    stride, rnd, name = int(m.group(1)), int(m.group(2)), m.group(3)
    rows = []
    for line in open(path):
        line = line.strip()
        mm = MODE_LINE.match(line)
        if not mm:
            continue
        mode, queries, coarse, refined = mm.group(1), int(mm.group(2)), \
            int(mm.group(3)), int(mm.group(4))
        vals = parse_tail(mm.group(5))
        vals["queries"] = queries
        vals["coarse_rows"] = coarse / queries
        vals["refined_rows"] = refined / queries
        rows.append((stride, rnd, name, mode, vals))
    return rows


def main():
    paths = sys.argv[1:] or sorted(glob.glob("v17_diag_*.log"))
    rows = []
    for p in paths:
        rows.extend(parse_file(p))
    if not rows:
        print("no diagnostic logs found")
        return

    groups = defaultdict(list)
    for stride, rnd, name, mode, vals in rows:
        groups[(name, stride, mode)].append((rnd, vals))

    print("%-11s %-6s %-14s %3s" % ("config", "stride", "mode", "rnd") +
          "".join("%10s" % f for f in FIELDS) +
          "%9s %9s %9s %9s %9s %9s %9s %11s" % (
              "sel", "trns", "enc", "lut", "scan", "clst", "t_blocks",
              "blk/q"))
    for (name, stride, mode), items in sorted(groups.items()):
        for rnd, v in sorted(items):
            tb = v.get("timed_blocks", 0)
            blocks_per_q = tb / v["queries"] * stride
            print("%-11s %-6d %-14s %3d" % (name, stride, mode, rnd) +
                  "".join("%9.1f" % v.get(f, 0.0) for f in FIELDS) +
                  "%9.1f %9.1f %9.1f %9.1f %9.1f %9.1f %9d %11.1f" % (
                      v.get("select", 0.0), v.get("transform", 0.0),
                      v.get("encode", 0.0), v.get("lut", 0.0),
                      v.get("scanner", 0.0), v.get("cluster", 0.0),
                      int(tb), blocks_per_q))

    print("\n== per-block rates (us/block): segment / blocks_per_query ==")
    print("%-11s %-6s %-14s %9s %9s %9s" % (
        "config", "stride", "mode", "cand/ns", "coarse/ns", "post/ns"))
    for (name, stride, mode), items in sorted(groups.items()):
        for rnd, v in sorted(items):
            tb = v.get("timed_blocks", 0)
            if not tb:
                continue
            timed_per_q = tb / v["queries"]
            full_per_q = timed_per_q * stride
            print("%-11s %-6d %-14s %3d %9.3f %9.3f %9.3f" % (
                name, stride, mode, rnd,
                v.get("cand", 0.0) / timed_per_q,
                v.get("coarse", 0.0) / timed_per_q,
                v.get("post_scan", 0.0) / timed_per_q) +
                "   (blocks/q=%.1f timed=%.1f)" % (full_per_q, timed_per_q))


if __name__ == "__main__":
    main()

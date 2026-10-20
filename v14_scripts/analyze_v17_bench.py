#!/usr/bin/env python3
"""Aggregate v17-E3/E4 bench QPS logs into ON/OFF comparison tables.

Usage: analyze_v17_bench.py [PATTERN ...]
Defaults: v17_ovh_*.log (E3) and v17_ref_*.log (E4).
Parses the "Avg latency: X ms qps: Y" line plus per-run load snapshots.
"""
import glob
import re
import sys
from collections import defaultdict

QPS = re.compile(r"Avg latency: ([\d.]+)ms qps: ([\d.]+)")
NAME = re.compile(r"v17_(ovh|ref)_(\w+)_(on|off)_r(\d+)\.log$")


def parse(path):
    m = NAME.match(path.rsplit("/", 1)[-1])
    if not m:
        return None
    phase, name, state, rnd = m.group(1), m.group(2), m.group(3), int(m.group(4))
    qps = None
    lat = None
    for line in open(path):
        mm = QPS.search(line)
        if mm:
            lat = float(mm.group(1))
            qps = float(mm.group(2))
            break
    load = ""
    load_path = path[:-4] + ".load.log"
    try:
        txt = open(load_path).read()
        loads = re.findall(r"load average: ([\d.]+)", txt)
        if loads:
            load = loads[-1]
    except OSError:
        pass
    if qps is None:
        return None
    return phase, name, state, rnd, qps, lat, load


def main():
    pats = sys.argv[1:] or ["v17_ovh_*.log", "v17_ref_*.log"]
    paths = []
    for p in pats:
        paths.extend(glob.glob(p))
    rows = [r for r in (parse(p) for p in sorted(paths)) if r]
    if not rows:
        print("no bench logs found")
        return

    for phase in ("ovh", "ref"):
        sub = [r for r in rows if r[0] == phase]
        if not sub:
            continue
        print("\n== %s (segment timing ON/OFF) ==" % phase)
        print("%-12s %3s %10s %10s %8s %10s %8s %8s" % (
            "config", "rnd", "off_qps", "on_qps", "d_pct",
            "off_lat", "on_lat", "load"))
        agg = defaultdict(lambda: defaultdict(dict))
        for _, name, state, rnd, qps, lat, load in sub:
            agg[name][rnd][state] = (qps, lat, load)
        deltas = []
        for name in sorted(agg):
            for rnd in sorted(agg[name]):
                d = agg[name][rnd]
                if "on" not in d or "off" not in d:
                    continue
                off_q, off_l, _ = d["off"]
                on_q, on_l, load = d["on"]
                pct = (off_q - on_q) / off_q * 100.0
                deltas.append((name, pct))
                print("%-12s %3d %10.1f %10.1f %8.2f %10.2f %8.2f %8s" % (
                    name, rnd, off_q, on_q, pct, off_l, on_l, load))
        if deltas:
            by_name = defaultdict(list)
            for name, pct in deltas:
                by_name[name].append(pct)
            print("-- mean ON/OFF QPS loss per config --")
            for name in sorted(by_name):
                v = by_name[name]
                print("%-12s rounds=%d mean_loss=%.2f%% (min %.2f max %.2f)" % (
                    name, len(v), sum(v) / len(v), min(v), max(v)))


if __name__ == "__main__":
    main()

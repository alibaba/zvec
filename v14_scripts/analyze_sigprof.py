#!/usr/bin/env python3
"""Symbolize and aggregate v17 SIGPROF leaf-PC samples.

Usage: analyze_sigprof.py SAMPLES.bin [--tail-secs N] [--top N]

Reads the (monotonic ns, rip) sample pairs written by sigprof_sampler.so,
the accompanying SAMPLES.bin.maps snapshot, resolves each rip against the
mapped objects with nm, and prints the leaf-function sample distribution
overall and for the last --tail-secs seconds (the steady bench window).
"""
import argparse
import bisect
import os
import struct
import subprocess
import sys
from collections import Counter


def load_samples(path):
    data = open(path, "rb").read()
    n = len(data) // 16
    samples = struct.unpack("<%dQ" % (2 * n), data)
    return [(samples[2 * i], samples[2 * i + 1]) for i in range(n)]


def load_maps(path):
    entries = []
    for line in open(path):
        parts = line.split()
        if len(parts) < 6 or "x" not in parts[1]:
            continue
        start, end = (int(v, 16) for v in parts[0].split("-"))
        offset = int(parts[2], 16)
        entries.append((start, end, offset, parts[5]))
    entries.sort()
    return entries


def object_bases(maps_entries):
    """Runtime load bias per object: min(start - offset) over its mappings.

    For PIE objects (shared libs and our PIE bench binary) this is the
    runtime address of link-time vaddr 0, so link_vaddr = rip - base.
    """
    bases = {}
    for start, _end, offset, path in maps_entries:
        b = start - offset
        if path not in bases or b < bases[path]:
            bases[path] = b
    return bases


def symbolize(maps_entries, prefix):
    """Build per-object (sorted addrs, (addr, size, name) rows) from nm -S."""
    tables = {}
    for _start, _end, _offset, path in maps_entries:
        if path in tables:
            continue
        syms = []
        if os.path.exists(path):
            for flags in (["nm", "-C", "-S", "--defined-only", path],
                          ["nm", "-C", "-D", "-S", "--defined-only", path]):
                try:
                    out = subprocess.run(
                        flags, capture_output=True, text=True,
                        timeout=120).stdout
                except Exception as exc:  # nm may fail on stripped objects
                    print("nm failed for %s: %s" % (path, exc), file=sys.stderr)
                    continue
                for line in out.splitlines():
                    toks = line.split(None, 3)
                    if len(toks) == 4 and toks[2] in "tTwW":
                        try:
                            syms.append((int(toks[0], 16),
                                         int(toks[1], 16), toks[3]))
                        except ValueError:
                            pass
                if syms:
                    break
        syms = sorted(set(syms))
        tables[path] = ([s[0] for s in syms], syms)
    return tables


def resolve(rip, maps_entries, starts, bases, tables):
    i = bisect.bisect_right(starts, rip) - 1
    if i < 0 or not (maps_entries[i][0] <= rip < maps_entries[i][1]):
        return "[unmapped]"
    start, _end, _offset, path = maps_entries[i]
    obj = os.path.basename(path)
    entry = tables.get(path)
    if not entry:
        return "unknown"
    addrs, syms = entry
    cands = []
    for cand in (rip - bases.get(path, 0), rip):
        if cand not in cands:
            cands.append(cand)
    for cand in cands:
        j = bisect.bisect_right(addrs, cand) - 1
        if j < 0:
            continue
        addr, size, name = syms[j]
        if size and addr <= cand < addr + size:
            return "%s:%s+0x%x" % (obj, name, cand - addr)
    return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("samples")
    ap.add_argument("--tail-secs", type=float, default=12.0)
    ap.add_argument("--top", type=int, default=25)
    args = ap.parse_args()

    samples = load_samples(args.samples)
    if not samples:
        print("no samples")
        return
    maps_path = args.samples + ".maps"
    maps_entries = load_maps(maps_path)
    starts = [e[0] for e in maps_entries]
    bases = object_bases(maps_entries)
    tables = symbolize(maps_entries, "")

    t_end = samples[-1][0]
    t_cut = t_end - int(args.tail_secs * 1e9)
    overall = Counter()
    tail = Counter()
    total_wall = (samples[-1][0] - samples[0][0]) / 1e9
    for ns, rip in samples:
        sym = resolve(rip, maps_entries, starts, bases, tables)
        overall[sym] += 1
        if ns >= t_cut:
            tail[sym] += 1

    n_total = len(samples)
    n_tail = sum(tail.values())
    print("samples=%d wall_s=%.1f tail_s=%.1f tail_samples=%d"
          % (n_total, total_wall, args.tail_secs, n_tail))
    print("\n== top %d overall ==" % args.top)
    for sym, c in overall.most_common(args.top):
        print("%6.2f%% %7d  %s" % (100.0 * c / n_total, c, sym))
    print("\n== top %d in tail %.0fs ==" % (args.top, args.tail_secs))
    for sym, c in tail.most_common(args.top):
        print("%6.2f%% %7d  %s" % (100.0 * c / max(n_tail, 1), c, sym))
    print("\n== top %d functions in tail %.0fs ==" % (args.top, args.tail_secs))
    fn_tail = Counter()
    for sym, c in tail.items():
        fn = sym.split("+0x")[0]
        fn_tail[fn] += c
    for fn, c in fn_tail.most_common(args.top):
        print("%6.2f%% %7d  %s" % (100.0 * c / max(n_tail, 1), c, fn))


if __name__ == "__main__":
    main()

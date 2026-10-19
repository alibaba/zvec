#!/usr/bin/env python3
"""v16 same-caliber two-sided profiling analyzer.

Inputs (all under v16_runs/): the 16 diagnostic runs (2 rounds x 8: twin
--segment-timing and refined --refine-stages, per side per query set) plus the
8 overhead-control bench logs (bench t1, segment timing on/off, per side).

Method notes (mirrored from the user's requirements):
- Whole-query truth comes from the refined runs: instrumentation there is two
  clock pairs per query (<0.1%), so base/refine/wall/residual are clean.
- Twin runs carry segment instrumentation measured at ~14% whole-query cost,
  symmetric on both sides (overhead control), so cross-side RATIOS and per-unit
  rate differences are usable while absolute values are inflated ~1.16x.
- Total time is never divided by scan volume to get "kernel cost": per-unit
  rates are computed per segment, and post_scan is split into a per-coarse-row
  term (alpha) and per-refined-row term (beta) by fitting the two query sets
  (tuning = low refine mix, indep = high refine mix) per side per round.
- nprobe differs by design (180 integrated vs 170 native, frozen configs), so
  workload is compared through measured counters (coarse rows, refined rows),
  never through nprobe equality.
- Attribution uses dedicated per-unit rates as the counterfactual: "what would
  the integrated base cost if it ran native-grade per-unit rates on its own
  measured workload". Terms are additive by construction in the twin domain;
  a separate rate-difference reconciliation quantifies fit noise.
"""
import re
import sys

BASE = "/home/xingrui/bench/integrate_rabitq_in_hnsw/v16_runs"

TWIN = {  # (round, side, set) -> (log, mode)
    ("r1", "int", "tuning"): ("v16_prof_int_tuning.diag.log", "reuse"),
    ("r1", "ded", "tuning"): ("v16_prof_ded_tuning.diag.log", "dedicated"),
    ("r2", "int", "tuning"): ("v16r2_prof_int_tuning.diag.log", "reuse"),
    ("r2", "ded", "tuning"): ("v16r2_prof_ded_tuning.diag.log", "dedicated"),
    ("r1", "int", "indep"): ("v16_prof_int_indep.diag.log", "reuse"),
    ("r1", "ded", "indep"): ("v16_prof_ded_indep.diag.log", "dedicated"),
    ("r2", "ded", "indep"): ("v16r2_prof_ded_indep.diag.log", "dedicated"),
    ("r2", "int", "indep"): ("v16r2_prof_int_indep.diag.log", "reuse"),
}
REF = {  # (round, side, set) -> log
    ("r1", "int", "tuning"): "v16_ref_int_tuning.diag.log",
    ("r1", "ded", "tuning"): "v16_ref_ded_tuning.diag.log",
    ("r2", "int", "tuning"): "v16r2_ref_int_tuning.diag.log",
    ("r2", "ded", "tuning"): "v16r2_ref_ded_tuning.diag.log",
    ("r1", "int", "indep"): "v16_ref_int_indep.diag.log",
    ("r1", "ded", "indep"): "v16_ref_ded_indep.diag.log",
    ("r2", "ded", "indep"): "v16r2_ref_ded_indep.diag.log",
    ("r2", "int", "indep"): "v16r2_ref_int_indep.diag.log",
}


def parse_diag(path, mode):
    text = open(path).read()
    load = re.search(r"load average: ([\d.]+), ([\d.]+), ([\d.]+)", text)
    seg = re.search(
        rf"{mode} queries=(\d+) active=\d+ coarse=(\d+) refined=(\d+) "
        r"refine_ratio=([\d.e-]+).*?"
        r"segments_us/q prep=([\d.e+-]+) coarse=([\d.e+-]+) "
        r"cand=([\d.e+-]+) post_scan=([\d.e+-]+) "
        r"finalize=([\d.e+-]+) query=([\d.e+-]+)", text)
    stg = re.search(
        r"stages_us/q base=([\d.e+-]+) refine=([\d.e+-]+) "
        r"wall=([\d.e+-]+) residual=([\d.e+-]+)", text)
    out = {"load1": float(load.group(1)) if load else None}
    if seg:
        out.update(queries=int(seg.group(1)), coarse=int(seg.group(2)),
                   refined=int(seg.group(3)),
                   prep=float(seg.group(5)), coarse_ns=float(seg.group(6)),
                   cand=float(seg.group(7)), post=float(seg.group(8)),
                   fin=float(seg.group(9)), query=float(seg.group(10)))
    if stg:
        out.update(base=float(stg.group(1)), refine=float(stg.group(2)),
                   wall=float(stg.group(3)), resid=float(stg.group(4)))
    return out


def parse_bench(path):
    text = open(path).read()
    qps = re.search(r"qps: ([\d.]+)", text)
    return float(qps.group(1)) if qps else None


def fit_post_scan(d_tuning, d_indep):
    """post_us = a * coarse_rows + b * refined_rows (two equations)."""
    c1, r1, p1 = d_tuning["coarse"] / d_tuning["queries"], \
        d_tuning["refined"] / d_tuning["queries"], d_tuning["post"]
    c2, r2, p2 = d_indep["coarse"] / d_indep["queries"], \
        d_indep["refined"] / d_indep["queries"], d_indep["post"]
    det = c1 * r2 - c2 * r1
    b_ns = (c1 * p2 - c2 * p1) / det * 1000.0        # ns per refined row
    a_ns = (r2 * p1 - r1 * p2) / det * 1000.0        # ns per coarse row
    return a_ns / 1000.0, b_ns / 1000.0              # us per row per query


def fmt(x, nd=1):
    return f"{x:.{nd}f}"


def main():
    twins = {k: parse_diag(f"{BASE}/{v[0]}", v[1]) for k, v in TWIN.items()}
    refs = {k: parse_diag(f"{BASE}/{v}", "refined") for k, v in REF.items()}

    print("=" * 78)
    print("A. Overhead control (bench t1, tuning set, timing ON vs OFF)")
    print("   instrumentation = counters + per-batch segment clocks")
    ovh = {}
    for side in ("int", "ded"):
        for rnd in ("r1", "r2"):
            off = parse_bench(f"{BASE}/bench_v16_ovh_{side}_off_{rnd}.log")
            on = parse_bench(f"{BASE}/bench_v16_ovh_{side}_on_{rnd}.log")
            ovh[(side, rnd)] = (off, on)
            print(f"   {side} r{rnd[-1]}: off {off:.1f} QPS, on {on:.1f} QPS, "
                  f"time inflation +{(off / on - 1) * 100:.1f}%")
    for side in ("int", "ded"):
        off = sum(ovh[(side, r)][0] for r in ("r1", "r2")) / 2
        on = sum(ovh[(side, r)][1] for r in ("r1", "r2")) / 2
        print(f"   {side} mean: off {off:.1f}, on {on:.1f} QPS -> "
              f"+{(off / on - 1) * 100:.1f}% time")
    print("   refine-stages mode: 2 clock pairs/query, <0.1% (not measured "
          "separately)")

    print("=" * 78)
    print("B. Clean stage split (refined runs, negligible instrumentation)")
    print("   [us/query]                base    refine   wall   residual  "
          "load1")
    for st in ("tuning", "indep"):
        for rnd in ("r1", "r2"):
            for side in ("int", "ded"):
                d = refs[(rnd, side, st)]
                print(f"   {st:6s} {side} r{rnd[-1]}: {fmt(d['base'],1):>9s}"
                      f" {fmt(d['refine'],1):>9s} {fmt(d['wall'],1):>9s}"
                      f" {fmt(d['resid'],2):>9s}  {d['load1']}")
        i2, d2 = refs[("r2", "int", st)], refs[("r2", "ded", st)]
        i1, d1 = refs[("r1", "int", st)], refs[("r1", "ded", st)]
        print(f"   {st} base ratio: r1 {i1['base'] / d1['base']:.3f} "
              f"r2 {i2['base'] / d2['base']:.3f}; "
              f"wall ratio r2 {i2['wall'] / d2['wall']:.3f}")
        print(f"   {st} base gap r2 = {i2['base'] - d2['base']:.1f} us/q "
              f"(wall gap {i2['wall'] - d2['wall']:.1f}); "
              f"refine gap {i2['refine'] - d2['refine']:.1f} us/q")

    print("=" * 78)
    print("C. Twin segment tables (instrumented, ~14% whole-query inflation,")
    print("   symmetric across sides; ratios usable, absolutes inflated)")
    print("   [us/query]  load  prep   coarse  cand   post    fin   query  "
          "unseg | coarse_rows/q  refined/q")
    for st in ("tuning", "indep"):
        for rnd in ("r1", "r2"):
            for side in ("int", "ded"):
                d = twins[(rnd, side, st)]
                unseg = d["query"] - (d["prep"] + d["coarse_ns"] + d["cand"] +
                                      d["post"] + d["fin"])
                print(f"   {st:6s} {side} r{rnd[-1]}: {d['load1']:4.2f} "
                      f"{fmt(d['prep']):>6s} {fmt(d['coarse_ns']):>8s} "
                      f"{fmt(d['cand']):>6s} {fmt(d['post']):>8s} "
                      f"{fmt(d['fin'],2):>7s} {fmt(d['query']):>8s} "
                      f"{fmt(unseg):>6s} | {d['coarse'] // d['queries']:>11,d} "
                      f"{d['refined'] // d['queries']:>10,d}")

    print("=" * 78)
    print("D. Workload counters (deterministic; identical across rounds)")
    for st in ("tuning", "indep"):
        i, d = twins[("r2", "int", st)], twins[("r2", "ded", st)]
        cq_i = i["coarse"] / i["queries"]
        cq_d = d["coarse"] / d["queries"]
        rq_i = i["refined"] / i["queries"]
        rq_d = d["refined"] / d["queries"]
        print(f"   {st}: coarse rows/q int {cq_i:,.0f} vs ded {cq_d:,.0f} "
              f"(+{(cq_i / cq_d - 1) * 100:.1f}%); "
              f"refined/q int {rq_i:,.0f} vs ded {rq_d:,.0f} "
              f"(+{(rq_i / rq_d - 1) * 100:.1f}%)")

    print("=" * 78)
    print("E. Per-unit rates (twin domain; coarse = coarse_ns/coarse_rows;")
    print("   post_scan fitted alpha (per coarse row) + beta (per refined))")
    fits = {}
    for rnd in ("r1", "r2"):
        for side in ("int", "ded"):
            a, b = fit_post_scan(twins[(rnd, side, "tuning")],
                                 twins[(rnd, side, "indep")])
            fits[(rnd, side)] = (a, b)
            for st in ("tuning", "indep"):
                d = twins[(rnd, side, st)]
                rate = d["coarse_ns"] * 1000.0 / (d["coarse"] / d["queries"])
                print(f"   r{rnd[-1]} {side} {st:6s}: coarse "
                      f"{rate:5.2f} ns/vec", end="")
            print(f" | post fit alpha {a * 1000:4.2f} ns/coarse-row, "
                  f"beta {b:6.1f} us/refined-row")

    print("=" * 78)
    print("F. Base-gap attribution (twin domain, r2 = balanced load rounds,")
    print("   counterfactual: integrated workload at dedicated per-unit rates)")
    for st in ("tuning", "indep"):
        i, d = twins[("r2", "int", st)], twins[("r2", "ded", st)]
        a_i, b_i = fits[("r2", "int")]
        a_d, b_d = fits[("r2", "ded")]
        cq_i = i["coarse"] / i["queries"]
        cq_d = d["coarse"] / d["queries"]
        rq_i = i["refined"] / i["queries"]
        rq_d = d["refined"] / d["queries"]
        rate_i = i["coarse_ns"] * 1000.0 / cq_i
        rate_d = d["coarse_ns"] * 1000.0 / cq_d
        gap = i["query"] - d["query"]
        # workload terms at dedicated rates
        w_coarse = (cq_i - cq_d) * rate_d / 1000.0
        w_refined = (rq_i - rq_d) * b_d
        w_alpha = (cq_i - cq_d) * a_d
        # kernel/prep/cand terms on integrated volumes
        k_coarse = cq_i * (rate_i - rate_d) / 1000.0
        k_alpha = cq_i * (a_i - a_d)
        k_beta = rq_i * (b_i - b_d)
        prep = i["prep"] - d["prep"]
        cand = i["cand"] - d["cand"]
        fin = i["fin"] - d["fin"]
        unseg = (i["query"] - (i["prep"] + i["coarse_ns"] + i["cand"] +
                               i["post"] + i["fin"])) - \
                (d["query"] - (d["prep"] + d["coarse_ns"] + d["cand"] +
                               d["post"] + d["fin"]))
        total = (w_coarse + w_refined + w_alpha + k_coarse + k_alpha + k_beta +
                 prep + cand + fin + unseg)
        print(f"   {st}: twin query gap {fmt(gap)} us/q (clean base gap "
              f"{fmt(refs[('r2','int',st)]['base'] - refs[('r2','ded',st)]['base'])} us/q)")
        print(f"     workload: coarse-volume  {fmt(w_coarse):>7s} us "
              f"({w_coarse / gap * 100:4.1f}%)")
        print(f"     workload: refined-count  {fmt(w_refined):>7s} us "
              f"({w_refined / gap * 100:4.1f}%)  <-- pruning quality")
        print(f"     workload: post per-row   {fmt(w_alpha):>7s} us "
              f"({w_alpha / gap * 100:4.1f}%)")
        print(f"     kernel:   coarse/vec     {fmt(k_coarse):>7s} us "
              f"({k_coarse / gap * 100:4.1f}%)")
        print(f"     kernel:   post per-row   {fmt(k_alpha):>7s} us "
              f"({k_alpha / gap * 100:4.1f}%)")
        print(f"     kernel:   post per-ref   {fmt(k_beta):>7s} us "
              f"({k_beta / gap * 100:4.1f}%)")
        print(f"     framework: prep          {fmt(prep):>7s} us "
              f"({prep / gap * 100:4.1f}%)")
        print(f"     framework: cand staging  {fmt(cand):>7s} us "
              f"({cand / gap * 100:4.1f}%)  (integrated only)")
        print(f"     other:    finalize/unseg {fmt(fin + unseg):>7s} us "
              f"({(fin + unseg) / gap * 100:4.1f}%)")
        print(f"     sum {fmt(total)} us vs gap {fmt(gap)} us (exact by")
        print(f"     construction; real uncertainty is in the per-unit rate")
        print(f"     inputs: single runs with load noise, est. +/-5-10%)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
